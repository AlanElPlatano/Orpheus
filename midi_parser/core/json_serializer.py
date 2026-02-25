"""
JSON serializer module for MIDI tokenization output.

This module handles the final serialization step, combining all processing
results into standardized JSON format according to the specification.
"""

import json
import numpy as np
import gzip
import logging
import re
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime

from miditoolkit import MidiFile

from midi_parser.config.defaults import (
    MidiParserConfig,
    OutputConfig,
    DEFAULT_CONFIG
)
from midi_parser.core.midi_loader import MidiMetadata, ValidationResult
from midi_parser.core.track_analyzer import TrackInfo
from midi_parser.core.tokenizer_manager import TokenizationResult
from midi_parser.core.token_reorderer import reorder_bar_tokens

logger = logging.getLogger(__name__)

class NumpyEncoder(json.JSONEncoder):
    """
    Custom JSON encoder that handles NumPy data types.
    
    Converts NumPy integers, floats, arrays, and other types to native Python types.
    """
    def default(self, obj):
        # Handle NumPy integers
        if isinstance(obj, (np.integer, np.int8, np.int16, np.int32, np.int64)):
            return int(obj)
        
        # Handle NumPy floats
        if isinstance(obj, (np.floating, np.float16, np.float32, np.float64)):
            return float(obj)
        
        # Handle NumPy arrays
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        
        # Handle NumPy booleans
        if isinstance(obj, np.bool_):
            return bool(obj)
        
        # Let the base class handle everything else
        return super().default(obj)

# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class SerializationResult:
    """Result of JSON serialization process."""
    success: bool = True
    output_path: Optional[Path] = None
    file_size_bytes: int = 0
    compressed: bool = False
    error_message: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/reporting."""
        return {
            "success": self.success,
            "output_path": str(self.output_path) if self.output_path else None,
            "file_size_bytes": self.file_size_bytes,
            "compressed": self.compressed,
            "error_message": self.error_message,
            "warnings": self.warnings
        }


@dataclass
class ProcessingMetadata:
    """Metadata about the processing pipeline execution."""
    processing_timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    parser_version: str = "2.0"
    processing_time_seconds: float = 0.0
    validation_passed: bool = True
    quality_score: Optional[float] = None
    warnings: List[str] = field(default_factory=list)


# ============================================================================
# Key Detection and Analysis
# ============================================================================

def detect_key_from_midi(midi: MidiFile) -> Tuple[str, float]:
    """
    Detect the musical key from MIDI data using simple heuristics.
    
    Args:
        midi: MidiFile object
        
    Returns:
        Tuple of (key_string, confidence)
    """
    # Check for explicit key signature
    if midi.key_signature_changes:
        key_sig = midi.key_signature_changes[0]
        return key_sig.key_name, 0.9 # Confidence of 90% if the key comes straight from the midi

    # From here on, we're manually calculating the key signature of the file
    # This implementation has several flaws, i plan on redoing this later on
    
    # Fallback to pitch class analysis
    pitch_classes = [0] * 12
    total_duration = 0
    
    for inst in midi.instruments:
        if inst.is_drum:
            continue # Skip drum tracks for key analysis
        for note in inst.notes:
            pitch_class = note.pitch % 12
            duration = note.end - note.start
            pitch_classes[pitch_class] += duration
            total_duration += duration
    
    if total_duration == 0:
        return "C", 0.2  # Default with low confidence
    
    # Normalize pitch classes
    for i in range(12):
        pitch_classes[i] /= total_duration
    
    # Simple major/minor detection based on common patterns
    major_keys = ["C", "G", "D", "A", "E", "B", "F#", "Db", "Ab", "Eb", "Bb", "F"]
    minor_keys = ["Am", "Em", "Bm", "F#m", "C#m", "G#m", "D#m", "Bbm", "Fm", "Cm", "Gm", "Dm"]
    
    # Find most prominent pitch class
    max_pitch_class = pitch_classes.index(max(pitch_classes))
    
    # Map to key (simplified - real key detection would be more complex)
    pitch_to_major = ["C", "Db", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]
    pitch_to_minor = ["Cm", "C#m", "Dm", "Ebm", "Em", "Fm", "F#m", "Gm", "G#m", "Am", "Bbm", "Bm"]
    
    # Check for minor third interval prominence (simplified minor detection)
    minor_third_idx = (max_pitch_class + 3) % 12
    major_third_idx = (max_pitch_class + 4) % 12
    
    if pitch_classes[minor_third_idx] > pitch_classes[major_third_idx] * 1.2:
        key = pitch_to_minor[max_pitch_class]
        confidence = 0.6
    else:
        key = pitch_to_major[max_pitch_class]
        confidence = 0.7
    
    return key, confidence


def calculate_average_tempo(tempo_changes: List[Dict[str, Union[int, float]]], duration_ticks: int) -> float:
    """
    Calculate weighted average tempo across the piece.
    
    Args:
        tempo_changes: List of tempo change events
        duration_ticks: Total duration in ticks
        
    Returns:
        Average tempo in BPM
    """
    if not tempo_changes:
        return 120.0  # Default MIDI tempo
    
    if len(tempo_changes) == 1:
        return tempo_changes[0]["bpm"]
    
    # Calculate weighted average by duration
    weighted_sum = 0.0
    total_weight = 0.0
    
    for i, tempo_change in enumerate(tempo_changes):
        start_tick = tempo_change["tick"]
        if i + 1 < len(tempo_changes):
            end_tick = tempo_changes[i + 1]["tick"]
        else:
            end_tick = duration_ticks
        
        duration = end_tick - start_tick
        if duration > 0:
            weighted_sum += tempo_change["bpm"] * duration
            total_weight += duration
    
    if total_weight > 0:
        return round(weighted_sum / total_weight, 1)
    
    return tempo_changes[0]["bpm"]


# ============================================================================
# File Naming Functions
# ============================================================================

def sanitize_filename(name: str, max_length: int = 50) -> str:
    """
    Sanitize a string for use in filenames.
    
    Args:
        name: Original string
        max_length: Maximum length for the sanitized string
        
    Returns:
        Sanitized string safe for filenames
    """
    # Remove or replace problematic characters
    sanitized = re.sub(r'[<>:"/\\|?*]', '_', name)
    sanitized = re.sub(r'[\s]+', '_', sanitized)  # Replace spaces
    sanitized = re.sub(r'[^\w\-_]', '', sanitized)  # Keep only alphanumeric, dash, underscore
    sanitized = re.sub(r'_+', '_', sanitized)  # Collapse multiple underscores
    sanitized = sanitized.strip('_')  # Remove leading/trailing underscores
    
    # Limit length
    if len(sanitized) > max_length:
        sanitized = sanitized[:max_length]
    
    # Ensure not empty
    if not sanitized:
        sanitized = "unnamed"
    
    return sanitized.lower()


def sanitize_key_for_filename(key: str) -> str:
    """
    Convert musical key notation to filename-safe format.
    
    Converts sharps (#) and flats (b) to full word equivalents:
    - C# -> Csharp
    - Bb -> Bflat
    - F#m -> Fsharpm
    - Bbm -> Bflatm
    
    Args:
        key: Musical key string (e.g., "C#", "Bbm", "F# major")
        
    Returns:
        Filename-safe key string
    
    Examples:
        >>> sanitize_key_for_filename("C#")
        'Csharp'
        >>> sanitize_key_for_filename("Bbm")
        'Bflatm'
        >>> sanitize_key_for_filename("F# minor")
        'Fsharpm'
    """
    if not key:
        return "C"
    
    # Remove common suffixes first to process them separately
    is_minor = False
    key_clean = key.strip()
    
    # Check for minor key indicators
    if key_clean.endswith('m') or 'minor' in key_clean.lower():
        is_minor = True
        key_clean = key_clean.replace('minor', '').replace('Minor', '').strip()
        # Remove trailing 'm' if it's there (but keep it for Am, Bm, etc.)
        if key_clean.endswith('m') and len(key_clean) > 1:
            key_clean = key_clean[:-1]
    
    # Remove 'major' suffix if present
    key_clean = key_clean.replace('major', '').replace('Major', '').strip()
    
    # Replace sharps with 'sharp'
    if '#' in key_clean:
        key_clean = key_clean.replace('#', 'sharp')
    
    # Replace flats with 'flat'
    # Handle both 'b' and '♭' symbols
    if '♭' in key_clean:
        key_clean = key_clean.replace('♭', 'flat')
    elif 'b' in key_clean and len(key_clean) > 1:
        # Only replace 'b' if it's after a note letter (A-G)
        # This handles cases like "Bb" but not standalone "B"
        result = ""
        for i, char in enumerate(key_clean):
            if char == 'b' and i > 0 and key_clean[i-1].upper() in 'ABCDEFG':
                result += 'flat'
            else:
                result += char
        key_clean = result
    
    # Add minor suffix back if needed
    if is_minor:
        key_clean += 'm'
    
    # Ensure first letter is uppercase, rest lowercase
    if key_clean:
        key_clean = key_clean[0].upper() + key_clean[1:].lower()
    
    # Fallback to C if something went wrong
    if not key_clean or not key_clean[0].isalpha():
        key_clean = "C"
    
    return key_clean


def generate_output_filename(
    original_path: Path,
    key: str,
    avg_tempo: float,
    tokenization: str,
    config: OutputConfig
) -> str:
    """
    Generate output filename according to specification.
    
    Format: {KEY}-{AVG_TEMPO}bpm-{TOKENIZATION}-{sanitized_title}.json
    
    Args:
        original_path: Original MIDI file path
        key: Detected musical key
        avg_tempo: Average tempo in BPM
        tokenization: Tokenization strategy used
        config: Output configuration
        
    Returns:
        Generated filename
    """
    # Extract title from original filename
    title = original_path.stem
    sanitized_title = sanitize_filename(title, max_length=40)
    
    # Sanitize key for filename (handles sharps and flats)
    sanitized_key = sanitize_key_for_filename(key)
    
    # Round tempo
    tempo_str = str(int(round(avg_tempo)))
    
    # Build filename from template
    template = config.file_naming_template
    filename = template.format(
        key=sanitized_key,
        tempo=tempo_str,
        tokenization=tokenization.lower(),
        title=sanitized_title
    )
    
    # Ensure total filename length is reasonable
    if len(filename) > config.max_filename_length:
        # Shorten the title part
        excess = len(filename) - config.max_filename_length
        if excess < len(sanitized_title):
            sanitized_title = sanitized_title[:-excess]
            filename = template.format(
                key=sanitized_key,
                tempo=tempo_str,
                tokenization=tokenization.lower(),
                title=sanitized_title
            )
    
    return filename


# ============================================================================
# JSON Serializer Class
# ============================================================================

class JSONSerializer:
    """
    Main JSON serializer for MIDI tokenization output.
    
    Combines all processing results into standardized JSON format
    according to the specification.
    """
    
    def __init__(self, config: Optional[MidiParserConfig] = None):
        """
        Initialize the JSON serializer.
        
        Args:
            config: Parser configuration (uses default if not provided)
        """
        self.config = config or DEFAULT_CONFIG
        self.output_config = self.config.output
    
    def create_output_json(
        self,
        midi_path: Path,
        midi: MidiFile,
        metadata: MidiMetadata,
        track_infos: List[TrackInfo],
        tokenization_results: List[TokenizationResult],
        validation: Optional[ValidationResult] = None,
        processing_metadata: Optional[ProcessingMetadata] = None
    ) -> Dict[str, Any]:
        """
        Create complete JSON output structure.
        
        This is the main function referenced in Section 4, Step 4.
        
        Args:
            midi_path: Original MIDI file path
            midi: MidiFile object
            metadata: Extracted MIDI metadata
            track_infos: Track analysis results
            tokenization_results: Tokenization results for each track
            validation: Optional validation results
            processing_metadata: Optional processing metadata
            
        Returns:
            Complete JSON structure as dictionary
        """
        # Detect key for filename
        key, key_confidence = detect_key_from_midi(midi)
        
        # Calculate average tempo
        avg_tempo = calculate_average_tempo(
            metadata.tempo_changes,
            metadata.duration_ticks
        )
        
        # Get primary tokenization strategy
        primary_strategy = (tokenization_results[0].tokenization_strategy 
                          if tokenization_results else self.config.tokenization)
        
        # Build tokenizer configuration
        tokenizer_config = {
            "pitch_range": list(self.config.tokenizer.pitch_range),
            "beat_resolution": self.config.tokenizer.beat_resolution,
            "num_velocities": self.config.tokenizer.num_velocities,
            "additional_tokens": dict(self.config.tokenizer.additional_tokens),
            "max_seq_length": self.config.tokenizer.max_seq_length
        }
        
        # Build metadata section
        json_metadata = {
            "ppq": metadata.ppq,
            "tempo_changes": metadata.tempo_changes,
            "time_signatures": metadata.time_signatures,
            "duration_seconds": round(metadata.duration_seconds, 2),
            "duration_ticks": metadata.duration_ticks,
            "track_count": metadata.track_count,
            "note_count": metadata.note_count
        }
        
        # Add optional metadata
        if metadata.key_signature:
            json_metadata["key_signature"] = metadata.key_signature
        else:
            json_metadata["key_signature"] = key
            json_metadata["key_confidence"] = round(key_confidence, 3)
        
        # Always include has_lyrics flag, regardless of if it's true or not
        json_metadata["has_lyrics"] = metadata.has_lyrics

        if metadata.has_markers:
            json_metadata["has_markers"] = True
        if metadata.instrument_programs:
            json_metadata["instrument_programs"] = metadata.instrument_programs
        
        # Build tracks section
        tracks = []
        global_tokens = []
        total_sequence_length = 0

        # Extract global tokens and apply bar-level reordering
        raw_vocabulary = None
        if tokenization_results and tokenization_results[0].success:
            global_tokens = tokenization_results[0].tokens
            raw_vocabulary = tokenization_results[0].vocabulary

            if raw_vocabulary:
                global_tokens, raw_vocabulary = reorder_bar_tokens(
                    global_tokens, raw_vocabulary
                )

            total_sequence_length = len(global_tokens)
            logger.info(f"Using global token sequence: {total_sequence_length} tokens")

        for i, track_info in enumerate(track_infos):
            track_data = {
                "index": track_info.index,
                "name": track_info.name or f"Track_{track_info.index}",
                "program": track_info.program,
                "is_drum": track_info.is_drum,
                "type": track_info.type,
                "note_count": track_info.statistics.total_notes
            }
            
            # Add optional track data
            if track_info.subtype:
                track_data["subtype"] = track_info.subtype
            if track_info.confidence < 0.9:  # Only include if not highly confident
                track_data["type_confidence"] = round(track_info.confidence, 3)
            if track_info.language_hints:
                track_data["language_hints"] = track_info.language_hints
            
            # Add track statistics if verbose mode
            if self.output_config.include_vocabulary:
                track_data["statistics"] = track_info.statistics.to_dict()

            tracks.append(track_data)
        
        # Build main JSON structure
        json_data = {
            "version": processing_metadata.parser_version if processing_metadata else "2.0",
            "source_file": str(midi_path.name),
            "tokenization": primary_strategy,
            "tokenizer_config": tokenizer_config,
            "metadata": json_metadata,
            "tracks": tracks,
            "global_tokens": global_tokens,  # Store one sequence for entire midi
            "sequence_length": total_sequence_length
        }
        
        # Add vocabulary if configured (use reordered vocabulary if available)
        if self.output_config.include_vocabulary:
            if raw_vocabulary:
                json_data["vocabulary"] = raw_vocabulary
                json_data["vocabulary_size"] = len(raw_vocabulary)
            elif tokenization_results:
                for result in tokenization_results:
                    if result.vocabulary:
                        json_data["vocabulary"] = result.vocabulary
                        json_data["vocabulary_size"] = len(result.vocabulary)
                        break
        
        # Add processing metadata if available
        if processing_metadata:
            json_data["processing"] = {
                "timestamp": processing_metadata.processing_timestamp,
                "processing_time_seconds": round(processing_metadata.processing_time_seconds, 3),
                "validation_passed": processing_metadata.validation_passed
            }
            if processing_metadata.quality_score is not None:
                json_data["processing"]["quality_score"] = round(processing_metadata.quality_score, 4)
            if processing_metadata.warnings:
                json_data["processing"]["warnings"] = processing_metadata.warnings
        
        # Add validation information if available
        if validation:
            json_data["validation"] = {
                "is_valid": validation.is_valid,
                "errors": validation.errors,
                "warnings": validation.warnings
            }
        
        # Add file hash for deduplication
        if metadata.file_hash:
            json_data["file_hash"] = metadata.file_hash
        
        return json_data
    
    def serialize_to_file(
        self,
        json_data: Dict[str, Any],
        output_dir: Path,
        original_path: Path,
        compress: Optional[bool] = None
    ) -> SerializationResult:
        """
        Serialize JSON data to file with optional compression.
        
        Args:
            json_data: JSON data dictionary
            output_dir: Output directory path
            original_path: Original MIDI file path
            compress: Whether to compress (uses config if None)
            
        Returns:
            SerializationResult with status and file information
        """
        result = SerializationResult()
        
        try:
            # Ensure output directory exists
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate filename
            key = json_data["metadata"].get("key_signature", "C")
            if isinstance(key, dict):  # Handle key with confidence
                key = key.get("key", "C")
            
            avg_tempo = calculate_average_tempo(
                json_data["metadata"]["tempo_changes"],
                json_data["metadata"]["duration_ticks"]
            )
            
            filename = generate_output_filename(
                original_path,
                key,
                avg_tempo,
                json_data["tokenization"],
                self.output_config
            )
            
            # Determine compression
            use_compression = compress if compress is not None else self.output_config.compress_json
            
            if use_compression:
                # Change file extension
                filename = filename.replace('.json', '.json.gz') 
                output_path = output_dir / filename
                
                # Serialize and compress
                json_str = self._serialize_json(json_data)
                compressed_data = gzip.compress(json_str.encode('utf-8'), compresslevel=6)
                
                with open(output_path, 'wb') as f:
                    f.write(compressed_data)
                
                result.compressed = True
                result.file_size_bytes = len(compressed_data)
            else:
                output_path = output_dir / filename
                
                # Serialize to file
                json_str = self._serialize_json(json_data)
                
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(json_str)
                
                result.compressed = False
                result.file_size_bytes = len(json_str.encode('utf-8'))
            
            result.output_path = output_path
            result.success = True
            
            logger.info(f"Successfully serialized to {output_path.name} "
                       f"({'compressed' if result.compressed else 'uncompressed'}, "
                       f"{result.file_size_bytes / 1024:.1f}KB)")
            
        except Exception as e:
            logger.error(f"Failed to serialize JSON: {e}")
            result.success = False
            result.error_message = str(e)
        
        return result
    
    def _serialize_json(self, data: Dict[str, Any]) -> str:
        """
        Serialize dictionary to JSON string with configured formatting.
        Handles NumPy types automatically.
        
        Args:
            data: Dictionary to serialize
            
        Returns:
            JSON string
        """
        if self.output_config.pretty_print:
            return json.dumps(
                data, 
                indent=2, 
                sort_keys=True, 
                ensure_ascii=False,
                cls=NumpyEncoder  # Use custom encoder
            )
        else:
            return json.dumps(
                data, 
                separators=(',', ':'), 
                sort_keys=True, 
                ensure_ascii=False,
                cls=NumpyEncoder  # Use custom encoder
            )
    
    def batch_serialize(
        self,
        results: List[Tuple[Path, Dict[str, Any]]],
        output_dir: Path,
        parallel: bool = False
    ) -> List[SerializationResult]:
        """
        Serialize multiple JSON outputs efficiently.
        
        Args:
            results: List of (original_path, json_data) tuples
            output_dir: Output directory
            parallel: Whether to use parallel processing
            
        Returns:
            List of SerializationResult objects
        """
        serialization_results = []
        
        # If parallel mode is enabled and there is more than one file to serialize
        if parallel and len(results) > 1:
            import concurrent.futures
            import multiprocessing
            
            # Determine how many worker threads to use
            # Either a configured limit or fallback to the number of CPU cores
            max_workers = self.config.processing.max_workers or multiprocessing.cpu_count()
            
            # Create a thread pool for concurrent execution
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = []

                # Submit each serialization task to the thread pool
                for original_path, json_data in results:
                    future = executor.submit(
                        self.serialize_to_file, # function to run in worker thread
                        json_data, 
                        output_dir,
                        original_path
                    )
                    futures.append(future)
                
                # Collect results as each thread finishes
                for future in concurrent.futures.as_completed(futures):
                    try:
                        # Wait for result (timeout protects against hanging threads)
                        result = future.result(timeout=30)
                        serialization_results.append(result)
                    except Exception as e:
                        # If something fails, log error and create a failed result
                        logger.error(f"Batch serialization error: {e}")
                        serialization_results.append(SerializationResult(
                            success=False,
                            error_message=str(e)
                        ))
        else:
            # Sequential fallback: process one file at a time
            for original_path, json_data in results:
                result = self.serialize_to_file(json_data, output_dir, original_path)
                serialization_results.append(result)
        
        # Log a summary: how many succeeded and total size of serialized files
        successful = sum(1 for r in serialization_results if r.success)
        total_size = sum(r.file_size_bytes for r in serialization_results if r.success)
        
        logger.info(f"Batch serialization complete: {successful}/{len(results)} successful, "
                   f"total size: {total_size / (1024 * 1024):.1f}MB")
        
        return serialization_results


# ============================================================================
# Utility Functions
# ============================================================================

def load_tokenized_json(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load a tokenized JSON file (with automatic decompression).
    
    Args:
        file_path: Path to JSON or JSON.gz file
        
    Returns:
        Loaded JSON data as dictionary
        
    Raises:
        FileNotFoundError: If file doesn't exist
        json.JSONDecodeError: If JSON is invalid
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"Tokenized file not found: {file_path}")
    
    # If it has the gz file extension
    if file_path.suffix == '.gz' or str(file_path).endswith('.json.gz'):
        # Compressed file
        with gzip.open(file_path, 'rt', encoding='utf-8') as f:
            data = json.load(f)
    else:
        # Regular JSON
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    
    return data


def validate_json_schema(json_data: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate that JSON data conforms to the specification schema.
    
    Args:
        json_data: JSON data to validate
        
    Returns:
        Tuple of (is_valid, list of issues)
    """
    issues = []
    
    # Check required top-level fields
    required_fields = ["version", "source_file", "tokenization", "metadata", "tracks"]
    for field in required_fields:
        if field not in json_data:
            issues.append(f"Missing required field: {field}")
    
    # Validate metadata structure
    if "metadata" in json_data:
        metadata = json_data["metadata"]
        required_metadata = ["ppq", "tempo_changes", "time_signatures", "duration_seconds"]
        for field in required_metadata:
            if field not in metadata:
                issues.append(f"Missing required metadata field: {field}")
    
    # Validate tracks structure
    if "tracks" in json_data:
        tracks = json_data["tracks"]
        if not isinstance(tracks, list):
            issues.append("Tracks must be a list")
        else:
            for i, track in enumerate(tracks):
                if not isinstance(track, dict):
                    issues.append(f"Track {i} must be a dictionary")
                    continue
                
                required_track_fields = ["name", "program", "is_drum", "type", "tokens"]
                for field in required_track_fields:
                    if field not in track:
                        issues.append(f"Track {i} missing required field: {field}")
    
    # Validate tokenizer config if present
    if "tokenizer_config" in json_data:
        config = json_data["tokenizer_config"]
        if "pitch_range" in config:
            if not isinstance(config["pitch_range"], list) or len(config["pitch_range"]) != 2:
                issues.append("Invalid pitch_range in tokenizer_config")
    
    is_valid = len(issues) == 0
    return is_valid, issues


def create_error_output(
    file_path: Path,
    error: Exception,
    error_type: str = "processing_error"
) -> Dict[str, Any]:
    """
    Create an error output JSON for failed processing.
    
    Args:
        file_path: Path to the file that failed
        error: Exception that occurred
        error_type: Type of error for categorization
        
    Returns:
        Error JSON structure
    """
    return {
        "version": "2.0",
        "source_file": str(file_path.name),
        "error": {
            "type": error_type,
            "message": str(error),
            "file_path": str(file_path)
        },
        "processing": {
            "timestamp": datetime.utcnow().isoformat(),
            "success": False
        }
    }


# ============================================================================
# Main Entry Point
# ============================================================================

def create_output_json(
    midi_path: Path,
    midi: MidiFile,
    metadata: MidiMetadata,
    track_infos: List[TrackInfo],
    tokenization_results: List[TokenizationResult],
    output_dir: Path,
    config: Optional[MidiParserConfig] = None,
    validation: Optional[ValidationResult] = None,
    processing_metadata: Optional[ProcessingMetadata] = None
) -> SerializationResult:
    """
    Main entry point for creating and saving tokenized JSON output.
    
    This is the primary function referenced in Section 4, Step 4 of the spec document.
    
    Args:
        midi_path: Original MIDI file path
        midi: Processed MidiFile object
        metadata: Extracted metadata
        track_infos: Track analysis results
        tokenization_results: Tokenization results
        output_dir: Output directory for JSON file
        config: Optional configuration
        validation: Optional validation results
        processing_metadata: Optional processing metadata
        
    Returns:
        SerializationResult with output status
        
    Example:
        >>> result = create_output_json(
        ...     midi_path, midi, metadata, track_infos, 
        ...     token_results, output_dir
        ... )
        >>> if result.success:
        >>>     print(f"Saved to {result.output_path}")
    """
    config = config or DEFAULT_CONFIG
    serializer = JSONSerializer(config)
    
    # Create JSON structure
    json_data = serializer.create_output_json(
        midi_path,
        midi,
        metadata,
        track_infos,
        tokenization_results,
        validation,
        processing_metadata
    )
    
    # Serialize to file
    result = serializer.serialize_to_file(
        json_data,
        output_dir,
        midi_path
    )
    
    return result


# Export main classes and functions
__all__ = [
    'JSONSerializer',
    'SerializationResult',
    'ProcessingMetadata',
    'create_output_json',
    'load_tokenized_json',
    'validate_json_schema',
    'create_error_output',
    'detect_key_from_midi',
    'calculate_average_tempo',
    'generate_output_filename',
    'sanitize_filename',
]