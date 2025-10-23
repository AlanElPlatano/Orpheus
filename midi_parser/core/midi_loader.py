"""
MIDI file loading and validation module.

This module provides functionality for loading MIDI files, extracting metadata,
and performing validation checks using miditoolkit.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List, Union
from dataclasses import dataclass, field
import hashlib

import miditoolkit
from miditoolkit import MidiFile

from midi_parser.config.defaults import (
    MidiParserConfig,
    ProcessingConfig,
    ERROR_HANDLING_STRATEGIES,
    DEFAULT_CONFIG
)

logger = logging.getLogger(__name__)


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class MidiMetadata:
    """Structured metadata extracted from a MIDI file."""
    ppq: int  # Pulses per quarter note (ticks per beat)
    tempo_changes: List[Dict[str, Union[int, float]]] = field(default_factory=list)
    time_signatures: List[Dict[str, int]] = field(default_factory=list)
    key_signature: Optional[str] = None
    duration_seconds: float = 0.0
    duration_ticks: int = 0
    track_count: int = 0
    note_count: int = 0
    instrument_programs: List[int] = field(default_factory=list)
    has_lyrics: bool = False
    has_markers: bool = False
    file_size_mb: float = 0.0
    file_hash: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary for JSON serialization."""
        return {
            "ppq": self.ppq,
            "tempo_changes": self.tempo_changes,
            "time_signatures": self.time_signatures,
            "key_signature": self.key_signature,
            "duration_seconds": self.duration_seconds,
            "duration_ticks": self.duration_ticks,
            "track_count": self.track_count,
            "note_count": self.note_count,
            "instrument_programs": self.instrument_programs,
            "has_lyrics": self.has_lyrics,
            "has_markers": self.has_markers,
            "file_size_mb": self.file_size_mb,
            "file_hash": self.file_hash
        }


@dataclass
class ValidationResult:
    """Results from MIDI file validation."""
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    def add_error(self, message: str) -> None:
        """Add an error message and mark as invalid."""
        self.errors.append(message)
        self.is_valid = False
    
    def add_warning(self, message: str) -> None:
        """Add a warning message without affecting validity."""
        self.warnings.append(message)


# ============================================================================
# MIDI Loading Functions
# ============================================================================

def load_midi_file(file_path: Union[str, Path]) -> Optional[MidiFile]:
    """
    Load a MIDI file using miditoolkit.
    
    Args:
        file_path: Path to the MIDI file
        
    Returns:
        MidiFile object if successful, None if failed
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file is not a valid MIDI file
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"MIDI file not found: {file_path}")
    
    if not file_path.suffix.lower() in ['.mid', '.midi']:
        raise ValueError(f"File is not a MIDI file: {file_path}")
    
    try:
        midi = MidiFile(str(file_path))
        logger.debug(f"Successfully loaded MIDI file: {file_path.name}")
        return midi
    except Exception as e:
        logger.error(f"Failed to load MIDI file {file_path.name}: {e}")
        
        # Check error handling strategy
        strategy = ERROR_HANDLING_STRATEGIES.get("corrupted_midi", "skip")
        if strategy == "skip":
            return None
        else:
            raise


def extract_metadata(midi: MidiFile, file_path: Optional[Path] = None) -> MidiMetadata:
    """
    Extract comprehensive metadata from a MIDI file.
    
    Args:
        midi: MidiFile object
        file_path: Optional path for file-specific metadata
        
    Returns:
        MidiMetadata object with extracted information
    """
    metadata = MidiMetadata(ppq=midi.ticks_per_beat)
    
    # Extract tempo changes
    for tempo in midi.tempo_changes:
        metadata.tempo_changes.append({
            "tick": tempo.time,
            "bpm": tempo.tempo
        })
    
    # Add default tempo if none specified
    if not metadata.tempo_changes:
        metadata.tempo_changes.append({
            "tick": 0,
            "bpm": 120.0  # Standard MIDI default
        })
    
    # Extract time signatures
    for ts in midi.time_signature_changes:
        metadata.time_signatures.append({
            "tick": ts.time,
            "numerator": ts.numerator,
            "denominator": ts.denominator
        })
    
    # Add default time signature if none specified
    if not metadata.time_signatures:
        metadata.time_signatures.append({
            "tick": 0,
            "numerator": 4,
            "denominator": 4
        })
    
    # Extract key signatures
    if midi.key_signature_changes:
        # Use the first key signature as primary
        key = midi.key_signature_changes[0]
        metadata.key_signature = f"{key.key_name}"
    
    # Calculate durations
    metadata.duration_ticks = midi.max_tick
    metadata.duration_seconds = midi.get_tick_to_time_mapping()[-1] if midi.max_tick > 0 else 0.0
    
    # Count tracks and notes
    metadata.track_count = len(midi.instruments)
    metadata.note_count = sum(len(inst.notes) for inst in midi.instruments)
    
    # Extract instrument programs (excluding drums)
    programs = set()
    for inst in midi.instruments:
        if not inst.is_drum:
            programs.add(inst.program)
    metadata.instrument_programs = sorted(list(programs))
    
    # Always set has_lyrics to False because token sequences contain no lyric tokens, only musical events
    metadata.has_lyrics = False
    
    # Check for markers
    metadata.has_markers = bool(midi.markers)
    
    # File-specific metadata
    if file_path and file_path.exists():
        metadata.file_size_mb = file_path.stat().st_size / (1024 * 1024)
        
        # Calculate file hash for deduplication
        with open(file_path, 'rb') as f:
            metadata.file_hash = hashlib.md5(f.read()).hexdigest()
    
    return metadata


# ============================================================================
# Validation Functions
# ============================================================================

def validate_midi_structure(midi: MidiFile, config: ProcessingConfig) -> ValidationResult:
    """
    Validate the structure and content of a MIDI file.
    
    Args:
        midi: MidiFile object to validate
        config: Processing configuration with validation thresholds
        
    Returns:
        ValidationResult object with validation status and messages
    """
    result = ValidationResult(is_valid=True)
    
    # Check for empty MIDI
    if not midi.instruments:
        result.add_error("MIDI file contains no tracks/instruments")
        return result
    
    # Check duration limits
    duration_seconds = midi.get_tick_to_time_mapping()[-1] if midi.max_tick > 0 else 0.0
    
    if duration_seconds <= 0:
        result.add_error("MIDI file has zero duration")
    elif duration_seconds > config.max_duration_seconds:
        strategy = ERROR_HANDLING_STRATEGIES.get("extreme_duration", "truncate")
        if strategy == "truncate":
            result.add_warning(f"MIDI duration ({duration_seconds:.1f}s) exceeds limit "
                             f"({config.max_duration_seconds}s) - will be truncated")
        else:
            result.add_error(f"MIDI duration ({duration_seconds:.1f}s) exceeds maximum "
                           f"({config.max_duration_seconds}s)")
    
    # Check for valid notes
    total_notes = 0
    invalid_notes = 0
    
    for inst in midi.instruments:
        for note in inst.notes:
            total_notes += 1
            
            # Check note validity
            if note.pitch < 0 or note.pitch > 127:
                invalid_notes += 1
                result.add_warning(f"Invalid note pitch: {note.pitch}")
            
            if note.velocity < 0 or note.velocity > 127:
                invalid_notes += 1
                result.add_warning(f"Invalid note velocity: {note.velocity}")
            
            if note.end <= note.start:
                invalid_notes += 1
                result.add_warning(f"Invalid note duration: start={note.start}, end={note.end}")
    
    if total_notes == 0:
        result.add_error("MIDI file contains no notes")
    elif invalid_notes > 0:
        error_ratio = invalid_notes / total_notes
        if error_ratio > 0.1:  # More than 10% invalid notes
            result.add_error(f"Too many invalid notes: {invalid_notes}/{total_notes}")
        else:
            result.add_warning(f"Found {invalid_notes} invalid notes (will be filtered)")
    
    # Check for reasonable PPQ
    if midi.ticks_per_beat <= 0:
        result.add_error(f"Invalid PPQ value: {midi.ticks_per_beat}")
    elif midi.ticks_per_beat > 9600:  # Unusually high
        result.add_warning(f"Unusually high PPQ value: {midi.ticks_per_beat}")
    
    # Check tempo validity
    for tempo in midi.tempo_changes:
        if tempo.tempo <= 0 or tempo.tempo > 500:
            result.add_warning(f"Unusual tempo: {tempo.tempo} BPM at tick {tempo.time}")
    
    return result

# Similar to a step in the preprocessing pipeline
def clean_midi_data(midi: MidiFile) -> MidiFile:
    """
    Clean and normalize MIDI data by removing invalid events.
    
    Args:
        midi: MidiFile object to clean
        
    Returns:
        Cleaned MidiFile object
    """
    cleaned_count = 0
    
    for inst in midi.instruments:
        # Filter out invalid notes
        valid_notes = []
        for note in inst.notes:
            if (0 <= note.pitch <= 127 and
                0 <= note.velocity <= 127 and
                note.end > note.start):
                valid_notes.append(note)
            else:
                cleaned_count += 1
        
        inst.notes = valid_notes
        
        # Sort notes by start time for consistency
        inst.notes.sort(key=lambda x: (x.start, x.pitch))
        
        # Clean control changes
        valid_controls = []
        for cc in inst.control_changes:
            if 0 <= cc.number <= 127 and 0 <= cc.value <= 127:
                valid_controls.append(cc)
            else:
                cleaned_count += 1
        inst.control_changes = valid_controls
    
    # Remove empty instruments based on strategy
    strategy = ERROR_HANDLING_STRATEGIES.get("empty_tracks", "remove")
    if strategy == "remove":
        midi.instruments = [inst for inst in midi.instruments if inst.notes]
    
    if cleaned_count > 0:
        logger.info(f"Cleaned {cleaned_count} invalid MIDI events")
    
    return midi


# ============================================================================
# Main Loading Function
# ============================================================================

def load_and_validate_midi(
    file_path: Union[str, Path],
    config: Optional[MidiParserConfig] = None
) -> Tuple[Optional[MidiFile], MidiMetadata, ValidationResult]:
    """
    Load and validate a MIDI file with comprehensive error handling.
    
    This is the main entry point for MIDI file loading as specified in
    the processing pipeline (Section 4, Step 1).
    
    Args:
        file_path: Path to the MIDI file
        config: Optional configuration (uses default if not provided)
        
    Returns:
        Tuple of (MidiFile object or None, metadata, validation result)
        
    Example:
        >>> midi, metadata, validation = load_and_validate_midi("song.mid")
        >>> if validation.is_valid:
        >>>     print(f"Loaded MIDI with {metadata.note_count} notes")
    """
    if config is None:
        config = DEFAULT_CONFIG
    
    file_path = Path(file_path)
    logger.info(f"Loading MIDI file: {file_path.name}")
    
    # Initialize return values
    midi = None
    metadata = MidiMetadata(ppq=480)  # Default PPQ
    validation = ValidationResult(is_valid=False)
    
    try:
        # Check file size before loading
        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        if file_size_mb > config.processing.max_file_size_mb:
            strategy = ERROR_HANDLING_STRATEGIES.get("memory_overflow", "chunk")
            if strategy == "chunk":
                validation.add_warning(f"Large file ({file_size_mb:.1f}MB) - will process in chunks")
            else:
                validation.add_error(f"File size ({file_size_mb:.1f}MB) exceeds limit "
                                   f"({config.processing.max_file_size_mb}MB)")
                return None, metadata, validation
        
        # Load the MIDI file
        midi = load_midi_file(file_path)
        if midi is None:
            validation.add_error("Failed to load MIDI file")
            return None, metadata, validation
        
        # Extract metadata
        metadata = extract_metadata(midi, file_path)
        
        # Validate structure
        validation = validate_midi_structure(midi, config.processing)
        
        if validation.is_valid:
            # Clean the MIDI data if validation passed
            midi = clean_midi_data(midi)
            
            # Re-extract metadata after cleaning
            metadata = extract_metadata(midi, file_path)
            
            logger.info(f"Successfully loaded and validated {file_path.name}: "
                       f"{metadata.note_count} notes, {metadata.duration_seconds:.1f}s")
        else:
            logger.warning(f"MIDI file {file_path.name} failed validation: {validation.errors}")
            
            # Check if we should still try to use it
            if ERROR_HANDLING_STRATEGIES.get("corrupted_midi", "skip") != "skip":
                # Attempt recovery by cleaning
                midi = clean_midi_data(midi)
                metadata = extract_metadata(midi, file_path)
                validation.add_warning("File processed despite validation errors")
                validation.is_valid = True  # Override for partial success
    
    except FileNotFoundError as e:
        validation.add_error(f"File not found: {e}")
        logger.error(f"File not found: {file_path}")
    
    except Exception as e:
        validation.add_error(f"Unexpected error: {e}")
        logger.error(f"Unexpected error loading {file_path.name}: {e}", exc_info=True)
    
    return midi, metadata, validation


# ============================================================================
# Chunking Support for Large Files
# ============================================================================

def find_musical_boundaries(
    midi: MidiFile,
    target_time: float,
    search_window: float = 4.0
) -> float:
    """
    Find the best musical boundary near a target time.
    
    Args:
        midi: MidiFile object
        target_time: Target time in seconds
        search_window: Time window to search for boundaries (seconds)
        
    Returns:
        Best boundary time in seconds
    """
    # Get tick to time mapping
    tick_to_time = midi.get_tick_to_time_mapping()
    target_tick = 0
    
    # Find target tick
    for tick, time in enumerate(tick_to_time):
        if time >= target_time:
            target_tick = tick
            break
    
    # Define search range
    search_start_time = max(0, target_time - search_window / 2)
    search_end_time = target_time + search_window / 2
    
    search_start_tick = 0
    search_end_tick = len(tick_to_time) - 1
    
    for tick, time in enumerate(tick_to_time):
        if time >= search_start_time and search_start_tick == 0:
            search_start_tick = tick
        if time >= search_end_time:
            search_end_tick = tick
            break
    
    # Find musical boundaries in search range
    boundaries = []
    
    # 1. Measure boundaries (highest priority)
    for ts_change in midi.time_signature_changes:
        if search_start_tick <= ts_change.time <= search_end_tick:
            time = tick_to_time[ts_change.time] if ts_change.time < len(tick_to_time) else target_time
            boundaries.append((time, 3, 'measure'))
    
    # Calculate measure boundaries based on time signatures
    current_ts = {'numerator': 4, 'denominator': 4, 'tick': 0}
    for ts_change in sorted(midi.time_signature_changes, key=lambda x: x.time):
        if ts_change.time > search_end_tick:
            break
        if ts_change.time <= search_start_tick:
            current_ts = {'numerator': ts_change.numerator, 'denominator': ts_change.denominator, 'tick': ts_change.time}
    
    # Find beats and measures in the search range
    current_tempo = 120.0  # Default tempo
    for tempo_change in sorted(midi.tempo_changes, key=lambda x: x.time):
        if tempo_change.time <= search_start_tick:
            current_tempo = tempo_change.tempo
        elif tempo_change.time > search_end_tick:
            break
    
    # Calculate ticks per measure
    ticks_per_quarter = midi.ticks_per_beat
    ticks_per_measure = ticks_per_quarter * 4 * current_ts['numerator'] // current_ts['denominator']
    
    # Add measure boundaries
    measure_start_tick = current_ts['tick']
    while measure_start_tick < search_end_tick:
        measure_start_tick += ticks_per_measure
        if search_start_tick <= measure_start_tick <= search_end_tick:
            time = tick_to_time[measure_start_tick] if measure_start_tick < len(tick_to_time) else target_time
            boundaries.append((time, 3, 'measure_calculated'))
    
    # 2. Natural pauses (medium priority)
    # Look for gaps in note activity
    all_notes = []
    for inst in midi.instruments:
        for note in inst.notes:
            if search_start_tick <= note.start <= search_end_tick or search_start_tick <= note.end <= search_end_tick:
                all_notes.append((note.start, note.end))
    
    all_notes.sort()
    
    # Find gaps between notes
    for i in range(len(all_notes) - 1):
        gap_start = all_notes[i][1]
        gap_end = all_notes[i + 1][0]
        gap_duration = gap_end - gap_start
        
        # Consider significant gaps (> quarter note) as natural boundaries
        if gap_duration > ticks_per_quarter and search_start_tick <= gap_start <= search_end_tick:
            time = tick_to_time[gap_start] if gap_start < len(tick_to_time) else target_time
            # Priority based on gap length
            priority = min(2.5, gap_duration / ticks_per_quarter * 0.5)
            boundaries.append((time, priority, 'natural_pause'))
    
    # 3. Beat boundaries (lower priority)
    ticks_per_beat = ticks_per_quarter
    beat_tick = (search_start_tick // ticks_per_beat) * ticks_per_beat
    while beat_tick < search_end_tick:
        beat_tick += ticks_per_beat
        if search_start_tick <= beat_tick <= search_end_tick:
            time = tick_to_time[beat_tick] if beat_tick < len(tick_to_time) else target_time
            boundaries.append((time, 1, 'beat'))
    
    # If no boundaries found, return target time
    if not boundaries:
        return target_time
    
    # Score boundaries by priority and distance to target
    def score_boundary(boundary_time, priority, boundary_type):
        distance = abs(boundary_time - target_time)
        # Normalize distance to search window
        distance_score = 1.0 - (distance / search_window)
        return priority * distance_score
    
    best_boundary = max(boundaries, key=lambda x: score_boundary(x[0], x[1], x[2]))
    
    logger.debug(f"Selected {best_boundary[2]} boundary at {best_boundary[0]:.2f}s "
                f"(target: {target_time:.2f}s)")
    
    return best_boundary[0]


def copy_all_midi_events(
    source: MidiFile,
    target: MidiFile,
    start_tick: int,
    end_tick: int,
    time_offset: int = 0
) -> None:
    """
    Copy all MIDI events from source to target within the specified tick range.
    
    Args:
        source: Source MidiFile
        target: Target MidiFile
        start_tick: Start tick (inclusive)
        end_tick: End tick (inclusive)
        time_offset: Tick offset to apply to copied events
    """
    # Copy tempo changes
    for tempo in source.tempo_changes:
        if start_tick <= tempo.time <= end_tick:
            new_tempo = miditoolkit.TempoChange(
                tempo=tempo.tempo,
                time=tempo.time - time_offset
            )
            target.tempo_changes.append(new_tempo)
    
    # Copy time signature changes
    for ts in source.time_signature_changes:
        if start_tick <= ts.time <= end_tick:
            new_ts = miditoolkit.TimeSignature(
                numerator=ts.numerator,
                denominator=ts.denominator,
                time=ts.time - time_offset
            )
            target.time_signature_changes.append(new_ts)
    
    # Copy key signature changes
    for ks in source.key_signature_changes:
        if start_tick <= ks.time <= end_tick:
            new_ks = miditoolkit.KeySignature(
                key_number=ks.key_number,
                time=ks.time - time_offset
            )
            target.key_signature_changes.append(new_ks)
    
    # Copy markers
    for marker in source.markers:
        if start_tick <= marker.time <= end_tick:
            new_marker = miditoolkit.Marker(
                text=marker.text,
                time=marker.time - time_offset
            )
            target.markers.append(new_marker)
    
    # Copy lyrics
    for lyric in source.lyrics:
        if start_tick <= lyric.time <= end_tick:
            new_lyric = miditoolkit.Lyric(
                text=lyric.text,
                time=lyric.time - time_offset
            )
            target.lyrics.append(new_lyric)


def copy_instrument_with_overlap_handling(
    source_inst: miditoolkit.Instrument,
    target_inst: miditoolkit.Instrument,
    start_tick: int,
    end_tick: int,
    time_offset: int = 0,
    preserve_overlapping: bool = True
) -> Dict[str, int]:
    """
    Copy instrument events with intelligent overlap handling.
    
    Args:
        source_inst: Source instrument
        target_inst: Target instrument
        start_tick: Start tick (inclusive)
        end_tick: End tick (inclusive)
        time_offset: Tick offset to apply
        preserve_overlapping: Whether to preserve overlapping notes
        
    Returns:
        Dictionary with copy statistics
    """
    stats = {
        'notes_copied': 0,
        'notes_truncated': 0,
        'notes_extended': 0,
        'control_changes_copied': 0,
        'program_changes_copied': 0
    }
    
    # Copy notes with overlap handling
    for note in source_inst.notes:
        note_start = note.start
        note_end = note.end
        
        # Determine if note should be included
        include_note = False
        truncate_start = False
        truncate_end = False
        
        if start_tick <= note_start <= end_tick:
            # Note starts within chunk
            include_note = True
            if note_end > end_tick:
                truncate_end = True
        elif preserve_overlapping and note_start < start_tick <= note_end:
            # Note starts before but overlaps into chunk
            include_note = True
            truncate_start = True
            if note_end > end_tick:
                truncate_end = True
        
        if include_note:
            # Adjust note timing
            new_start = max(note_start, start_tick) - time_offset
            new_end = min(note_end, end_tick) - time_offset
            
            # Ensure minimum note duration
            if new_end <= new_start:
                new_end = new_start + 1  # Minimum 1 tick duration
            
            new_note = miditoolkit.Note(
                velocity=note.velocity,
                pitch=note.pitch,
                start=new_start,
                end=new_end
            )
            target_inst.notes.append(new_note)
            stats['notes_copied'] += 1
            
            if truncate_start or truncate_end:
                stats['notes_truncated'] += 1
    
    # Copy control changes
    for cc in source_inst.control_changes:
        if start_tick <= cc.time <= end_tick:
            new_cc = miditoolkit.ControlChange(
                number=cc.number,
                value=cc.value,
                time=cc.time - time_offset
            )
            target_inst.control_changes.append(new_cc)
            stats['control_changes_copied'] += 1
    
    # Copy program changes
    for pc in source_inst.program_changes:
        if start_tick <= pc.time <= end_tick:
            new_pc = miditoolkit.ProgramChange(
                program=pc.program,
                time=pc.time - time_offset
            )
            target_inst.program_changes.append(new_pc)
            stats['program_changes_copied'] += 1
    
    # Copy pitch bends
    for pb in source_inst.pitch_bends:
        if start_tick <= pb.time <= end_tick:
            new_pb = miditoolkit.PitchBend(
                pitch=pb.pitch,
                time=pb.time - time_offset
            )
            target_inst.pitch_bends.append(new_pb)
    
    return stats


def chunk_midi_file(
    midi: MidiFile,
    chunk_size_seconds: float,
    overlap_seconds: float = 2.0,
    respect_musical_boundaries: bool = True,
    preserve_overlapping_notes: bool = True,
    min_chunk_size_seconds: float = 10.0
) -> List[MidiFile]:
    """
    Split a large MIDI file into smaller, musically coherent chunks.
    
    1. Respects musical boundaries (measures, natural pauses, beats)
    2. Handles overlapping notes intelligently
    3. Properly adjusts timing for all MIDI events
    4. Copies all MIDI data (control changes, program changes, etc.)
    
    Args:
        midi: MidiFile object to chunk
        chunk_size_seconds: Target size of each chunk in seconds
        overlap_seconds: Overlap between chunks for continuity
        respect_musical_boundaries: Whether to align chunks to musical boundaries
        preserve_overlapping_notes: Whether to include notes that extend into chunks
        min_chunk_size_seconds: Minimum chunk size to avoid tiny chunks
        
    Returns:
        List of MidiFile chunks with preserved musical integrity
    """
    chunks = []
    total_time = midi.get_tick_to_time_mapping()[-1] if midi.max_tick > 0 else 0.0
    
    if total_time <= chunk_size_seconds:
        logger.info("File shorter than chunk size, returning original")
        return [midi]
    
    # Ensure we have default tempo and time signature
    if not midi.tempo_changes:
        midi.tempo_changes.append(miditoolkit.TempoChange(tempo=120, time=0))
    if not midi.time_signature_changes:
        midi.time_signature_changes.append(miditoolkit.TimeSignature(4, 4, 0))
    
    # Sort all events by time
    midi.tempo_changes.sort(key=lambda x: x.time)
    midi.time_signature_changes.sort(key=lambda x: x.time)
    midi.key_signature_changes.sort(key=lambda x: x.time)
    
    tick_to_time = midi.get_tick_to_time_mapping()
    current_start_time = 0.0
    chunk_index = 0
    
    while current_start_time < total_time:
        # Calculate target end time
        target_end_time = min(current_start_time + chunk_size_seconds, total_time)
        
        # Find musical boundaries if enabled
        if respect_musical_boundaries and target_end_time < total_time:
            actual_end_time = find_musical_boundaries(midi, target_end_time)
            # Ensure we don't create chunks that are too small
            if actual_end_time - current_start_time < min_chunk_size_seconds:
                actual_end_time = min(current_start_time + min_chunk_size_seconds, total_time)
        else:
            actual_end_time = target_end_time
        
        # Convert times to ticks
        start_tick = 0
        end_tick = midi.max_tick
        
        for tick, time in enumerate(tick_to_time):
            if time >= current_start_time and start_tick == 0:
                start_tick = tick
            if time >= actual_end_time:
                end_tick = tick
                break
        
        logger.info(f"Creating chunk {chunk_index + 1}: "
                   f"{current_start_time:.2f}s - {actual_end_time:.2f}s "
                   f"(ticks {start_tick} - {end_tick})")
        
        # Create new chunk
        chunk = MidiFile(ticks_per_beat=midi.ticks_per_beat)
        
        # Copy all MIDI events in range
        copy_all_midi_events(chunk, midi, start_tick, end_tick, start_tick)
        
        # Add initial tempo and time signature if chunk doesn't start at beginning
        if start_tick > 0:
            # Find the most recent tempo before chunk start
            current_tempo = 120  # Default
            for tempo in reversed(midi.tempo_changes):
                if tempo.time <= start_tick:
                    current_tempo = tempo.tempo
                    break
            
            # Add tempo at chunk start if not already present
            has_initial_tempo = any(t.time == 0 for t in chunk.tempo_changes)
            if not has_initial_tempo:
                chunk.tempo_changes.insert(0, miditoolkit.TempoChange(tempo=current_tempo, time=0))
            
            # Find the most recent time signature before chunk start
            current_ts = (4, 4)  # Default
            for ts in reversed(midi.time_signature_changes):
                if ts.time <= start_tick:
                    current_ts = (ts.numerator, ts.denominator)
                    break
            
            # Add time signature at chunk start if not already present
            has_initial_ts = any(ts.time == 0 for ts in chunk.time_signature_changes)
            if not has_initial_ts:
                chunk.time_signature_changes.insert(0, 
                    miditoolkit.TimeSignature(current_ts[0], current_ts[1], 0))
        
        # Copy instruments with overlap handling
        total_stats = {'notes_copied': 0, 'notes_truncated': 0, 'control_changes_copied': 0}
        
        for source_inst in midi.instruments:
            # Create new instrument
            chunk_inst = miditoolkit.Instrument(
                program=source_inst.program,
                is_drum=source_inst.is_drum,
                name=f"{source_inst.name}_chunk_{chunk_index + 1}" if source_inst.name else f"Track_{len(chunk.instruments)}"
            )
            
            # Copy events with overlap handling
            stats = copy_instrument_with_overlap_handling(
                source_inst, chunk_inst, start_tick, end_tick, start_tick,
                preserve_overlapping_notes
            )
            
            # Update total stats
            for key in total_stats:
                if key in stats:
                    total_stats[key] += stats[key]
            
            # Only add instrument if it has content
            if chunk_inst.notes or chunk_inst.control_changes or chunk_inst.program_changes:
                chunk.instruments.append(chunk_inst)
        
        logger.info(f"Chunk {chunk_index + 1} stats: {total_stats['notes_copied']} notes copied, "
                   f"{total_stats['notes_truncated']} truncated")
        
        # Only add chunk if it has musical content
        if chunk.instruments:
            chunks.append(chunk)
            chunk_index += 1
        else:
            logger.warning(f"Skipping empty chunk at {current_start_time:.2f}s")
        
        # Calculate next chunk start with overlap
        if actual_end_time >= total_time:
            break
            
        next_start_time = actual_end_time - overlap_seconds
        # Ensure we make progress
        if next_start_time <= current_start_time:
            next_start_time = current_start_time + min_chunk_size_seconds
        
        current_start_time = next_start_time
    
    logger.info(f"Split MIDI into {len(chunks)} musically coherent chunks")
    return chunks


# Export main functions
__all__ = [
    'load_and_validate_midi',
    'load_midi_file',
    'extract_metadata',
    'validate_midi_structure',
    'clean_midi_data',
    'chunk_midi_file',
    # 3 of the functions used inside 'chunk_midi_file' are not exported as they're only used internally
    'MidiMetadata',
    'ValidationResult',
]
