"""
GUI adapter for MIDI Parser - Orpheus Project

This module provides a GUI-friendly interface to the core MIDI parsing
functionality, adding progress tracking, cancellation, and thread safety.

Location: midi_parser/interface/gui_adapter.py
"""

import logging
from pathlib import Path
from typing import Optional, Callable, Dict, Any, List
from threading import Thread, Event
from dataclasses import dataclass
from enum import Enum
import time

from midi_parser.core.midi_loader import load_and_validate_midi
from midi_parser.core.track_analyzer import analyze_tracks
from midi_parser.core.tokenizer_manager import tokenize_midi
from midi_parser.core.json_serializer import create_output_json, ProcessingMetadata
from midi_parser.config.defaults import MidiParserConfig

logger = logging.getLogger(__name__)


# ============================================================================
# Enums and Data Classes
# ============================================================================

class ProcessingStage(Enum):
    """Stages of MIDI processing for progress tracking."""
    LOADING = "loading"
    VALIDATING = "validating"
    ANALYZING = "analyzing"
    TOKENIZING = "tokenizing"
    SERIALIZING = "serializing"
    COMPLETE = "complete"
    ERROR = "error"
    CANCELLED = "cancelled"


@dataclass
class ProcessingProgress:
    """Progress information for GUI updates."""
    stage: ProcessingStage
    current: int  # 0-100
    total: int  # Always 100
    message: str
    file_name: str


@dataclass
class ProcessingResult:
    """Complete result from processing operation."""
    success: bool
    file_path: Optional[Path] = None
    output_path: Optional[Path] = None
    error_message: Optional[str] = None
    warnings: Optional[List[str]] = None
    processing_time: float = 0.0
    
    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []


class OperationCancelledError(Exception):
    """Raised when an operation is cancelled by user."""
    pass


# ============================================================================
# Main GUI Adapter Class
# ============================================================================

class MidiParserGUI:
    """
    GUI-friendly wrapper for MIDI parser operations.
    
    Provides:
    - Progress callbacks for UI updates
    - Cancellation support
    - Thread-safe operation
    - User-friendly error messages
    - Memory estimation
    - Pre-flight validation
    
    Example:
        >>> parser = MidiParserGUI(config=my_config)
        >>> result = parser.process_file(
        ...     input_path,
        ...     output_dir,
        ...     progress_callback=update_ui
        ... )
        >>> if result.success:
        ...     print(f"Saved to {result.output_path}")
    """
    
    def __init__(
        self,
        config: Optional[MidiParserConfig] = None,
        log_callback: Optional[Callable[[str, str], None]] = None
    ):
        """
        Initialize GUI adapter.
        
        Args:
            config: Parser configuration (uses default if None)
            log_callback: Callback for log messages (level, message)
        """
        self.config = config
        self.log_callback = log_callback
        self._cancel_event = Event()
    
    def _log(self, level: str, message: str) -> None:
        """Send log message to callback and standard logger."""
        if self.log_callback:
            try:
                self.log_callback(level, message)
            except Exception as e:
                logger.warning(f"Log callback failed: {e}")
        
        log_func = getattr(logger, level.lower(), logger.info)
        log_func(message)
    
    def _check_cancelled(self) -> None:
        """Check if operation was cancelled and raise exception if so."""
        if self._cancel_event.is_set():
            raise OperationCancelledError("Operation cancelled by user")
    
    def estimate_memory_usage(self, file_path: Path) -> Dict[str, Any]:
        """
        Estimate memory requirements for processing a file.
        
        Args:
            file_path: Path to MIDI file
            
        Returns:
            Dictionary with memory estimates in MB
            
        Example:
            >>> mem = parser.estimate_memory_usage(Path("song.mid"))
            >>> if not mem["safe_to_process"]:
            ...     print(f"Warning: May use {mem['peak_mb']}MB")
        """
        try:
            file_size_mb = file_path.stat().st_size / (1024 * 1024)
        except Exception:
            file_size_mb = 0
        
        # Conservative estimates based on typical processing overhead
        estimates = {
            "file_size_mb": round(file_size_mb, 2),
            "loading_mb": round(file_size_mb * 3, 2),
            "tokenization_mb": round(file_size_mb * 5, 2),
            "peak_mb": round(file_size_mb * 8, 2),
            "safe_to_process": file_size_mb < 50  # Conservative limit
        }
        
        return estimates
    
    def validate_before_processing(
        self,
        file_path: Path,
        output_dir: Path
    ) -> tuple[bool, List[str]]:
        """
        Validate inputs before starting expensive operations.
        
        Args:
            file_path: Input MIDI file
            output_dir: Output directory
            
        Returns:
            Tuple of (is_valid, list of issues)
            
        Example:
            >>> is_valid, issues = parser.validate_before_processing(
            ...     Path("song.mid"),
            ...     Path("output")
            ... )
            >>> if not is_valid:
            ...     print("\\n".join(issues))
        """
        issues = []
        
        # Check input file
        if not file_path.exists():
            issues.append(f"Input file does not exist: {file_path}")
        elif not file_path.is_file():
            issues.append(f"Path is not a file: {file_path}")
        elif not file_path.suffix.lower() in ['.mid', '.midi']:
            issues.append(f"File is not a MIDI file: {file_path.suffix}")
        
        # Check output directory
        if not output_dir.exists():
            try:
                output_dir.mkdir(parents=True, exist_ok=True)
                self._log("info", f"Created output directory: {output_dir}")
            except Exception as e:
                issues.append(f"Cannot create output directory: {e}")
        elif not output_dir.is_dir():
            issues.append(f"Output path is not a directory: {output_dir}")
        
        # Check write permissions
        try:
            test_file = output_dir / ".write_test"
            test_file.touch()
            test_file.unlink()
        except Exception as e:
            issues.append(f"No write permission in output directory: {e}")
        
        # Check memory if file exists
        if file_path.exists() and file_path.is_file():
            mem_info = self.estimate_memory_usage(file_path)
            if not mem_info["safe_to_process"]:
                issues.append(
                    f"Large file ({mem_info['file_size_mb']:.1f}MB) may require "
                    f"{mem_info['peak_mb']:.1f}MB of memory"
                )
        
        return len(issues) == 0, issues
    
    def process_file(
        self,
        input_path: Path,
        output_dir: Path,
        progress_callback: Optional[Callable[[ProcessingProgress], None]] = None
    ) -> ProcessingResult:
        """
        Process a single MIDI file with progress tracking.
        
        Args:
            input_path: Path to MIDI file
            output_dir: Output directory for JSON
            progress_callback: Optional callback for progress updates
            
        Returns:
            ProcessingResult with status and output information
            
        Example:
            >>> def show_progress(prog):
            ...     print(f"{prog.stage.value}: {prog.message}")
            >>> 
            >>> result = parser.process_file(
            ...     Path("song.mid"),
            ...     Path("output"),
            ...     progress_callback=show_progress
            ... )
            >>> print(f"Success: {result.success}")
        """
        start_time = time.time()
        self._cancel_event.clear()
        
        result = ProcessingResult(
            success=False,
            file_path=input_path
        )
        
        def update_progress(
            stage: ProcessingStage,
            current: int,
            total: int,
            message: str
        ):
            """Helper to send progress updates."""
            if progress_callback:
                try:
                    progress_callback(ProcessingProgress(
                        stage=stage,
                        current=current,
                        total=total,
                        message=message,
                        file_name=input_path.name
                    ))
                except Exception as e:
                    self._log("warning", f"Progress callback failed: {e}")
        
        try:
            # Pre-flight validation
            is_valid, issues = self.validate_before_processing(input_path, output_dir)
            if not is_valid:
                result.error_message = "; ".join(issues)
                update_progress(ProcessingStage.ERROR, 0, 100, "Validation failed")
                return result
            
            # Stage 1: Loading (0% -> 20%)
            update_progress(ProcessingStage.LOADING, 0, 100, "Loading MIDI file...")
            self._check_cancelled()
            self._log("info", f"Loading {input_path.name}")
            
            midi, metadata, validation = load_and_validate_midi(input_path, self.config)
            
            if not validation.is_valid:
                result.error_message = "; ".join(validation.errors)
                result.warnings = validation.warnings
                update_progress(ProcessingStage.ERROR, 0, 100, "MIDI validation failed")
                return result
            
            update_progress(ProcessingStage.LOADING, 20, 100, "MIDI loaded successfully")
            result.warnings.extend(validation.warnings)
            
            # Stage 2: Analyzing tracks (20% -> 40%)
            update_progress(ProcessingStage.ANALYZING, 20, 100, "Analyzing tracks...")
            self._check_cancelled()
            self._log("info", f"Analyzing {len(midi.instruments)} tracks")
            
            track_infos = analyze_tracks(midi, self.config)
            
            if not track_infos:
                result.error_message = "No valid tracks found in MIDI file"
                update_progress(ProcessingStage.ERROR, 0, 100, "No valid tracks")
                return result
            
            update_progress(
                ProcessingStage.ANALYZING,
                40,
                100,
                f"Found {len(track_infos)} valid tracks"
            )
            
            # Stage 3: Tokenizing (40% -> 70%)
            update_progress(ProcessingStage.TOKENIZING, 40, 100, "Tokenizing MIDI...")
            self._check_cancelled()
            self._log("info", "Starting tokenization")
            
            # Tokenize the entire MIDI file
            tokenization_result = tokenize_midi(
                midi,
                self.config,
                track_infos=track_infos,
                auto_select=False
            )
            
            if not tokenization_result.success:
                result.error_message = f"Tokenization failed: {tokenization_result.error_message}"
                update_progress(ProcessingStage.ERROR, 0, 100, "Tokenization failed")
                return result
            
            # For compatibility, wrap in list (json_serializer expects list)
            tokenization_results = [tokenization_result]
            
            if tokenization_result.warnings:
                result.warnings.extend(tokenization_result.warnings)
            
            update_progress(
                ProcessingStage.TOKENIZING,
                70,
                100,
                f"Tokenization complete: {len(tokenization_result.tokens)} tokens"
            )
            
            # Stage 4: Serializing (70% -> 95%)
            update_progress(ProcessingStage.SERIALIZING, 70, 100, "Creating JSON output...")
            self._check_cancelled()
            self._log("info", "Serializing to JSON")
            
            processing_metadata = ProcessingMetadata(
                processing_time_seconds=time.time() - start_time,
                validation_passed=True,
                warnings=result.warnings
            )
            
            serialization_result = create_output_json(
                input_path,
                midi,
                metadata,
                track_infos,
                tokenization_results,
                output_dir,
                self.config,
                validation,
                processing_metadata
            )
            
            if not serialization_result.success:
                result.error_message = serialization_result.error_message
                update_progress(ProcessingStage.ERROR, 0, 100, "Serialization failed")
                return result
            
            update_progress(ProcessingStage.SERIALIZING, 95, 100, "JSON saved successfully")
            
            # Complete (95% -> 100%)
            result.success = True
            result.output_path = serialization_result.output_path
            result.processing_time = time.time() - start_time
            
            update_progress(
                ProcessingStage.COMPLETE,
                100,
                100,
                f"Processing complete in {result.processing_time:.1f}s"
            )
            
            self._log("info", f"Successfully processed {input_path.name}")
        
        except OperationCancelledError:
            result.error_message = "Processing cancelled by user"
            update_progress(ProcessingStage.CANCELLED, 0, 100, "Cancelled")
            self._log("warning", f"Processing cancelled: {input_path.name}")
        
        except Exception as e:
            result.error_message = f"Unexpected error: {str(e)}"
            update_progress(ProcessingStage.ERROR, 0, 100, f"Error: {str(e)}")
            self._log("error", f"Error processing {input_path.name}: {e}")
            
            # Log full traceback for debugging
            import traceback
            self._log("debug", traceback.format_exc())
        
        return result
    
    def process_file_async(
        self,
        input_path: Path,
        output_dir: Path,
        progress_callback: Optional[Callable[[ProcessingProgress], None]] = None,
        completion_callback: Optional[Callable[[ProcessingResult], None]] = None
    ) -> Thread:
        """
        Process file in background thread (non-blocking).
        
        Args:
            input_path: Path to MIDI file
            output_dir: Output directory
            progress_callback: Optional callback for progress updates
            completion_callback: Optional callback when processing completes
            
        Returns:
            Thread object (already started)
            
        Example:
            >>> def on_complete(result):
            ...     print(f"Done! Success: {result.success}")
            >>> 
            >>> thread = parser.process_file_async(
            ...     Path("song.mid"),
            ...     Path("output"),
            ...     completion_callback=on_complete
            ... )
            >>> # GUI remains responsive while processing
        """
        def worker():
            result = self.process_file(input_path, output_dir, progress_callback)
            if completion_callback:
                try:
                    completion_callback(result)
                except Exception as e:
                    self._log("error", f"Completion callback failed: {e}")
        
        thread = Thread(target=worker, daemon=True, name=f"Parser-{input_path.name}")
        thread.start()
        return thread
    
    def cancel_operation(self):
        """
        Cancel the current operation.
        
        The operation will stop at the next checkpoint and return
        a ProcessingResult with cancelled status.
        """
        self._cancel_event.set()
        self._log("info", "Cancellation requested")
    
    def process_batch(
        self,
        input_files: List[Path],
        output_dir: Path,
        file_progress_callback: Optional[Callable[[int, int, str], None]] = None,
        item_progress_callback: Optional[Callable[[ProcessingProgress], None]] = None
    ) -> List[ProcessingResult]:
        """
        Process multiple files with batch progress tracking.
        
        Args:
            input_files: List of MIDI files to process
            output_dir: Output directory
            file_progress_callback: Callback for batch progress (current, total, filename)
            item_progress_callback: Callback for individual file progress
            
        Returns:
            List of ProcessingResult objects
            
        Example:
            >>> def batch_progress(current, total, filename):
            ...     print(f"File {current}/{total}: {filename}")
            >>> 
            >>> results = parser.process_batch(
            ...     [Path("song1.mid"), Path("song2.mid")],
            ...     Path("output"),
            ...     file_progress_callback=batch_progress
            ... )
            >>> success_count = sum(1 for r in results if r.success)
        """
        results = []
        
        for i, file_path in enumerate(input_files):
            try:
                self._check_cancelled()
            except OperationCancelledError:
                self._log("info", "Batch processing cancelled")
                # Add cancelled results for remaining files
                for remaining_file in input_files[i:]:
                    results.append(ProcessingResult(
                        success=False,
                        file_path=remaining_file,
                        error_message="Batch processing cancelled"
                    ))
                break
            
            if file_progress_callback:
                try:
                    file_progress_callback(i, len(input_files), file_path.name)
                except Exception as e:
                    self._log("warning", f"File progress callback failed: {e}")
            
            result = self.process_file(file_path, output_dir, item_progress_callback)
            results.append(result)
            
            # Optionally stop batch on critical errors
            if not result.success and "memory" in result.error_message.lower():
                self._log("error", f"Batch processing stopped due to memory issue in {file_path.name}")
                break
        
        if file_progress_callback:
            try:
                file_progress_callback(len(input_files), len(input_files), "Complete")
            except Exception:
                pass
        
        successful = sum(1 for r in results if r.success)
        self._log("info", f"Batch complete: {successful}/{len(results)} files processed successfully")
        
        return results


# ============================================================================
# Utility Functions
# ============================================================================

def validate_simple_mode_structure(track_infos: List) -> tuple[bool, str]:
    """
    Validate MIDI structure for simple mode (2 tracks: melody + chord).
    
    Args:
        track_infos: List of TrackInfo objects from analyzer
        
    Returns:
        Tuple of (is_valid, message)
        
    Example:
        >>> is_valid, msg = validate_simple_mode_structure(track_infos)
        >>> if not is_valid:
        ...     print(f"Validation failed: {msg}")
    """
    if len(track_infos) != 2:
        return False, f"Expected exactly 2 tracks, found {len(track_infos)}"
    
    track_types = [t.type for t in track_infos]
    
    has_melody = "melody" in track_types
    has_chord = "chord" in track_types
    
    if not has_melody:
        return False, "No melody track found. Expected track types: melody, chord"
    
    if not has_chord:
        return False, "No chord track found. Expected track types: melody, chord"
    
    # Check melody monophony
    melody_track = next(t for t in track_infos if t.type == "melody")
    if melody_track.statistics.avg_polyphony > 1.3:
        return False, (
            f"Melody track is not monophonic "
            f"(average polyphony: {melody_track.statistics.avg_polyphony:.2f}). "
            f"Please ensure melody has no overlapping notes."
        )
    
    # Check chord track has reasonable polyphony
    chord_track = next(t for t in track_infos if t.type == "chord")
    if chord_track.statistics.avg_polyphony < 2.0:
        return False, (
            f"Chord track has insufficient polyphony "
            f"(average: {chord_track.statistics.avg_polyphony:.2f}). "
            f"Expected at least 2 simultaneous notes for chords."
        )
    
    return True, "✅ Valid structure: Monophonic melody + Chord track"


# Export all public classes and functions
__all__ = [
    'MidiParserGUI',
    'ProcessingStage',
    'ProcessingProgress',
    'ProcessingResult',
    'OperationCancelledError',
    'validate_simple_mode_structure',
]