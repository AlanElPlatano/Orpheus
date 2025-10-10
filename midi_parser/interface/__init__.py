"""
Interface adapters for MIDI Parser - Orpheus Project

This module provides user-friendly wrappers around the core processing
functionality, adding features needed for GUI, CLI, and API interfaces:

- Progress tracking and cancellation
- User-friendly error messages
- Memory estimation
- Pre-flight validation
- Thread-safe operations
- Batch processing support

Location: midi_parser/interface/__init__.py

Example:
    >>> from midi_parser.interface import MidiParserGUI, ProcessingProgress
    >>> 
    >>> def show_progress(progress: ProcessingProgress):
    ...     print(f"{progress.stage.value}: {progress.message}")
    >>> 
    >>> parser = MidiParserGUI()
    >>> result = parser.process_file(
    ...     Path("song.mid"),
    ...     Path("output"),
    ...     progress_callback=show_progress
    ... )
    >>> 
    >>> if result.success:
    ...     print(f"Output saved to: {result.output_path}")
"""

from .gui_adapter import (
    MidiParserGUI,
    ProcessingStage,
    ProcessingProgress,
    ProcessingResult,
    OperationCancelledError,
    validate_simple_mode_structure
)

__version__ = "1.0.0"

__all__ = [
    # Main adapter class
    'MidiParserGUI',
    
    # Enums and data classes
    'ProcessingStage',
    'ProcessingProgress',
    'ProcessingResult',
    
    # Exceptions
    'OperationCancelledError',
    
    # Utility functions
    'validate_simple_mode_structure',
]