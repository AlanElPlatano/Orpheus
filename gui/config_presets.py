"""
Configuration presets for Orpheus MIDI Parser.

This module defines configuration presets for different processing modes:
- Simple Mode: Optimized for 2-track melody + chord structure
- Advanced Mode: General-purpose MIDI processing
"""

from midi_parser.config.defaults import (
    MidiParserConfig,
    TokenizerConfig,
    TrackClassificationConfig,
    OutputConfig
)


def get_simple_mode_config(compress: bool = True) -> MidiParserConfig:
    """
    Configuration for simple 2-track mode (melody + chord).
    
    Optimized for specific use case with strict validation:
    - Monophonic melody track
    - Sustained chord track
    - Strict structure enforcement
    
    Args:
        compress: Whether to compress output JSON files
    
    Returns:
        MidiParserConfig configured for simple mode
    """
    return MidiParserConfig(
        tokenization="REMI",
        tokenizer=TokenizerConfig(
            pitch_range=(21, 108),  # Full piano range
            beat_resolution=8,  # Good for rhythm precision
            num_velocities=16,  # Sufficient dynamics
            additional_tokens={
                "Chord": True,  # Important for chord tracks
                "Rest": True,
                "Tempo": True,
                "TimeSignature": True
            },
            max_seq_length=2048
        ),
        track_classification=TrackClassificationConfig(
            min_notes_per_track=10,
            melody_max_polyphony=1.2,  # Strict monophony
            chord_threshold=2.5,  # At least 2-3 notes for chords
            bass_pitch_threshold=50,
            max_empty_ratio=0.7
        ),
        output=OutputConfig(
            compress_json=compress,
            pretty_print=False,
            include_vocabulary=True  # Required for data augmentation (transposition)
        )
    )


def get_advanced_mode_config(compress: bool = True) -> MidiParserConfig:
    """
    Configuration for advanced/general MIDI processing.
    
    More flexible settings for diverse MIDI content:
    - Multiple track types supported
    - Relaxed validation rules
    - Higher sequence length capacity
    
    Args:
        compress: Whether to compress output JSON files
        
    Returns:
        MidiParserConfig configured for advanced mode
    """
    return MidiParserConfig(
        tokenization="REMI",
        tokenizer=TokenizerConfig(
            pitch_range=(21, 108),
            beat_resolution=8,
            num_velocities=32,  # More dynamic range
            additional_tokens={
                "Chord": True,
                "Rest": True,
                "Tempo": True,
                "TimeSignature": True,
                "Pedal": True,
                "PitchBend": False
            },
            max_seq_length=4096  # Longer sequences
        ),
        track_classification=TrackClassificationConfig(
            min_notes_per_track=5,
            melody_max_polyphony=2.0,  # More flexible
            chord_threshold=2.0,
            bass_pitch_threshold=55,
            max_empty_ratio=0.8
        ),
        output=OutputConfig(
            compress_json=compress,
            pretty_print=True,  # Readable output
            include_vocabulary=True  # Include vocab for analysis
        )
    )