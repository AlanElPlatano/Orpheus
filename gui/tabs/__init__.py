"""
Tab modules for Orpheus Gradio GUI.

Each module creates a complete tab with UI and event handlers:
- preprocess_tab: MIDI preprocessing functionality
- parser_tab: MIDI tokenization and JSON output
- augmentation_tab: Data augmentation via transposition
- json_to_midi_tab: JSON to MIDI batch conversion
- training_tab: AI model training interface
- generator_tab: Music generation interface
- custom_chords_tab: Generate melody from custom chord MIDI files
"""

from .preprocess_tab import create_preprocess_tab
from .parser_tab import create_parser_tab
from .augmentation_tab import create_augmentation_tab
from .json_to_midi_tab import create_json_to_midi_tab
from .training_tab import create_training_tab
from .generator_tab import create_generator_tab
from .custom_chords_tab import create_custom_chords_tab

__all__ = [
    'create_preprocess_tab',
    'create_parser_tab',
    'create_augmentation_tab',
    'create_json_to_midi_tab',
    'create_training_tab',
    'create_generator_tab',
    'create_custom_chords_tab',
]