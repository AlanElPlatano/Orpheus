"""
Tab modules for Orpheus Gradio GUI.

Each module creates a complete tab with UI and event handlers:
- preprocess_tab: MIDI preprocessing functionality
- parser_tab: MIDI tokenization and JSON output
- training_tab: AI model training interface
- generator_tab: Music generation interface
"""

from .preprocess_tab import create_preprocess_tab
from .parser_tab import create_parser_tab
from .training_tab import create_training_tab
from .generator_tab import create_generator_tab

__all__ = [
    'create_preprocess_tab',
    'create_parser_tab',
    'create_training_tab',
    'create_generator_tab',
]