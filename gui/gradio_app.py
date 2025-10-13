"""
Main Gradio application for Orpheus Project.

This is the entry point that assembles all tabs into a complete application.

Usage:
    From project root:
    python -m gui.gradio_app
"""

import gradio as gr
from pathlib import Path

from gui.tabs import (
    create_preprocess_tab,
    create_parser_tab,
    create_json_to_midi_tab,
    create_training_tab,
    create_generator_tab
)


def create_interface() -> gr.Blocks:
    """
    Create the complete Gradio interface by assembling all tabs.
    
    Returns:
        Gradio Blocks interface
    """
    with gr.Blocks(
        title="Orpheus - AI Music Generation",
        theme=gr.themes.Soft(primary_hue="blue", secondary_hue="purple")
    ) as app:

        # Header
        gr.Markdown("""
        # Orpheus - Music Composer
        
        Complete toolkit for MIDI processing, tokenization, and AI music generation.
        """)

        # Create all tabs
        with gr.Tabs():
            create_preprocess_tab()
            create_parser_tab()
            create_training_tab()
            create_generator_tab()
            create_json_to_midi_tab()
        
        # Footer
        gr.Markdown("""
        ---
        **Orpheus Project** | AI-powered music generation from MIDI
        """)

    return app


def main():
    """Main entry point for the Gradio application."""

    # Ensure output directories exist
    Path("./processed").mkdir(exist_ok=True)
    Path("./generated").mkdir(exist_ok=True)

    # Create and launch the interface
    app = create_interface()
    
    app.launch(
        server_name="localhost",
        server_port=7860,
        share=False,
        show_error=True,
        favicon_path=None,
    )


if __name__ == "__main__":
    main()