"""
MIDI Player tab for Orpheus Gradio GUI.

This tab provides a browser-based MIDI player with piano roll visualization
for previewing generated MIDI files.
"""

import sys
import gradio as gr
import base64
from pathlib import Path
from typing import Tuple, List, Optional
import logging

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

logger = logging.getLogger(__name__)


# ============================================================================
# Helper Functions
# ============================================================================

def get_midi_files(directory: str) -> List[str]:
    """
    Get list of MIDI files from a directory.

    Args:
        directory: Path to directory to search

    Returns:
        List of MIDI file paths (relative to directory)
    """
    try:
        dir_path = Path(directory)
        if not dir_path.exists():
            logger.warning(f"Directory does not exist: {directory}")
            return []

        # Find all .mid and .midi files
        midi_files = []
        for pattern in ['*.mid', '*.midi']:
            midi_files.extend(dir_path.glob(pattern))

        # Sort by modification time (newest first)
        midi_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)

        # Return just the filenames
        return [f.name for f in midi_files]

    except Exception as e:
        logger.error(f"Error listing MIDI files: {e}")
        return []


def load_midi_as_base64(directory: str, filename: str) -> Optional[str]:
    """
    Load a MIDI file and encode it as base64.

    Args:
        directory: Directory containing the MIDI file
        filename: Name of the MIDI file

    Returns:
        Base64-encoded MIDI data, or None if error
    """
    try:
        file_path = Path(directory) / filename

        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            return None

        with open(file_path, 'rb') as f:
            midi_data = f.read()

        # Encode to base64
        base64_data = base64.b64encode(midi_data).decode('utf-8')

        logger.info(f"Loaded MIDI file: {filename} ({len(midi_data)} bytes)")

        return base64_data

    except Exception as e:
        logger.error(f"Error loading MIDI file: {e}")
        return None


def get_midi_metadata(directory: str, filename: str) -> str:
    """
    Extract metadata from a MIDI file.

    Args:
        directory: Directory containing the MIDI file
        filename: Name of the MIDI file

    Returns:
        Formatted metadata string
    """
    try:
        from miditoolkit import MidiFile

        file_path = Path(directory) / filename

        if not file_path.exists():
            return "File not found"

        midi = MidiFile(str(file_path))

        # Extract basic info
        num_tracks = len(midi.instruments)
        total_notes = sum(len(inst.notes) for inst in midi.instruments)

        # Get tempo (first tempo change or default)
        tempo = 120
        if midi.tempo_changes:
            tempo = int(midi.tempo_changes[0].tempo)

        # Get duration
        if midi.max_tick > 0:
            duration_sec = midi.max_tick / midi.ticks_per_beat / tempo * 60
            duration_str = f"{int(duration_sec // 60)}:{int(duration_sec % 60):02d}"
        else:
            duration_str = "0:00"

        # Get time signature
        time_sig = "4/4"
        if midi.time_signature_changes:
            ts = midi.time_signature_changes[0]
            time_sig = f"{ts.numerator}/{ts.denominator}"

        # Format metadata
        metadata = f"""**File:** {filename}
**Duration:** {duration_str}
**Tempo:** {tempo} BPM
**Time Signature:** {time_sig}
**Tracks:** {num_tracks}
**Total Notes:** {total_notes}"""

        return metadata

    except Exception as e:
        logger.error(f"Error extracting metadata: {e}")
        return f"Error reading file: {str(e)}"


def create_midi_player_html(base64_data: str, filename: str) -> str:
    """
    Create HTML for the MIDI player with visualization.

    Args:
        base64_data: Base64-encoded MIDI data
        filename: Name of the file (for display)

    Returns:
        HTML string with embedded MIDI player
    """
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <style>
            body {{
                margin: 0;
                padding: 20px;
                font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
            }}

            .player-container {{
                max-width: 1000px;
                margin: 0 auto;
                background: white;
                border-radius: 16px;
                padding: 30px;
                box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            }}

            .header {{
                text-align: center;
                margin-bottom: 30px;
            }}

            .header h2 {{
                margin: 0 0 10px 0;
                color: #333;
                font-size: 24px;
            }}

            .filename {{
                color: #666;
                font-size: 14px;
                font-family: monospace;
            }}

            midi-player {{
                display: block;
                width: 100%;
                margin: 20px 0;
                border-radius: 8px;
                overflow: hidden;
                background: #f5f5f5;
            }}

            midi-visualizer {{
                display: block;
                width: 100%;
                height: 400px;
                border-radius: 8px;
                background: #1a1a1a;
                margin-top: 20px;
            }}

            .info {{
                margin-top: 20px;
                padding: 15px;
                background: #f8f9fa;
                border-radius: 8px;
                font-size: 13px;
                color: #666;
                text-align: center;
            }}
        </style>

        <!-- Load html-midi-player from CDN -->
        <script src="https://cdn.jsdelivr.net/combine/npm/tone@14.7.58,npm/@magenta/music@1.23.1/es6/core.js,npm/focus-visible@5,npm/html-midi-player@1.5.0"></script>
    </head>
    <body>
        <div class="player-container">
            <div class="header">
                <h2>🎹 MIDI Player</h2>
                <div class="filename">{filename}</div>
            </div>

            <!-- MIDI Player Controls -->
            <midi-player
                src="data:audio/midi;base64,{base64_data}"
                sound-font
                visualizer="#myVisualizer">
            </midi-player>

            <!-- Piano Roll Visualizer -->
            <midi-visualizer
                type="piano-roll"
                id="myVisualizer">
            </midi-visualizer>

            <div class="info">
                🎵 Use the controls above to play, pause, and seek through the MIDI file
            </div>
        </div>
    </body>
    </html>
    """

    return html


# ============================================================================
# Backend Functions
# ============================================================================

def refresh_file_list(directory: str) -> gr.update:
    """Refresh the list of MIDI files."""
    files = get_midi_files(directory)

    if not files:
        return gr.update(choices=["No MIDI files found"], value=None)

    return gr.update(choices=files, value=files[0] if files else None)


def load_and_display_midi(
    directory: str,
    filename: Optional[str]
) -> Tuple[str, str, str]:
    """
    Load a MIDI file and prepare it for display.

    Args:
        directory: Directory containing MIDI files
        filename: Selected filename

    Returns:
        Tuple of (player_html, metadata, status)
    """
    try:
        if not filename or filename == "No MIDI files found":
            empty_html = """
            <div style="padding: 40px; text-align: center; color: #666;">
                <p style="font-size: 18px;">📂 No file selected</p>
                <p>Select a MIDI file from the list to preview it</p>
            </div>
            """
            return empty_html, "No file selected", "Ready"

        # Load MIDI as base64
        base64_data = load_midi_as_base64(directory, filename)

        if base64_data is None:
            error_html = f"""
            <div style="padding: 40px; text-align: center; color: #d32f2f;">
                <p style="font-size: 18px;">❌ Error loading file</p>
                <p>Could not load {filename}</p>
            </div>
            """
            return error_html, "Error loading file", f"Error: Could not load {filename}"

        # Create player HTML
        player_html = create_midi_player_html(base64_data, filename)

        # Extract metadata
        metadata = get_midi_metadata(directory, filename)

        status = f"✅ Loaded: {filename}"

        return player_html, metadata, status

    except Exception as e:
        logger.error(f"Error in load_and_display_midi: {e}", exc_info=True)
        error_html = f"""
        <div style="padding: 40px; text-align: center; color: #d32f2f;">
            <p style="font-size: 18px;">❌ Error</p>
            <p>{str(e)}</p>
        </div>
        """
        return error_html, f"Error: {str(e)}", f"Error: {str(e)}"


# ============================================================================
# Gradio Interface
# ============================================================================

def create_midi_player_tab() -> gr.Tab:
    """Create the MIDI player tab with complete UI."""
    with gr.Tab("MIDI Player") as tab:

        gr.Markdown("""
        ## 🎵 MIDI Player

        Preview your generated MIDI files with an interactive piano roll visualizer.
        """)

        with gr.Row():
            # ===== LEFT COLUMN: File Browser =====
            with gr.Column(scale=1):
                gr.Markdown("### File Browser")

                # Directory input
                directory_input = gr.Textbox(
                    label="MIDI Directory",
                    value="./generated",
                    placeholder="Path to MIDI files directory"
                )

                with gr.Row():
                    refresh_btn = gr.Button("🔄 Refresh", size="sm")

                # File list
                file_list = gr.Radio(
                    label="Available MIDI Files",
                    choices=[],
                    value=None,
                    interactive=True
                )

                gr.Markdown("---")

                # Metadata display
                gr.Markdown("### File Information")

                metadata_display = gr.Markdown("No file selected")

                # Status
                status_display = gr.Textbox(
                    label="Status",
                    value="Ready",
                    interactive=False,
                    lines=1
                )

            # ===== RIGHT COLUMN: MIDI Player =====
            with gr.Column(scale=2):
                gr.Markdown("### Player")

                # MIDI Player (embedded HTML)
                player_html = gr.HTML(
                    value="""
                    <div style="padding: 60px; text-align: center; color: #999; background: #f5f5f5; border-radius: 8px;">
                        <p style="font-size: 24px; margin-bottom: 10px;">🎹</p>
                        <p style="font-size: 16px;">Select a MIDI file to start playing</p>
                    </div>
                    """
                )

                gr.Markdown("""
                ---

                **Tips:**
                - Click a file from the list on the left to load it
                - Use the player controls to play, pause, and seek
                - The piano roll shows all notes in the MIDI file
                - Different tracks are shown in different colors
                """)

        # ===== Event Handlers =====

        # Initial load: populate file list when tab is opened
        tab.select(
            fn=lambda dir: refresh_file_list(dir),
            inputs=[directory_input],
            outputs=[file_list]
        )

        # Refresh button: reload file list
        refresh_btn.click(
            fn=refresh_file_list,
            inputs=[directory_input],
            outputs=[file_list]
        )

        # File selection: load and display MIDI
        file_list.change(
            fn=load_and_display_midi,
            inputs=[directory_input, file_list],
            outputs=[player_html, metadata_display, status_display]
        )

        # Directory change: refresh file list
        directory_input.change(
            fn=refresh_file_list,
            inputs=[directory_input],
            outputs=[file_list]
        )

    return tab


__all__ = ['create_midi_player_tab']
