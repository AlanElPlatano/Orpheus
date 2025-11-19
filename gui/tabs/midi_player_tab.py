"""
MIDI Player tab for Orpheus Gradio GUI.

This tab provides a browser-based MIDI player with piano roll visualization
for previewing generated MIDI files.
"""

import sys
import gradio as gr
from pathlib import Path
from typing import Tuple, List, Optional, Dict, Any
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
        List of MIDI file paths (relative to directory), sorted alphabetically
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

        # Sort alphabetically
        midi_files.sort(key=lambda p: p.name.lower())

        # Return just the filenames
        return [f.name for f in midi_files]

    except Exception as e:
        logger.error(f"Error listing MIDI files: {e}")
        return []


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


def extract_midi_notes(midi_path: Path) -> Dict[str, Any]:
    """
    Extract note data from MIDI file for visualization.

    Args:
        midi_path: Path to MIDI file

    Returns:
        Dictionary with note data and metadata
    """
    try:
        from miditoolkit import MidiFile

        midi = MidiFile(str(midi_path))

        notes = []
        for track_idx, instrument in enumerate(midi.instruments):
            for note in instrument.notes:
                notes.append({
                    'pitch': note.pitch,
                    'start': note.start / midi.ticks_per_beat,  # Convert to beats
                    'end': note.end / midi.ticks_per_beat,
                    'velocity': note.velocity,
                    'track': track_idx,
                    'instrument': instrument.name or f"Track {track_idx}"
                })

        # Calculate duration in beats
        if notes:
            duration = max(n['end'] for n in notes)
        else:
            duration = 0

        return {
            'notes': notes,
            'duration': duration,
            'tempo': midi.tempo_changes[0].tempo if midi.tempo_changes else 120,
            'ticks_per_beat': midi.ticks_per_beat
        }

    except Exception as e:
        logger.error(f"Error extracting MIDI notes: {e}")
        return {'notes': [], 'duration': 0, 'tempo': 120, 'ticks_per_beat': 480}


def create_midi_player_html(midi_path: Path, filename: str) -> str:
    """
    Create HTML for the MIDI player with visualization.
    Uses inline JavaScript to bypass CSP restrictions.

    Args:
        midi_path: Path to MIDI file
        filename: Name of the file (for display)

    Returns:
        HTML string with embedded MIDI player
    """

    # Extract note data from MIDI
    midi_data = extract_midi_notes(midi_path)

    # Read the inline player JavaScript
    player_js_path = Path(__file__).parent.parent / "static" / "midi-player" / "inline-midi-player.js"

    if player_js_path.exists():
        with open(player_js_path, 'r') as f:
            player_js = f.read()
    else:
        # Fallback if file doesn't exist
        player_js = "console.error('Player JavaScript not found');"

    # Convert Python data to JSON for JavaScript
    import json
    midi_data_json = json.dumps(midi_data)

    html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <style>
        * {{
            box-sizing: border-box !important;
        }}

        body {{
            margin: 0 !important;
            padding: 20px !important;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Helvetica', 'Arial', sans-serif !important;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
            min-height: 100vh !important;
        }}

        .player-container {{
            max-width: 1000px !important;
            margin: 0 auto !important;
            background: #ffffff !important;
            border-radius: 16px !important;
            padding: 30px !important;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3) !important;
        }}

        .header {{
            text-align: center !important;
            margin-bottom: 30px !important;
        }}

        .header h2 {{
            margin: 0 0 10px 0 !important;
            color: #1f2937 !important;
            font-size: 24px !important;
            font-weight: 600 !important;
        }}

        .filename {{
            color: #6b7280 !important;
            font-size: 14px !important;
            font-family: 'Courier New', monospace !important;
            background: #f3f4f6 !important;
            padding: 8px 16px !important;
            border-radius: 6px !important;
            display: inline-block !important;
        }}

        .info {{
            margin-top: 20px !important;
            padding: 15px !important;
            background: #f0fdf4 !important;
            border-radius: 8px !important;
            font-size: 13px !important;
            color: #166534 !important;
            text-align: center !important;
            border: 1px solid #86efac !important;
        }}

        #midiPlayerContainer {{
            margin: 20px 0 !important;
        }}
    </style>
</head>
<body>
    <div class="player-container">
        <div class="header">
            <h2>🎹 MIDI Player</h2>
            <div class="filename">{filename}</div>
        </div>

        <div id="midiPlayerContainer"></div>

        <div class="info">
            🎵 Use the controls above to play, pause, and seek through the MIDI file
        </div>
    </div>

    <script>
    {player_js}

    // Initialize player with MIDI data
    const midiData = {midi_data_json};
    const player = new InlineMIDIPlayer('midiPlayerContainer');
    player.loadMIDIData(midiData);
    </script>
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
            <div style="padding: 60px; text-align: center; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 12px;">
                <div style="background: white; padding: 40px; border-radius: 8px; display: inline-block;">
                    <p style="font-size: 48px; margin: 0 0 20px 0;">📂</p>
                    <p style="font-size: 18px; color: #1f2937; margin: 0 0 10px 0; font-weight: 600;">No file selected</p>
                    <p style="font-size: 14px; color: #6b7280; margin: 0;">Select a MIDI file from the dropdown to preview it</p>
                </div>
            </div>
            """
            return empty_html, "No file selected", "Ready"

        # Get MIDI file path
        file_path = Path(directory) / filename

        if not file_path.exists():
            error_html = f"""
            <div style="padding: 60px; text-align: center; background: #fee; border-radius: 12px;">
                <div style="background: white; padding: 40px; border-radius: 8px; display: inline-block; border: 2px solid #ef4444;">
                    <p style="font-size: 48px; margin: 0 0 20px 0;">❌</p>
                    <p style="font-size: 18px; color: #991b1b; margin: 0 0 10px 0; font-weight: 600;">Error loading file</p>
                    <p style="font-size: 14px; color: #7f1d1d; margin: 0;">Could not load {filename}</p>
                </div>
            </div>
            """
            return error_html, "Error loading file", f"Error: Could not load {filename}"

        # Create player HTML
        player_html = create_midi_player_html(file_path, filename)

        # Extract metadata
        metadata = get_midi_metadata(directory, filename)

        status = f"✅ Loaded: {filename}"

        return player_html, metadata, status

    except Exception as e:
        logger.error(f"Error in load_and_display_midi: {e}", exc_info=True)
        error_html = f"""
        <div style="padding: 60px; text-align: center; background: #fee; border-radius: 12px;">
            <div style="background: white; padding: 40px; border-radius: 8px; display: inline-block; border: 2px solid #ef4444;">
                <p style="font-size: 48px; margin: 0 0 20px 0;">❌</p>
                <p style="font-size: 18px; color: #991b1b; margin: 0 0 10px 0; font-weight: 600;">Error</p>
                <p style="font-size: 14px; color: #7f1d1d; margin: 0; font-family: monospace;">{str(e)}</p>
            </div>
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

                # File list - dropdown for clean file explorer feel
                file_list = gr.Dropdown(
                    label="Select MIDI File",
                    choices=[],
                    value=None,
                    interactive=True,
                    info="Files are sorted alphabetically"
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
                    <div style="padding: 60px; text-align: center; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 12px;">
                        <div style="background: white; padding: 40px; border-radius: 8px; display: inline-block;">
                            <p style="font-size: 48px; margin: 0 0 20px 0;">🎹</p>
                            <p style="font-size: 18px; color: #1f2937; margin: 0 0 10px 0; font-weight: 600;">MIDI Player Ready</p>
                            <p style="font-size: 14px; color: #6b7280; margin: 0;">Select a file from the dropdown to start playing</p>
                        </div>
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
