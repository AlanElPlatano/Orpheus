"""
Gradio GUI for Orpheus Project

Usage:
    (from project root)
    python -m gui.gradio_app
"""

import gradio as gr
from pathlib import Path
from typing import List, Optional, Tuple
import json
from datetime import datetime

from midi_parser.interface import (
    MidiParserGUI,
    ProcessingStage,
    ProcessingProgress,
    ProcessingResult,
    OperationCancelledError
)
from midi_parser.config.defaults import (
    MidiParserConfig,
    TokenizerConfig,
    TrackClassificationConfig,
    OutputConfig
)


# ============================================================================
# Configuration Presets
# ============================================================================

def get_simple_mode_config() -> MidiParserConfig:
    """
    Configuration for simple 2-track mode (melody + chord).
    
    Optimized for your specific use case with strict validation.
    """
    return MidiParserConfig(
        tokenization="REMI",  # Best for structured music
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
            compress_json=True,
            pretty_print=False,
            include_vocabulary=False
        )
    )


def get_advanced_mode_config() -> MidiParserConfig:
    """Configuration for advanced/general MIDI processing."""
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
            compress_json=True,
            pretty_print=True,  # Readable output
            include_vocabulary=True  # Include vocab for analysis
        )
    )


# ============================================================================
# Global State Management
# ============================================================================

class AppState:
    """Global application state."""
    def __init__(self):
        self.parser: Optional[MidiParserGUI] = None
        self.current_mode = "simple"
        self.processing = False
        self.logs = []
        self.results = []
    
    def initialize_parser(self, mode: str):
        """Initialize parser with appropriate config."""
        config = get_simple_mode_config() if mode == "simple" else get_advanced_mode_config()
        self.parser = MidiParserGUI(
            config=config,
            log_callback=self.log_handler
        )
        self.current_mode = mode
    
    def log_handler(self, level: str, message: str):
        """Handle log messages from parser."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] [{level.upper()}] {message}"
        self.logs.append(log_entry)


# Global state instance
app_state = AppState()


# ============================================================================
# Processing Functions
# ============================================================================

def validate_simple_mode_structure(track_infos: List) -> Tuple[bool, str]:
    """
    Validate that MIDI has the expected 2-track structure.
    
    Returns:
        Tuple of (is_valid, message)
    """
    if len(track_infos) != 2:
        return False, f"❌ Expected exactly 2 tracks, found {len(track_infos)}"
    
    track_types = [t.type for t in track_infos]
    
    has_melody = "melody" in track_types
    has_chord = "chord" in track_types
    
    if not has_melody:
        return False, "❌ No melody track found"
    
    if not has_chord:
        return False, "❌ No chord track found"
    
    # Check melody is monophonic
    melody_track = next(t for t in track_infos if t.type == "melody")
    if melody_track.statistics.avg_polyphony > 1.3:
        return False, f"⚠️ Melody track is not monophonic (polyphony: {melody_track.statistics.avg_polyphony:.2f})"
    
    return True, "✅ Structure valid: Monophonic melody + Chord track"


def process_single_file(
    file_path: str,
    output_dir: str,
    mode: str,
    progress=gr.Progress()
) -> Tuple[str, str, str, str]:
    """
    Process a single MIDI file.
    
    Returns:
        Tuple of (status, output_path, statistics, logs)
    """
    # Initialize parser if needed
    if app_state.parser is None or app_state.current_mode != mode:
        app_state.initialize_parser(mode)
    
    app_state.logs.clear()
    
    input_path = Path(file_path)
    output_path = Path(output_dir)
    
    # Progress callback for Gradio
    def progress_callback(prog: ProcessingProgress):
        progress(
            prog.current / 100,
            desc=f"{prog.stage.value.title()}: {prog.message}"
        )
    
    try:
        # Pre-flight checks
        progress(0, desc="Validating input...")
        
        mem_info = app_state.parser.estimate_memory_usage(input_path)
        if not mem_info["safe_to_process"]:
            return (
                f"⚠️ Warning: Large file ({mem_info['file_size_mb']:.1f}MB)",
                "",
                f"Estimated memory usage: {mem_info['peak_mb']:.1f}MB",
                "\n".join(app_state.logs)
            )
        
        # Process file
        result = app_state.parser.process_file(
            input_path,
            output_path,
            progress_callback=progress_callback
        )
        
        if result.success:
            # Load the output JSON to display stats
            try:
                with open(result.output_path, 'r') as f:
                    output_data = json.load(f)
                
                # Build statistics
                stats = f"""
## ✅ Processing Complete!

**Output File:** `{result.output_path.name}`
**Processing Time:** {result.processing_time:.2f}s

### Statistics:
- **Tracks:** {len(output_data.get('tracks', []))}
- **Total Tokens:** {output_data.get('sequence_length', 0):,}
- **Vocabulary Size:** {output_data.get('vocabulary_size', 0):,}
- **Duration:** {output_data.get('metadata', {}).get('duration_seconds', 0):.1f}s
- **Note Count:** {output_data.get('metadata', {}).get('note_count', 0):,}

### Track Breakdown:
"""
                for track in output_data.get('tracks', []):
                    stats += f"\n- **{track['name']}** ({track['type']}): {track['token_count']:,} tokens, {track['note_count']} notes"
                
                if result.warnings:
                    stats += f"\n\n### ⚠️ Warnings:\n"
                    for warning in result.warnings:
                        stats += f"- {warning}\n"
                
                # Validate simple mode structure
                if mode == "simple":
                    # Note: We'd need to pass track_infos through the result
                    # For now, just validate based on output
                    track_types = [t['type'] for t in output_data.get('tracks', [])]
                    if len(track_types) == 2 and 'melody' in track_types and 'chord' in track_types:
                        stats += f"\n\n✅ **Simple Mode Validation:** Structure is correct!"
                    else:
                        stats += f"\n\n⚠️ **Simple Mode Warning:** Expected melody + chord, got: {', '.join(track_types)}"
                
                status = "✅ Success"
                
            except Exception as e:
                stats = f"Processing succeeded but couldn't load output: {e}"
                status = "⚠️ Partial Success"
            
            return (
                status,
                str(result.output_path),
                stats,
                "\n".join(app_state.logs[-20:])  # Last 20 log entries
            )
        else:
            return (
                f"❌ Failed: {result.error_message}",
                "",
                "No statistics available",
                "\n".join(app_state.logs[-20:])
            )
    
    except Exception as e:
        return (
            f"❌ Error: {str(e)}",
            "",
            "Processing failed",
            "\n".join(app_state.logs[-20:])
        )


def process_batch_files(
    input_dir: str,
    output_dir: str,
    mode: str,
    progress=gr.Progress()
) -> Tuple[str, str, str]:
    """
    Process multiple MIDI files from a directory.

    Returns:
        Tuple of (status, summary, logs)
    """
    if not input_dir:
        return "❌ No input directory selected", "", ""

    # Get all MIDI files from input directory
    input_path = Path(input_dir)
    if not input_path.exists() or not input_path.is_dir():
        return "❌ Invalid input directory", "", ""

    file_paths = list(input_path.glob("*.mid")) + list(input_path.glob("*.midi"))
    if not file_paths:
        return "❌ No MIDI files found in directory", "", ""

    # Initialize parser
    if app_state.parser is None or app_state.current_mode != mode:
        app_state.initialize_parser(mode)

    app_state.logs.clear()
    app_state.results.clear()

    output_path = Path(output_dir)
    
    # Progress tracking
    def file_progress_callback(current: int, total: int, filename: str):
        progress(
            current / total,
            desc=f"Processing {current}/{total}: {filename}"
        )
    
    def item_progress_callback(prog: ProcessingProgress):
        # Inner progress (optional, can be more detailed)
        pass
    
    # Process batch
    results = app_state.parser.process_batch(
        file_paths,
        output_path,
        file_progress_callback=file_progress_callback,
        item_progress_callback=item_progress_callback
    )
    
    app_state.results = results
    
    # Build summary
    successful = sum(1 for r in results if r.success)
    failed = len(results) - successful
    total_time = sum(r.processing_time for r in results)
    
    summary = f"""
## 📊 Batch Processing Complete

**Total Files:** {len(results)}
**Successful:** {successful} ✅
**Failed:** {failed} ❌
**Total Time:** {total_time:.1f}s
**Average Time:** {total_time/len(results):.1f}s per file

### Results:
"""
    
    for i, (file_path, result) in enumerate(zip(file_paths, results), 1):
        status_icon = "✅" if result.success else "❌"
        summary += f"\n{i}. {status_icon} **{file_path.name}**"
        if result.success:
            summary += f" → `{result.output_path.name}` ({result.processing_time:.1f}s)"
        else:
            summary += f" → {result.error_message}"
    
    status = f"✅ Batch complete: {successful}/{len(results)} successful"
    
    return (
        status,
        summary,
        "\n".join(app_state.logs[-50:])  # Last 50 log entries for batch
    )


def cancel_processing():
    """Cancel ongoing processing."""
    if app_state.parser:
        app_state.parser.cancel_operation()
        return "⏹️ Cancellation requested..."
    return "No active processing"


# ============================================================================
# Gradio Interface
# ============================================================================

def create_interface():
    """Create the Gradio interface."""

    with gr.Blocks(
        title="Orpheus",
        theme=gr.themes.Soft(primary_hue="blue", secondary_hue="purple")
    ) as app:

        gr.Markdown("# Orpheus")

        with gr.Tabs() as tabs:

            # ================================================================
            # TAB 1: Preprocess
            # ================================================================
            with gr.Tab("Preprocess"):
                gr.Markdown("*Preprocessing functionality coming soon...*")

            # ================================================================
            # TAB 2: Parser
            # ================================================================
            with gr.Tab("Parser"):

                with gr.Row():
                    # Left column - inputs
                    with gr.Column(scale=1):
                        mode_parser = gr.Radio(
                            choices=[
                                ("Simple Mode", "simple"),
                                ("Advanced Mode", "advanced")
                            ],
                            value="simple",
                            label="Mode",
                            scale=1
                        )

                        input_dir_parser = gr.Textbox(
                            label="Input Directory",
                            value="./source_midis",
                            scale=1
                        )

                        output_dir_parser = gr.Textbox(
                            label="Output Directory",
                            value="./processed",
                            scale=1
                        )

                        process_parser_btn = gr.Button(
                            "Process",
                            variant="primary",
                            size="lg"
                        )

                        cancel_parser_btn = gr.Button(
                            "Cancel",
                            variant="stop"
                        )

                    # Right column - results
                    with gr.Column(scale=2):
                        status_parser = gr.Textbox(
                            label="Status",
                            interactive=False,
                            scale=1
                        )

                        summary_parser = gr.Markdown("*No files processed yet*")

                with gr.Accordion("Processing Logs", open=False):
                    logs_parser = gr.Textbox(
                        lines=10,
                        interactive=False,
                        show_label=False
                    )

                # Event handlers
                process_parser_btn.click(
                    fn=process_batch_files,
                    inputs=[input_dir_parser, output_dir_parser, mode_parser],
                    outputs=[status_parser, summary_parser, logs_parser]
                )

                cancel_parser_btn.click(
                    fn=cancel_processing,
                    outputs=status_parser
                )

            # ================================================================
            # TAB 3: Training
            # ================================================================
            with gr.Tab("Training"):
                gr.Markdown("*Training functionality coming soon...*")

    return app


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    
    # Create output directory if it doesn't exist
    Path("./processed").mkdir(exist_ok=True)
    
    # Launch the interface
    app = create_interface()
    
    app.launch(
        server_name="localhost",  # Allow external access
        server_port=7860,
        share=False,  # Set to True to create public link
        show_error=True,
        favicon_path=None,  # Add your icon here if you have one
    )