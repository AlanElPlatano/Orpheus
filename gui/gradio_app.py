"""
Gradio GUI for MIDI Parser - Orpheus Project

A comprehensive interface for processing MIDI files into tokenized JSON
for AI music generation training.

Usage:
    python gui/gradio_app.py
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
    files: List[str],
    output_dir: str,
    mode: str,
    progress=gr.Progress()
) -> Tuple[str, str, str]:
    """
    Process multiple MIDI files.
    
    Returns:
        Tuple of (status, summary, logs)
    """
    if not files:
        return "❌ No files selected", "", ""
    
    # Initialize parser
    if app_state.parser is None or app_state.current_mode != mode:
        app_state.initialize_parser(mode)
    
    app_state.logs.clear()
    app_state.results.clear()
    
    output_path = Path(output_dir)
    file_paths = [Path(f) for f in files]
    
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
        title="MIDI Parser - Orpheus",
        theme=gr.themes.Soft(primary_hue="blue", secondary_hue="purple")
    ) as app:
        
        gr.Markdown(
            """
            # 🎵 MIDI Parser - Orpheus Project
            
            Transform MIDI files into tokenized JSON for AI music generation training.
            """
        )
        
        with gr.Tabs() as tabs:
            
            # ================================================================
            # TAB 1: Single File Processing
            # ================================================================
            with gr.Tab("📄 Single File"):
                
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### Input Configuration")
                        
                        mode_single = gr.Radio(
                            choices=[
                                ("🎯 Simple Mode (Melody + Chord)", "simple"),
                                ("🔧 Advanced Mode (General MIDI)", "advanced")
                            ],
                            value="simple",
                            label="Processing Mode",
                            info="Simple mode enforces 2-track structure for music generation"
                        )
                        
                        file_input = gr.File(
                            label="Upload MIDI File",
                            file_types=[".mid", ".midi"],
                            type="filepath"
                        )
                        
                        output_dir_single = gr.Textbox(
                            label="Output Directory",
                            value="./processed",
                            info="Where to save the JSON output"
                        )
                        
                        with gr.Row():
                            process_btn = gr.Button(
                                "🚀 Process File",
                                variant="primary",
                                size="lg"
                            )
                            cancel_btn = gr.Button(
                                "⏹️ Cancel",
                                variant="stop"
                            )
                    
                    with gr.Column(scale=2):
                        gr.Markdown("### Processing Results")
                        
                        status_single = gr.Textbox(
                            label="Status",
                            interactive=False
                        )
                        
                        output_path_single = gr.Textbox(
                            label="Output File Path",
                            interactive=False
                        )
                        
                        stats_single = gr.Markdown("*No file processed yet*")
                
                with gr.Accordion("📋 Processing Logs", open=False):
                    logs_single = gr.Textbox(
                        label="Logs",
                        lines=10,
                        interactive=False,
                        show_label=False
                    )
                
                # Event handlers
                process_btn.click(
                    fn=process_single_file,
                    inputs=[file_input, output_dir_single, mode_single],
                    outputs=[status_single, output_path_single, stats_single, logs_single]
                )
                
                cancel_btn.click(
                    fn=cancel_processing,
                    outputs=status_single
                )
            
            # ================================================================
            # TAB 2: Batch Processing
            # ================================================================
            with gr.Tab("📁 Batch Processing"):
                
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### Batch Configuration")
                        
                        mode_batch = gr.Radio(
                            choices=[
                                ("🎯 Simple Mode", "simple"),
                                ("🔧 Advanced Mode", "advanced")
                            ],
                            value="simple",
                            label="Processing Mode"
                        )
                        
                        files_input = gr.File(
                            label="Upload MIDI Files",
                            file_count="multiple",
                            file_types=[".mid", ".midi"],
                            type="filepath"
                        )
                        
                        output_dir_batch = gr.Textbox(
                            label="Output Directory",
                            value="./processed",
                        )
                        
                        with gr.Row():
                            process_batch_btn = gr.Button(
                                "🚀 Process Batch",
                                variant="primary",
                                size="lg"
                            )
                            cancel_batch_btn = gr.Button(
                                "⏹️ Cancel",
                                variant="stop"
                            )
                    
                    with gr.Column(scale=2):
                        gr.Markdown("### Batch Results")
                        
                        status_batch = gr.Textbox(
                            label="Status",
                            interactive=False
                        )
                        
                        summary_batch = gr.Markdown("*No batch processed yet*")
                
                with gr.Accordion("📋 Batch Logs", open=False):
                    logs_batch = gr.Textbox(
                        label="Logs",
                        lines=15,
                        interactive=False,
                        show_label=False
                    )
                
                # Event handlers
                process_batch_btn.click(
                    fn=process_batch_files,
                    inputs=[files_input, output_dir_batch, mode_batch],
                    outputs=[status_batch, summary_batch, logs_batch]
                )
                
                cancel_batch_btn.click(
                    fn=cancel_processing,
                    outputs=status_batch
                )
            
            # ================================================================
            # TAB 3: Configuration & Help
            # ================================================================
            with gr.Tab("⚙️ Configuration & Help"):
                
                gr.Markdown(
                    """
                    ## 🎯 Simple Mode vs Advanced Mode
                    
                    ### Simple Mode (Recommended for Music Generation)
                    - **Purpose:** Process MIDI files with exactly 2 tracks
                    - **Structure:** Monophonic melody + Chord track
                    - **Use Case:** Training AI for demo generation
                    - **Validation:** Strict checks for proper structure
                    - **Output:** Optimized token sequences for learning
                    
                    ### Advanced Mode (General Purpose)
                    - **Purpose:** Process any MIDI file
                    - **Structure:** Multiple tracks, any type
                    - **Use Case:** General MIDI analysis and tokenization
                    - **Validation:** Flexible, preserves all content
                    - **Output:** Comprehensive tokenization with metadata
                    
                    ---
                    
                    ## 📊 Output Format
                    
                    Processed files are saved as JSON (optionally compressed) with:
                    - Token sequences for each track
                    - Metadata (tempo, time signature, key)
                    - Track information (type, statistics)
                    - Vocabulary information
                    
                    **File naming:** `{KEY}-{TEMPO}bpm-{TOKENIZATION}-{title}.json`
                    
                    Example: `Cmajor-120bpm-remi-my_song.json`
                    
                    ---
                    
                    ## 🎼 Preparing MIDI Files for Simple Mode
                    
                    1. **Melody Track:**
                       - Must be strictly monophonic (one note at a time)
                       - Should contain the main melodic line
                       - Typical instruments: Piano, Synth Lead, Voice
                    
                    2. **Chord Track:**
                       - Should have sustained chords (no rhythmic variation)
                       - Each chord sustains until the next chord starts
                       - Typical instruments: Piano, Pad, Strings
                    
                    3. **Track Naming:**
                       - Name tracks clearly: "Melody", "Chords", etc.
                       - Supports multiple languages
                       - Auto-detection based on musical characteristics
                    
                    ---
                    
                    ## 💡 Tips for Best Results
                    
                    - **Clean your MIDI files** before processing (use preprocessor)
                    - **Remove empty tracks** to avoid noise
                    - **Use consistent tempo** for better tokenization
                    - **Keep files under 50MB** for optimal performance
                    - **Batch process** similar files together
                    
                    ---
                    
                    ## 🚨 Troubleshooting
                    
                    **"Expected 2 tracks, found X"**
                    → Use preprocessor to reduce tracks or switch to Advanced Mode
                    
                    **"Melody track is not monophonic"**
                    → Remove overlapping notes or adjust polyphony threshold
                    
                    **"Processing takes too long"**
                    → Try reducing file size or using simpler tokenization
                    
                    **"Memory warning"**
                    → Close other applications or process smaller files
                    """
                )
        
        gr.Markdown(
            """
            ---
            
            **Orpheus MIDI Parser** | Built with ❤️ for AI Music Generation
            
            *See logs tab for detailed processing information*
            """
        )
    
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
        server_name="0.0.0.0",  # Allow external access
        server_port=7860,
        share=False,  # Set to True to create public link
        show_error=True,
        favicon_path=None,  # Add your icon here if you have one
    )