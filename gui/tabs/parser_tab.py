"""
Parser tab for Orpheus Gradio GUI.

This tab handles MIDI file tokenization and JSON output generation.
Supports both single file and batch processing with progress tracking.
"""

import gradio as gr
from pathlib import Path
from typing import Tuple, List
import json
import gzip

from midi_parser.interface import ProcessingProgress
from gui.state import app_state


# ============================================================================
# Validation Functions
# ============================================================================

def validate_simple_mode_structure(track_infos: List) -> Tuple[bool, str]:
    """
    Validate that MIDI has the expected 2-track structure.
    
    Args:
        track_infos: List of TrackInfo objects
    
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
    
    melody_track = next(t for t in track_infos if t.type == "melody")
    if melody_track.statistics.avg_polyphony > 1.3:
        return False, f"⚠️ Melody track is not monophonic (polyphony: {melody_track.statistics.avg_polyphony:.2f})"
    
    return True, "✅ Structure valid: Monophonic melody + Chord track"


# ============================================================================
# Processing Functions
# ============================================================================

def process_batch_files(
    input_dir: str,
    output_dir: str,
    mode: str,
    compress: bool,
    progress=gr.Progress()
) -> Tuple[str, str, str]:
    """
    Process multiple MIDI files from a directory.

    Args:
        input_dir: Directory containing MIDI files
        output_dir: Output directory for JSON files
        mode: Processing mode ('simple' or 'advanced')
        compress: Whether to compress output files
        progress: Gradio progress tracker

    Returns:
        Tuple of (status, summary, logs)
    """
    if not input_dir:
        return "❌ No input directory selected", "", ""

    input_path = Path(input_dir)
    if not input_path.exists() or not input_path.is_dir():
        return "❌ Invalid input directory", "", ""

    file_paths = list(input_path.glob("*.mid")) + list(input_path.glob("*.midi"))
    if not file_paths:
        return "❌ No MIDI files found in directory", "", ""

    # Initialize or reinitialize parser if settings changed
    needs_reinit = (
        app_state.parser is None or 
        app_state.current_mode != mode or 
        app_state.current_compression != compress
    )
    
    if needs_reinit:
        app_state.initialize_parser(mode, compress)

    app_state.clear_logs()
    app_state.results.clear()

    output_path = Path(output_dir)
    
    def file_progress_callback(current: int, total: int, filename: str):
        """Update progress bar for batch processing."""
        progress(
            current / total,
            desc=f"Processing {current}/{total}: {filename}"
        )
    
    def item_progress_callback(prog: ProcessingProgress):
        """Progress callback for individual file processing."""
        pass
    
    # Process all files
    results = app_state.parser.process_batch(
        file_paths,
        output_path,
        file_progress_callback=file_progress_callback,
        item_progress_callback=item_progress_callback
    )
    
    app_state.results = results

    # Calculate statistics
    successful = sum(1 for r in results if r.success)
    failed = len(results) - successful

    # Count truncated files (successful but with truncation warnings)
    truncated = sum(
        1 for r in results
        if r.success and r.warnings and any("exceeds max length" in w for w in r.warnings)
    )
    successful_no_warnings = successful - truncated

    total_time = sum(r.processing_time for r in results)

    total_size_kb = sum(
        r.output_path.stat().st_size / 1024
        for r in results if r.success and r.output_path
    )

    compression_info = " (compressed)" if compress else " (uncompressed)"

    # Build summary header
    summary = f"""
## Batch Processing Complete

**Total Files:** {len(results)}
**Successful:** {successful_no_warnings} ✅
**Truncated:** {truncated} ⚠️
**Failed:** {failed} ❌
**Total Time:** {total_time:.1f}s
**Average Time:** {total_time/len(results):.1f}s per file
**Total Output Size:** {total_size_kb:.1f} KB{compression_info}

### Results:
"""
    
    for i, (file_path, result) in enumerate(zip(file_paths, results), 1):
        # Determine if file was truncated
        has_truncation_warning = False
        truncation_message = ""
        if result.success and result.warnings:
            for warning in result.warnings:
                if "exceeds max length" in warning:
                    has_truncation_warning = True
                    truncation_message = warning
                    break

        # Set appropriate icon
        if result.success:
            status_icon = "⚠️" if has_truncation_warning else "✅"
        else:
            status_icon = "❌"

        summary += f"\n{i}. {status_icon} **{file_path.name}**"

        if result.success:
            file_size = result.output_path.stat().st_size / 1024
            summary += f" → `{result.output_path.name}` ({file_size:.1f} KB, {result.processing_time:.1f}s)"

            # Add truncation warning if present
            if has_truncation_warning:
                summary += f" - {truncation_message}"
        else:
            summary += f" → {result.error_message}"
    
    status = f"✅ Batch complete: {successful}/{len(results)} successful"
    
    return (
        status,
        summary,
        app_state.get_recent_logs(50)
    )


def cancel_processing() -> str:
    """
    Cancel ongoing processing operation.
    
    Returns:
        Status message
    """
    if app_state.parser:
        app_state.parser.cancel_operation()
        return "Cancellation requested..."
    return "No active processing"


# ============================================================================
# Tab Creation
# ============================================================================

def create_parser_tab() -> gr.Tab:
    """
    Create the parser tab with UI and event handlers.

    Returns:
        Gradio Tab component
    """
    with gr.Tab("MIDI Parser") as tab:
        
        with gr.Row():
            # Left column - Configuration and controls
            with gr.Column(scale=1):
                mode_parser = gr.Radio(
                    choices=[
                        ("Simple Mode", "simple"),
                        ("Advanced Mode", "advanced")
                    ],
                    value="simple",
                    label="Mode",
                    info="Simple mode for melody+chord, Advanced for any MIDI"
                )

                input_dir_parser = gr.Textbox(
                    label="Input Directory",
                    value="./source_midis",
                    placeholder="Path to folder containing MIDI files"
                )

                output_dir_parser = gr.Textbox(
                    label="Output Directory",
                    value="./processed",
                    placeholder="Path where JSON files will be saved"
                )

                compress_json = gr.Checkbox(
                    label="Compress JSON output (.json.gz)",
                    value=False,
                    info="""🔵 Compressed (.json.gz): 5-10x smaller files, great for storage
🟢 Uncompressed (.json): Faster PyTorch DataLoader reads, better for training

For AI training: Uncompressed is recommended as it eliminates decompression overhead during repeated DataLoader reads across epochs. Compressed is better if disk space is limited."""
                )

                with gr.Row():
                    process_parser_btn = gr.Button(
                        "Process Files",
                        variant="primary",
                        size="lg"
                    )

                    cancel_parser_btn = gr.Button(
                        "Cancel",
                        variant="stop"
                    )

            # Right column - Results and status
            with gr.Column(scale=2):
                status_parser = gr.Textbox(
                    label="Status",
                    interactive=False
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
            inputs=[input_dir_parser, output_dir_parser, mode_parser, compress_json],
            outputs=[status_parser, summary_parser, logs_parser]
        )

        cancel_parser_btn.click(
            fn=cancel_processing,
            outputs=status_parser
        )
    
    return tab