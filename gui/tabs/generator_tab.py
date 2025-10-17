"""
Generator tab for Orpheus Gradio GUI.

This tab provides a complete interface for music generation using trained models:
- Model loading and configuration
- Generation parameter controls
- Real-time progress monitoring
- Results display and download
"""

import sys
import gradio as gr
import pandas as pd
from pathlib import Path
from typing import Tuple, List, Dict, Any
import logging

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from gui.state import app_state
from pytorch.generation import (
    GenerationConfig,
    MusicGenerator,
    GenerationResult,
    create_quality_config,
    create_creative_config,
    create_custom_config
)
from pytorch.data.constants import MAJOR_KEYS, MINOR_KEYS

logger = logging.getLogger(__name__)


# ============================================================================
# Backend Functions
# ============================================================================

def list_available_checkpoints(checkpoint_dir: str = "pytorch/checkpoints") -> List[str]:
    """List available model checkpoints."""
    try:
        checkpoint_path = Path(checkpoint_dir)
        if not checkpoint_path.exists():
            return []

        checkpoints = list(checkpoint_path.glob("*.pt"))
        checkpoints.sort(key=lambda p: p.stat().st_mtime, reverse=True)

        return [str(ckpt.name) for ckpt in checkpoints]

    except Exception as e:
        logger.error(f"Error listing checkpoints: {e}")
        return []


def load_model(checkpoint_name: str, checkpoint_dir: str = "pytorch/checkpoints") -> Tuple[str, bool]:
    """Load model from checkpoint."""
    try:
        if not checkpoint_name:
            return "No checkpoint selected", False

        checkpoint_path = Path(checkpoint_dir) / checkpoint_name

        if not checkpoint_path.exists():
            return f"Checkpoint not found: {checkpoint_name}", False

        app_state.add_generation_log(f"Loading model from {checkpoint_name}...")

        # Create generator with default quality config
        config = create_quality_config()
        generator = MusicGenerator(config)

        # Load checkpoint
        if not generator.load_checkpoint(checkpoint_path):
            app_state.add_generation_log("Failed to load checkpoint")
            return "Failed to load checkpoint", False

        # Store in app state
        app_state.initialize_generator(generator)
        app_state.add_generation_log("Model loaded successfully")

        return f"Model loaded: {checkpoint_name}", True

    except Exception as e:
        import traceback
        error_msg = f"Error loading model: {str(e)}\n{traceback.format_exc()}"
        app_state.add_generation_log(error_msg)
        return error_msg, False


def start_generation(
    mode: str,
    num_files: int,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
    max_bars: int,
    output_dir: str,
    progress=gr.Progress()
) -> Tuple[str, pd.DataFrame, str]:
    """Start music generation."""
    try:
        if not app_state.generator_loaded:
            return "No model loaded. Please load a checkpoint first.", pd.DataFrame(), ""

        app_state.generation_active = True
        app_state.add_generation_log(f"Starting generation: {num_files} files in {mode} mode")

        # Create configuration based on mode
        if mode == "Quality (Conservative)":
            config = create_quality_config()
        elif mode == "Creative (Experimental)":
            config = create_creative_config()
        else:  # Custom
            config = create_custom_config(
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty
            )

        # Update config with GUI settings
        config.max_generation_bars = max_bars
        config.output_dir = Path(output_dir)
        config.output_dir.mkdir(parents=True, exist_ok=True)

        # Update generator config
        app_state.generator.config = config

        # Define progress callback
        def progress_callback(current, total, result: GenerationResult):
            progress_value = current / total
            progress(progress_value, desc=f"Generating {current}/{total}")

            # Add result to state
            app_state.add_generation_result({
                'index': current,
                'filename': result.midi_path.name if result.midi_path else "N/A",
                'success': result.success,
                'violations': result.num_violations,
                'time': f"{result.generation_time:.1f}s",
                'bars': result.num_bars,
                'tokens': result.sequence_length,
                'midi_path': str(result.midi_path) if result.midi_path else None
            })

            # Log result
            app_state.add_generation_log(result.get_summary())

        # Generate files
        results = app_state.generator.generate_batch(
            num_files=num_files,
            progress_callback=progress_callback
        )

        app_state.generation_active = False

        # Create results dataframe
        results_data = []
        for i, result in enumerate(results, 1):
            results_data.append({
                '#': i,
                'File': result.midi_path.name if result.midi_path else "Failed",
                'Status': "✓" if result.success else "✗",
                'Bars': result.num_bars,
                'Violations': result.num_violations,
                'Time': f"{result.generation_time:.1f}s"
            })

        results_df = pd.DataFrame(results_data)

        # Generate summary
        successful = sum(1 for r in results if r.success)
        total_time = sum(r.generation_time for r in results)

        status = (
            f"Generation complete! {successful}/{num_files} successful. "
            f"Total time: {total_time:.1f}s. Files saved to {output_dir}"
        )

        app_state.add_generation_log(status)

        return status, results_df, app_state.get_recent_generation_logs()

    except Exception as e:
        import traceback
        error_msg = f"Error during generation: {str(e)}\n{traceback.format_exc()}"
        app_state.add_generation_log(error_msg)
        app_state.generation_active = False
        return error_msg, pd.DataFrame(), app_state.get_recent_generation_logs()


def update_mode_settings(mode: str) -> Tuple[float, float, float, bool, bool, bool]:
    """Update sampling settings based on selected mode."""
    if mode == "Quality (Conservative)":
        return 0.8, 0.95, 1.1, False, False, False
    elif mode == "Creative (Experimental)":
        return 1.1, 0.92, 1.05, False, False, False
    else:  # Custom
        return 0.8, 0.95, 1.1, True, True, True


def get_generation_statistics() -> Dict[str, str]:
    """Get generation statistics from current session."""
    if not app_state.generation_results:
        return {
            'success_rate': "N/A",
            'avg_time': "N/A",
            'total_violations': "N/A"
        }

    successful = sum(1 for r in app_state.generation_results if r['success'])
    total = len(app_state.generation_results)
    success_rate = (successful / total * 100) if total > 0 else 0

    total_time = sum(float(r['time'].replace('s', '')) for r in app_state.generation_results)
    avg_time = total_time / total if total > 0 else 0

    total_violations = sum(r['violations'] for r in app_state.generation_results)

    return {
        'success_rate': f"{success_rate:.1f}%",
        'avg_time': f"{avg_time:.1f}s",
        'total_violations': str(total_violations)
    }


# ============================================================================
# Gradio Interface
# ============================================================================

def create_generator_tab() -> gr.Tab:
    """Create the generator tab with complete UI."""
    with gr.Tab("Generator") as tab:

        gr.Markdown("""
        ## Music Generation

        Generate new MIDI files using trained transformer models.
        """)

        with gr.Row():
            # ===== LEFT COLUMN: Configuration =====
            with gr.Column(scale=1):
                gr.Markdown("### Model")

                # Model loading
                checkpoint_dropdown = gr.Dropdown(
                    label="Model Checkpoint",
                    choices=list_available_checkpoints(),
                    value=None
                )

                with gr.Row():
                    load_model_btn = gr.Button("Load Model", size="sm")
                    refresh_checkpoints_btn = gr.Button("Refresh", size="sm")

                model_status = gr.Textbox(
                    label="Model Status",
                    value="No model loaded",
                    interactive=False,
                    lines=2
                )

                gr.Markdown("### Generation Settings")

                # Mode selection
                mode_dropdown = gr.Dropdown(
                    label="Mode",
                    choices=["Quality (Conservative)", "Creative (Experimental)", "Custom"],
                    value="Quality (Conservative)",
                    info="Quality mode stays close to training data, Creative mode is more experimental"
                )

                # Sampling parameters
                with gr.Accordion("Sampling Parameters", open=False):
                    temperature_slider = gr.Slider(
                        label="Temperature",
                        minimum=0.5,
                        maximum=1.5,
                        value=0.8,
                        step=0.05,
                        interactive=False,
                        info="Higher = more creative, lower = more conservative"
                    )

                    top_p_slider = gr.Slider(
                        label="Nucleus (Top-p)",
                        minimum=0.8,
                        maximum=1.0,
                        value=0.95,
                        step=0.01,
                        interactive=False,
                        info="Probability mass to sample from"
                    )

                    repetition_penalty_slider = gr.Slider(
                        label="Repetition Penalty",
                        minimum=1.0,
                        maximum=1.5,
                        value=1.1,
                        step=0.05,
                        interactive=False,
                        info="Penalty for repeated tokens"
                    )

                # Conditioning (planned but not implemented)
                with gr.Accordion("Conditioning (Coming Soon)", open=False):
                    key_dropdown = gr.Dropdown(
                        label="Key Signature",
                        choices=["Auto"] + MAJOR_KEYS + MINOR_KEYS,
                        value="Auto",
                        interactive=False
                    )

                    tempo_slider = gr.Slider(
                        label="Tempo (BPM)",
                        minimum=90,
                        maximum=140,
                        value=125,
                        step=1,
                        interactive=False
                    )

                    time_sig_dropdown = gr.Dropdown(
                        label="Time Signature",
                        choices=["Auto", "4/4", "6/8"],
                        value="Auto",
                        interactive=False
                    )

                    gr.Markdown("*These features are structured in code but not yet implemented*")

                gr.Markdown("### Output Settings")

                # Generation settings
                num_files_slider = gr.Slider(
                    label="Number of Files",
                    minimum=1,
                    maximum=50,
                    value=5,
                    step=1
                )

                max_bars_slider = gr.Slider(
                    label="Max Length (bars)",
                    minimum=8,
                    maximum=64,
                    value=32,
                    step=4
                )

                output_dir_textbox = gr.Textbox(
                    label="Output Directory",
                    value="./generated"
                )

                # Generate button
                generate_btn = gr.Button(
                    "Generate Music",
                    variant="primary",
                    size="lg"
                )

            # ===== RIGHT COLUMN: Results & Monitoring =====
            with gr.Column(scale=2):
                gr.Markdown("### Generation Status")

                generation_status = gr.Textbox(
                    label="Status",
                    value="Ready to generate",
                    interactive=False,
                    lines=2
                )

                # Live logs
                gr.Markdown("### Logs")

                logs_display = gr.Textbox(
                    label="Generation Logs",
                    value="No logs yet...",
                    interactive=False,
                    lines=10,
                    autoscroll=True
                )

                # Results table
                gr.Markdown("### Generated Files")

                results_table = gr.Dataframe(
                    headers=['#', 'File', 'Status', 'Bars', 'Violations', 'Time'],
                    label="Results",
                    interactive=False
                )

                # Statistics
                gr.Markdown("### Statistics")

                with gr.Row():
                    success_rate_display = gr.Textbox(
                        label="Success Rate",
                        value="N/A",
                        interactive=False
                    )

                    avg_time_display = gr.Textbox(
                        label="Avg Time",
                        value="N/A",
                        interactive=False
                    )

                    total_violations_display = gr.Textbox(
                        label="Total Violations",
                        value="N/A",
                        interactive=False
                    )

                # File download
                with gr.Accordion("Download Files", open=False):
                    download_info = gr.Markdown("Generate files first to enable downloads")

        # ===== Event Handlers =====

        # Refresh checkpoints list
        def on_refresh_checkpoints():
            checkpoints = list_available_checkpoints()
            return gr.Dropdown(choices=checkpoints)

        refresh_checkpoints_btn.click(
            fn=on_refresh_checkpoints,
            outputs=[checkpoint_dropdown]
        )

        # Load model
        load_model_btn.click(
            fn=load_model,
            inputs=[checkpoint_dropdown],
            outputs=[model_status, generate_btn]
        ).then(
            fn=lambda: app_state.get_recent_generation_logs(),
            outputs=[logs_display]
        )

        # Mode change updates sampling parameters
        mode_dropdown.change(
            fn=update_mode_settings,
            inputs=[mode_dropdown],
            outputs=[
                temperature_slider,
                top_p_slider,
                repetition_penalty_slider,
                temperature_slider,
                top_p_slider,
                repetition_penalty_slider
            ]
        )

        # Generate button
        generate_btn.click(
            fn=start_generation,
            inputs=[
                mode_dropdown,
                num_files_slider,
                temperature_slider,
                top_p_slider,
                repetition_penalty_slider,
                max_bars_slider,
                output_dir_textbox
            ],
            outputs=[
                generation_status,
                results_table,
                logs_display
            ]
        ).then(
            fn=lambda: get_generation_statistics(),
            outputs=[
                success_rate_display,
                avg_time_display,
                total_violations_display
            ]
        )

        # Auto-refresh logs every 2 seconds when generation is active
        timer = gr.Timer(2)
        timer.tick(
            fn=lambda: app_state.get_recent_generation_logs() if app_state.generation_active else gr.skip(),
            outputs=[logs_display]
        )

        # Also refresh when tab is selected
        tab.select(
            fn=lambda: app_state.get_recent_generation_logs(),
            outputs=[logs_display]
        )

    return tab
