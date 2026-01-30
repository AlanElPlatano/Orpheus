"""
Custom Chords tab for Orpheus Gradio GUI.

This tab allows users to upload their own MIDI chord files and generate
melodies on top of them using trained transformer models.
"""

import sys
import gradio as gr
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
import logging
import shutil

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from gui.state import app_state
from ml_core.generation.chord_injection import (
    parse_chord_midi,
    ChordValidationResult,
    get_token_count_with_context
)
from ml_core.generation import (
    GenerationConfig,
    create_quality_config,
    create_creative_config,
    create_custom_config
)
from ml_core.generation.midi_export import tokens_to_midi
from ml_core.data.constants import (
    MAJOR_KEYS,
    MINOR_KEYS,
    KEY_TO_ID,
    TEMPO_NONE_VALUE,
    CONDITION_NONE_ID,
    TIME_SIG_TO_ID,
    CONTEXT_LENGTH
)
import torch

logger = logging.getLogger(__name__)


# ============================================================================
# Helper Functions
# ============================================================================

def parse_time_signature(time_sig_str: str) -> Tuple[int, int]:
    """Parse time signature string like '4/4' into tuple (numerator, denominator)."""
    if time_sig_str == "Auto":
        return (0, 0)  # Special case for auto
    parts = time_sig_str.split('/')
    if len(parts) == 2:
        try:
            return (int(parts[0]), int(parts[1]))
        except ValueError:
            pass
    return (4, 4)  # Default


def format_time_signature(time_sig_tuple: Optional[Tuple[int, int]]) -> str:
    """Format time signature tuple as string."""
    if time_sig_tuple is None or time_sig_tuple == (0, 0):
        return "Auto"
    return f"{time_sig_tuple[0]}/{time_sig_tuple[1]}"


# ============================================================================
# Backend Functions
# ============================================================================

def upload_and_parse_chord_midi(
    midi_file: Any  # gr.File upload
) -> Tuple[str, str, str, float, str, str, bool]:
    """
    Handle MIDI file upload, parse, and validate.

    Returns:
        Tuple of (status, validation_info, key_value, tempo_value, time_sig_value, token_info, accept_button_visible)
    """
    try:
        if midi_file is None:
            return (
                "No file uploaded",
                "",
                "Auto",
                125.0,
                "Auto",
                "",
                False
            )

        # Get uploaded file path
        if hasattr(midi_file, 'name'):
            midi_path = Path(midi_file.name)
        else:
            midi_path = Path(midi_file)

        if not midi_path.exists():
            return (
                "❌ File not found",
                "",
                "Auto",
                125.0,
                "Auto",
                "",
                False
            )

        # Create temp directory
        temp_dir = Path("temp")
        temp_dir.mkdir(exist_ok=True)

        # Parse MIDI
        logger.info(f"Parsing uploaded MIDI: {midi_path.name}")
        result = parse_chord_midi(
            midi_path=midi_path,
            temp_dir=temp_dir,
            max_tokens=int(CONTEXT_LENGTH * 0.75)  # Reserve 25% for melody
        )

        # Build status message
        if result.is_valid:
            status = f"✅ Validation successful: {midi_path.name}"
        else:
            status = f"❌ Validation failed: {midi_path.name}"

        # Build validation info
        validation_parts = []

        if result.errors:
            validation_parts.append("**Errors:**")
            for error in result.errors:
                validation_parts.append(f"  - ❌ {error}")

        if result.warnings:
            validation_parts.append("**Warnings:**")
            for warning in result.warnings:
                validation_parts.append(f"  - ⚠️ {warning}")

        if result.is_valid:
            validation_parts.append("\n**Validation Checks:**")
            validation_parts.append("  - ✓ Single track detected")
            validation_parts.append("  - ✓ All tokens within vocabulary")
            validation_parts.append("  - ✓ Chord program enforced (Program_29)")

            # Get token count info
            if result.tokens:
                token_info = get_token_count_with_context(result.tokens, CONTEXT_LENGTH)
                token_summary = (
                    f"**Token Count:** {token_info['chord_tokens']} / {token_info['max_chord_tokens']} "
                    f"({token_info['percentage_used']:.1f}%)\n"
                    f"**Remaining for melody:** {token_info['remaining_for_melody']} tokens"
                )
            else:
                token_summary = ""

        else:
            token_summary = ""

        validation_info = "\n".join(validation_parts)

        # Extract metadata for UI
        key_value = "Auto"
        tempo_value = 125.0
        time_sig_value = "Auto"

        if result.metadata:
            if result.metadata.tempo:
                tempo_value = result.metadata.tempo
            if result.metadata.time_signature:
                time_sig_value = format_time_signature(result.metadata.time_signature)

        # Store in app state
        if result.is_valid:
            app_state.custom_chord_tokens = result.tokens
            app_state.custom_chord_metadata = result.metadata
            app_state.custom_chord_file = midi_path.name
        else:
            app_state.custom_chord_tokens = None
            app_state.custom_chord_metadata = None
            app_state.custom_chord_file = None

        return (
            status,
            validation_info,
            key_value,
            tempo_value,
            time_sig_value,
            token_summary,
            result.is_valid
        )

    except Exception as e:
        import traceback
        error_msg = f"❌ Error parsing MIDI: {str(e)}"
        logger.error(f"{error_msg}\n{traceback.format_exc()}")
        return (
            error_msg,
            f"**Error Details:**\n{traceback.format_exc()}",
            "Auto",
            125.0,
            "Auto",
            "",
            False
        )


def generate_melody_from_custom_chords(
    mode: str,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
    max_bars: int,
    output_dir: str,
    key_signature: str,
    tempo: float,
    time_signature: str,
    progress=gr.Progress()
) -> Tuple[str, str, Optional[str]]:
    """
    Generate melody on top of user-provided chords.

    Returns:
        Tuple of (status, logs, output_midi_path)
    """
    try:
        # Validate generator is loaded
        if not app_state.generator_loaded or app_state.generator is None:
            return (
                "❌ No model loaded. Please load a checkpoint in the Generator tab first.",
                "",
                None
            )

        # Validate chord tokens are present
        if not hasattr(app_state, 'custom_chord_tokens') or app_state.custom_chord_tokens is None:
            return (
                "❌ No chord file loaded. Please upload and validate a chord MIDI file first.",
                "",
                None
            )

        app_state.add_generation_log(f"Starting melody generation from custom chords: {app_state.custom_chord_file}")
        progress(0.1, desc="Initializing generation...")

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

        # Apply conditioning
        config.key = None if key_signature == "Auto" else key_signature
        config.tempo = None if tempo == 0.0 else tempo
        config.time_signature = None if time_signature == "Auto" else parse_time_signature(time_signature)

        # Update generator config
        app_state.generator.config = config

        # Create conditioning tensors
        device = app_state.generator.device

        # Key conditioning
        if config.key and config.key in KEY_TO_ID:
            key_id = KEY_TO_ID[config.key]
        else:
            key_id = CONDITION_NONE_ID
        key_ids = torch.tensor([key_id], dtype=torch.long, device=device)

        # Tempo conditioning
        if config.tempo:
            tempo_value = float(config.tempo)
        else:
            tempo_value = TEMPO_NONE_VALUE
        tempo_values = torch.tensor([tempo_value], dtype=torch.float32, device=device)

        # Time signature conditioning
        if config.time_signature and config.time_signature in TIME_SIG_TO_ID:
            time_sig_id = TIME_SIG_TO_ID[config.time_signature]
        else:
            time_sig_id = CONDITION_NONE_ID
        time_sig_ids = torch.tensor([time_sig_id], dtype=torch.long, device=device)

        progress(0.3, desc="Generating melody...")

        # Generate melody using custom chords
        generated_tokens = app_state.generator.two_stage_gen.generate_melody_from_chords(
            chord_tokens=app_state.custom_chord_tokens,
            seed=config.seed,
            temperature=config.temperature,
            key_ids=key_ids,
            tempo_values=tempo_values,
            time_sig_ids=time_sig_ids
        )

        progress(0.7, desc="Converting to MIDI...")

        # Generate output filename
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        key_str = config.key if config.key else "auto"
        tempo_str = f"{int(config.tempo)}bpm" if config.tempo else "autobpm"
        output_filename = f"custom_chords_{key_str}_{tempo_str}_{timestamp}.mid"
        output_path = config.output_dir / output_filename

        # Convert tokens to MIDI
        success, midi_path, error_msg = tokens_to_midi(
            token_ids=generated_tokens,
            vocab_info=app_state.generator.vocab_info,
            output_path=output_path
        )

        if not success:
            app_state.add_generation_log(f"Failed to export MIDI: {error_msg}")
            return (
                f"❌ Generation succeeded but MIDI export failed: {error_msg}",
                app_state.get_recent_generation_logs(),
                None
            )

        progress(1.0, desc="Complete!")

        status = f"✅ Melody generated successfully! File saved: {output_filename}"
        app_state.add_generation_log(status)
        app_state.add_generation_log(f"  - Tokens: {len(generated_tokens)}")
        app_state.add_generation_log(f"  - Output: {output_path}")

        return (
            status,
            app_state.get_recent_generation_logs(),
            str(midi_path) if midi_path else None
        )

    except Exception as e:
        import traceback
        error_msg = f"❌ Error during generation: {str(e)}\n{traceback.format_exc()}"
        app_state.add_generation_log(error_msg)
        logger.error(error_msg)
        return (
            error_msg,
            app_state.get_recent_generation_logs(),
            None
        )


def update_mode_settings(mode: str) -> Tuple[gr.update, gr.update, gr.update]:
    """Update sampling settings based on selected mode."""
    if mode == "Quality (Conservative)":
        return (
            gr.update(value=0.8, interactive=False),
            gr.update(value=0.95, interactive=False),
            gr.update(value=1.1, interactive=False)
        )
    elif mode == "Creative (Experimental)":
        return (
            gr.update(value=1.1, interactive=False),
            gr.update(value=0.92, interactive=False),
            gr.update(value=1.05, interactive=False)
        )
    else:  # Custom
        return (
            gr.update(value=0.8, interactive=True),
            gr.update(value=0.95, interactive=True),
            gr.update(value=1.1, interactive=True)
        )


# ============================================================================
# Gradio Interface
# ============================================================================

def create_custom_chords_tab() -> gr.Tab:
    """Create the custom chords tab with complete UI."""
    with gr.Tab("Custom Chords") as tab:

        gr.Markdown("""
        ## Generate Melody from Custom Chords

        Provide your own MIDI file with chords and choose a model to generate a melody on top of it.
        Your chords will be used as reference and preserved in the output.
        """)

        with gr.Row():
            # ===== LEFT COLUMN: Upload & Configuration =====
            with gr.Column(scale=1):
                gr.Markdown("### 1. Upload Chord MIDI")

                # File upload
                chord_file_upload = gr.File(
                    label="Chord MIDI File",
                    file_types=[".mid", ".midi"],
                    type="filepath"
                )

                upload_status = gr.Textbox(
                    label="Upload Status",
                    value="No file uploaded",
                    interactive=False,
                    lines=2
                )

                gr.Markdown("### 2. Verify Metadata")

                gr.Markdown("*Adjust the extracted metadata if needed, then click Accept.*")

                # Metadata controls
                key_dropdown = gr.Dropdown(
                    label="Key Signature",
                    choices=["Auto"] + MAJOR_KEYS + MINOR_KEYS,
                    value="Auto",
                    interactive=True,
                    info="Key for melody generation (Auto = unconditioned)"
                )

                tempo_slider = gr.Slider(
                    label="Tempo (BPM)",
                    minimum=90,
                    maximum=140,
                    value=125,
                    step=1,
                    interactive=True,
                    info="Target tempo"
                )

                time_sig_dropdown = gr.Dropdown(
                    label="Time Signature",
                    choices=["Auto", "4/4", "6/8", "3/4"],
                    value="Auto",
                    interactive=True,
                    info="Time signature constraint"
                )

                accept_metadata_btn = gr.Button(
                    "Accept Metadata",
                    variant="secondary",
                    visible=False
                )

                gr.Markdown("### 3. Generation Settings")

                # Mode selection
                mode_dropdown = gr.Dropdown(
                    label="Mode",
                    choices=["Quality (Conservative)", "Creative (Experimental)", "Custom"],
                    value="Quality (Conservative)",
                    info="Quality mode stays close to training data"
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
                        info="Higher = more creative"
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

                gr.Markdown("### 4. Output Settings")

                max_bars_slider = gr.Slider(
                    label="Max Length (bars)",
                    minimum=8,
                    maximum=64,
                    value=32,
                    step=4,
                    info="Maximum generation length"
                )

                output_dir_textbox = gr.Textbox(
                    label="Output Directory",
                    value="./generated",
                    info="Where to save the generated MIDI"
                )

                # Generate button
                generate_btn = gr.Button(
                    "Generate Melody",
                    variant="primary",
                    size="lg"
                )

            # ===== RIGHT COLUMN: Validation & Results =====
            with gr.Column(scale=2):
                gr.Markdown("### Validation Results")

                validation_display = gr.Markdown(
                    "*Upload a chord MIDI file to see validation results*"
                )

                token_count_display = gr.Markdown("")

                gr.Markdown("### Generation Status")

                generation_status = gr.Textbox(
                    label="Status",
                    value="Ready to generate",
                    interactive=False,
                    lines=2
                )

                # Logs
                gr.Markdown("### Logs")

                logs_display = gr.Textbox(
                    label="Generation Logs",
                    value="No logs yet...",
                    interactive=False,
                    lines=8,
                    autoscroll=True
                )

                # Output file
                gr.Markdown("### Output")

                output_file_display = gr.File(
                    label="Generated MIDI",
                    interactive=False
                )

        # ===== Event Handlers =====

        # File upload handler
        chord_file_upload.upload(
            fn=upload_and_parse_chord_midi,
            inputs=[chord_file_upload],
            outputs=[
                upload_status,
                validation_display,
                key_dropdown,
                tempo_slider,
                time_sig_dropdown,
                token_count_display,
                accept_metadata_btn
            ]
        )

        # Accept metadata button (currently just logs)
        accept_metadata_btn.click(
            fn=lambda: "Metadata accepted. Ready to generate!",
            outputs=[upload_status]
        )

        # Mode change updates sampling parameters
        mode_dropdown.change(
            fn=update_mode_settings,
            inputs=[mode_dropdown],
            outputs=[
                temperature_slider,
                top_p_slider,
                repetition_penalty_slider
            ]
        )

        # Generate button
        generate_btn.click(
            fn=generate_melody_from_custom_chords,
            inputs=[
                mode_dropdown,
                temperature_slider,
                top_p_slider,
                repetition_penalty_slider,
                max_bars_slider,
                output_dir_textbox,
                key_dropdown,
                tempo_slider,
                time_sig_dropdown
            ],
            outputs=[
                generation_status,
                logs_display,
                output_file_display
            ]
        )

        # Refresh logs when tab is selected
        tab.select(
            fn=lambda: app_state.get_recent_generation_logs() if hasattr(app_state, 'generation_logs') else "No logs yet...",
            outputs=[logs_display]
        )

    return tab
