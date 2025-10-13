"""
JSON to MIDI tab for Orpheus Gradio GUI.

This tab handles batch conversion of JSON tokenized files back to MIDI format.
Supports batch processing with progress tracking.
"""

import gradio as gr
from pathlib import Path
from typing import Tuple, List, Optional
import logging
import time
import tempfile
import os

from miditoolkit import MidiFile

from midi_parser.config.defaults import MidiParserConfig, get_preset_config
from midi_parser.core.tokenizer_manager import TokenizerManager
from midi_parser.core.json_serializer import load_tokenized_json

logger = logging.getLogger(__name__)


# ============================================================================
# JSON to MIDI Converter Class
# ============================================================================

class JSONToMIDIConverter:
    """Handles conversion of JSON tokenized files back to MIDI format."""

    def __init__(self, config: Optional[MidiParserConfig] = None):
        """
        Initialize the converter.

        Args:
            config: MIDI parser configuration (uses default if None)
        """
        self.config = config or self._create_default_config()
        self.tokenizer_manager = TokenizerManager(self.config)
        self._cancel_requested = False

    def _create_default_config(self) -> MidiParserConfig:
        """Create default configuration for conversion."""
        base_config = MidiParserConfig()

        try:
            corrido_preset = get_preset_config("use_case_presets", "corrido_demo")

            base_config.tokenizer.pitch_range = tuple(
                corrido_preset.get("config_overrides", {})
                .get("tokenizer", {})
                .get("pitch_range", [36, 84])
            )
            base_config.tokenizer.beat_resolution = (
                corrido_preset.get("config_overrides", {})
                .get("tokenizer", {})
                .get("beat_resolution", 4)
            )
            base_config.tokenizer.num_velocities = (
                corrido_preset.get("config_overrides", {})
                .get("tokenizer", {})
                .get("num_velocities", 8)
            )

            logger.info("Using corrido_demo preset configuration")
        except (ValueError, KeyError) as e:
            logger.warning(f"Could not load corrido_demo preset: {e}. Using defaults.")

        return base_config

    def cancel_operation(self):
        """Request cancellation of current operation."""
        self._cancel_requested = True

    def json_to_midi(
        self,
        json_path: Path,
        output_dir: Path
    ) -> Tuple[bool, Optional[Path], Optional[str]]:
        """
        Convert JSON file back to MIDI format using tokens.

        Args:
            json_path: Path to input JSON file
            output_dir: Directory to save output MIDI file

        Returns:
            Tuple of (success, midi_path, error_message)
        """
        try:
            # Check for cancellation
            if self._cancel_requested:
                return False, None, "Operation cancelled"

            # Load JSON data
            json_data = load_tokenized_json(json_path)

            # Get tokenization strategy
            strategy = json_data.get("tokenization", "REMI")

            # Use the tokenizer config that was stored in the JSON during encoding
            # This makes sure the tokenizer vocabulary matches the tokens being decoded
            # Basically, makes round-trip possible by speaking the same language
            stored_config = json_data.get("tokenizer_config", {})

            if stored_config:
                # Create a new config based on the stored configuration
                from midi_parser.config.defaults import TokenizerConfig
                tokenizer_config = TokenizerConfig()

                # Apply stored configuration values
                if "pitch_range" in stored_config:
                    tokenizer_config.pitch_range = tuple(stored_config["pitch_range"])
                if "beat_resolution" in stored_config:
                    tokenizer_config.beat_resolution = stored_config["beat_resolution"]
                if "num_velocities" in stored_config:
                    tokenizer_config.num_velocities = stored_config["num_velocities"]
                if "additional_tokens" in stored_config:
                    tokenizer_config.additional_tokens = stored_config["additional_tokens"]
                if "max_seq_length" in stored_config:
                    tokenizer_config.max_seq_length = stored_config["max_seq_length"]

                logger.info(f"Using stored tokenizer config from JSON:")
                logger.info(f"  - pitch_range: {tokenizer_config.pitch_range}")
                logger.info(f"  - beat_resolution: {tokenizer_config.beat_resolution}")
                logger.info(f"  - num_velocities: {tokenizer_config.num_velocities}")

                # Create tokenizer with the stored configuration
                tokenizer = self.tokenizer_manager.create_tokenizer(strategy, tokenizer_config)
            else:
                # Fallback: use default config if no stored config found
                logger.warning("No tokenizer_config found in JSON, using default configuration")
                tokenizer = self.tokenizer_manager.create_tokenizer(strategy)

            # Get the global token sequence
            global_tokens = json_data.get("global_tokens", [])

            if not global_tokens:
                return False, None, "No global_tokens found in JSON"

            logger.info(f"Reconstructing from {len(global_tokens)} global tokens")

            # Decode the global token sequence
            try:
                score = tokenizer.decode(global_tokens)
                logger.info(f"Decoded score with {len(score.tracks)} tracks")

                # Convert symusic Score to miditoolkit MidiFile
                with tempfile.NamedTemporaryFile(mode='wb', suffix='.mid', delete=False) as tmp_file:
                    tmp_path = tmp_file.name

                try:
                    score.dump_midi(tmp_path)
                    reconstructed_midi = MidiFile(tmp_path)
                finally:
                    try:
                        os.unlink(tmp_path)
                    except:
                        pass

            except Exception as e:
                logger.error(f"Failed to decode tokens: {e}")
                return False, None, f"Failed to decode tokens: {str(e)}"

            # Update track metadata from JSON
            tracks_data = json_data.get("tracks", [])
            total_notes_reconstructed = 0

            logger.info(f"Reconstructed MIDI has {len(reconstructed_midi.instruments)} tracks, "
                       f"original had {len(tracks_data)} tracks")

            for i, instrument in enumerate(reconstructed_midi.instruments):
                if i < len(tracks_data):
                    track_data = tracks_data[i]
                    instrument.name = track_data.get("name", f"Track_{i}")
                    instrument.program = track_data.get("program", 0)
                    instrument.is_drum = track_data.get("is_drum", False)

                total_notes_reconstructed += len(instrument.notes)
                logger.info(f"  Track {i}: {instrument.name} ({len(instrument.notes)} notes)")

            if not reconstructed_midi.instruments:
                return False, None, "No instruments in reconstructed MIDI"

            # Generate output filename
            output_filename = json_path.stem.replace('.json', '') + '_reconstructed.mid'
            output_path = output_dir / output_filename

            # Save the reconstructed MIDI
            reconstructed_midi.dump(str(output_path))

            logger.info(f"Reconstructed MIDI saved: {output_path.name}")
            logger.info(f"  Total tracks: {len(reconstructed_midi.instruments)}")
            logger.info(f"  Total notes: {total_notes_reconstructed}")

            return True, output_path, None

        except Exception as e:
            error_msg = f"Error during JSON to MIDI conversion: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return False, None, error_msg

    def process_batch(
        self,
        json_files: List[Path],
        output_dir: Path,
        progress_callback=None
    ) -> List[dict]:
        """
        Process multiple JSON files in batch.

        Args:
            json_files: List of JSON file paths
            output_dir: Output directory for MIDI files
            progress_callback: Callback for progress updates (current, total, filename)

        Returns:
            List of result dictionaries
        """
        results = []
        output_dir.mkdir(parents=True, exist_ok=True)

        for i, json_path in enumerate(json_files, 1):
            if self._cancel_requested:
                results.append({
                    "json_file": json_path.name,
                    "success": False,
                    "error": "Operation cancelled",
                    "processing_time": 0
                })
                continue

            if progress_callback:
                progress_callback(i, len(json_files), json_path.name)

            start_time = time.time()
            success, midi_path, error = self.json_to_midi(json_path, output_dir)
            processing_time = time.time() - start_time

            results.append({
                "json_file": json_path.name,
                "success": success,
                "midi_file": midi_path.name if midi_path else None,
                "error": error,
                "processing_time": processing_time
            })

        self._cancel_requested = False
        return results


# ============================================================================
# Global converter instance
# ============================================================================

_converter: Optional[JSONToMIDIConverter] = None


def get_converter() -> JSONToMIDIConverter:
    """Get or create the global converter instance."""
    global _converter
    if _converter is None:
        _converter = JSONToMIDIConverter()
    return _converter


# ============================================================================
# Processing Functions
# ============================================================================

def process_json_batch(
    input_dir: str,
    output_dir: str,
    progress=gr.Progress()
) -> Tuple[str, str]:
    """
    Process multiple JSON files from a directory and convert them to MIDI.

    Args:
        input_dir: Directory containing JSON files
        output_dir: Output directory for MIDI files
        progress: Gradio progress tracker

    Returns:
        Tuple of (status, summary)
    """
    if not input_dir:
        return "No input directory selected", ""

    input_path = Path(input_dir)
    if not input_path.exists() or not input_path.is_dir():
        return "Invalid input directory", ""

    # Find all JSON files (including .json.gz)
    json_files = list(input_path.glob("*.json")) + list(input_path.glob("*.json.gz"))
    if not json_files:
        return "No JSON files found in directory", ""

    output_path = Path(output_dir)
    converter = get_converter()

    def file_progress_callback(current: int, total: int, filename: str):
        """Update progress bar for batch processing."""
        progress(
            current / total,
            desc=f"Converting {current}/{total}: {filename}"
        )

    # Process all files
    results = converter.process_batch(
        json_files,
        output_path,
        progress_callback=file_progress_callback
    )

    # Calculate statistics
    successful = sum(1 for r in results if r["success"])
    failed = len(results) - successful
    total_time = sum(r["processing_time"] for r in results)

    # Build summary
    summary = f"""
## Batch Conversion Complete

**Total Files:** {len(results)}
**Successful:** {successful}
**Failed:** {failed}
**Total Time:** {total_time:.1f}s
**Average Time:** {total_time/len(results):.1f}s per file

### Results:
"""

    for i, result in enumerate(results, 1):
        status_icon = "" if result["success"] else ""
        summary += f"\n{i}. {status_icon} **{result['json_file']}**"
        if result["success"]:
            summary += f" → `{result['midi_file']}` ({result['processing_time']:.1f}s)"
        else:
            summary += f" → {result['error']}"

    status = f"Batch complete: {successful}/{len(results)} successful"

    return status, summary


def cancel_conversion() -> str:
    """
    Cancel ongoing conversion operation.

    Returns:
        Status message
    """
    converter = get_converter()
    converter.cancel_operation()
    return "Cancellation requested..."


# ============================================================================
# Tab Creation
# ============================================================================

def create_json_to_midi_tab() -> gr.Tab:
    """
    Create the JSON to MIDI tab with UI and event handlers.

    Returns:
        Gradio Tab component
    """
    with gr.Tab("JSON to MIDI") as tab:

        gr.Markdown("""
        ### Convert tokenized JSON files back to MIDI format

        This tab performs batch conversion of JSON files (generated by the Parser tab) back into MIDI files.
        All JSON files in the input folder will be processed and saved to the output folder.
        """)

        with gr.Row():
            # Left column - Configuration and controls
            with gr.Column(scale=1):
                input_dir_json = gr.Textbox(
                    label="Input Directory (JSON files)",
                    value="./processed",
                    placeholder="Path to folder containing JSON files"
                )

                output_dir_json = gr.Textbox(
                    label="Output Directory (MIDI files)",
                    value="./generated",
                    placeholder="Path where MIDI files will be saved"
                )

                with gr.Row():
                    process_json_btn = gr.Button(
                        "Convert to MIDI",
                        variant="primary",
                        size="lg"
                    )

                    cancel_json_btn = gr.Button(
                        "Cancel",
                        variant="stop"
                    )

            # Right column - Results and status
            with gr.Column(scale=2):
                status_json = gr.Textbox(
                    label="Status",
                    interactive=False
                )

                summary_json = gr.Markdown("*No files converted yet*")

        # Event handlers
        process_json_btn.click(
            fn=process_json_batch,
            inputs=[input_dir_json, output_dir_json],
            outputs=[status_json, summary_json]
        )

        cancel_json_btn.click(
            fn=cancel_conversion,
            outputs=status_json
        )

    return tab
