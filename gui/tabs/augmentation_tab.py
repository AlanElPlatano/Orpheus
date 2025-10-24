"""
Data Augmentation tab for Orpheus Gradio GUI.

This tab handles transposing tokenized JSON files to all 12 chromatic keys
for data augmentation purposes.
"""

import gradio as gr
from pathlib import Path
from typing import Tuple, List
import json

from augmentation.transpose_tokenized_json import (
    batch_transpose_directory,
    SEMITONE_OFFSETS
)


def transpose_batch_files(
    input_dir: str,
    output_dir: str,
    mode: str,
    overwrite: bool,
    progress=gr.Progress()
) -> Tuple[str, str]:
    """
    Transpose all JSON files in a directory to multiple keys.

    Args:
        input_dir: Directory containing tokenized JSON files
        output_dir: Output directory for transposed files
        mode: Transposition mode ('all', 'higher', 'lower')
        overwrite: Whether to overwrite existing files
        progress: Gradio progress tracker

    Returns:
        Tuple of (status, summary)
    """
    if not input_dir:
        return "❌ No input directory selected", ""

    input_path = Path(input_dir)
    if not input_path.exists() or not input_path.is_dir():
        return "❌ Invalid input directory", ""

    json_files = list(input_path.glob("*.json"))

    if not json_files:
        return "❌ No JSON files found in directory", ""

    progress(0, desc="Starting transposition...")

    semitone_offsets = SEMITONE_OFFSETS.get(mode, SEMITONE_OFFSETS['default'])

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    total_operations = len(json_files) * len(semitone_offsets)
    current_operation = 0

    successful = 0
    failed = 0
    skipped = 0
    output_files = []

    for json_file in json_files:
        if '-transpose' in json_file.stem:
            current_operation += len(semitone_offsets)
            skipped += len(semitone_offsets)
            continue

        for semitones in semitone_offsets:
            if semitones == 0:
                current_operation += 1
                continue

            current_operation += 1
            progress(
                current_operation / total_operations,
                desc=f"Transposing {json_file.name} by {semitones:+d} semitones..."
            )

            try:
                from augmentation.transpose_tokenized_json import transpose_json_file

                output_file = transpose_json_file(
                    json_file,
                    output_path,
                    semitones,
                    overwrite
                )

                if output_file:
                    successful += 1
                    output_files.append(output_file)
                else:
                    failed += 1

            except Exception as e:
                failed += 1

    summary = f"""
## Data Augmentation Complete

**Input Files:** {len([f for f in json_files if '-transpose' not in f.stem])}
**Transpositions per File:** {len(semitone_offsets)}
**Total Operations:** {successful + failed}
**Successful:** {successful} ✅
**Failed:** {failed} ❌
**Skipped (already transposed):** {skipped}

**Semitone Offsets Used:** {semitone_offsets}

**Output Directory:** `{output_dir}`

### Dataset Expansion:
- Original files: {len([f for f in json_files if '-transpose' not in f.stem])}
- Augmented files created: {successful}
- **Total dataset size: {len([f for f in json_files if '-transpose' not in f.stem]) + successful} files**
"""

    status = f"✅ Augmentation complete: {successful} files created, {failed} failed"

    return status, summary


def clean_augmented_files(
    input_dir: str,
    progress=gr.Progress()
) -> Tuple[str, str]:
    """
    Delete all transposed files (files with '-transpose' in name) from directory.

    Args:
        input_dir: Directory containing JSON files
        progress: Gradio progress tracker

    Returns:
        Tuple of (status, summary)
    """
    if not input_dir:
        return "❌ No input directory selected", ""

    input_path = Path(input_dir)
    if not input_path.exists() or not input_path.is_dir():
        return "❌ Invalid input directory", ""

    progress(0, desc="Scanning for augmented files...")

    json_files = list(input_path.glob("*-transpose*.json"))

    if not json_files:
        return "ℹ️ No augmented files found to clean", "No files with '-transpose' pattern found in directory."

    progress(0.3, desc=f"Found {len(json_files)} augmented files...")

    deleted_count = 0
    failed_count = 0
    deleted_files = []

    for i, json_file in enumerate(json_files):
        progress(
            0.3 + (0.7 * (i + 1) / len(json_files)),
            desc=f"Deleting {i + 1}/{len(json_files)}: {json_file.name}"
        )

        try:
            json_file.unlink()
            deleted_count += 1
            deleted_files.append(json_file.name)
        except Exception as e:
            failed_count += 1

    summary = f"""
## Cleanup Complete

**Augmented Files Found:** {len(json_files)}
**Deleted:** {deleted_count} ✅
**Failed:** {failed_count} ❌

### Files Deleted:
"""

    for filename in deleted_files[:20]:
        summary += f"\n- {filename}"

    if len(deleted_files) > 20:
        summary += f"\n- ... and {len(deleted_files) - 20} more"

    status = f"✅ Cleanup complete: {deleted_count} augmented files removed"

    return status, summary


def get_directory_stats(input_dir: str) -> str:
    """
    Get statistics about JSON files in the directory.

    Args:
        input_dir: Directory to analyze

    Returns:
        Markdown-formatted statistics
    """
    if not input_dir:
        return "*No directory selected*"

    input_path = Path(input_dir)
    if not input_path.exists() or not input_path.is_dir():
        return "*Invalid directory*"

    json_files = list(input_path.glob("*.json"))

    if not json_files:
        return "*No JSON files found*"

    original_files = [f for f in json_files if '-transpose' not in f.stem]
    augmented_files = [f for f in json_files if '-transpose' in f.stem]

    total_size_mb = sum(f.stat().st_size for f in json_files) / (1024 * 1024)

    stats = f"""
### Directory Statistics

**Total JSON Files:** {len(json_files)}
- Original files: {len(original_files)}
- Augmented files: {len(augmented_files)}

**Total Size:** {total_size_mb:.1f} MB

**Potential Dataset Size:** {len(original_files) * 12} files (after full augmentation to 12 keys)
"""

    return stats


def create_augmentation_tab() -> gr.Tab:
    """
    Create the data augmentation tab with UI and event handlers.

    Returns:
        Gradio Tab component
    """
    with gr.Tab("Data Augmentation") as tab:

        gr.Markdown("""
        ## Transpose Tokenized Files for Data Augmentation

        This tool transposes tokenized JSON files to all 12 chromatic keys, expanding your dataset
        by 12x. Each original file generates 11 transposed versions (one for each key).

        **Note:** Files already containing '-transpose' in their name will be skipped to avoid
        re-transposing augmented data.
        """)

        with gr.Row():
            # Left column - Configuration and controls
            with gr.Column(scale=1):

                input_dir_aug = gr.Textbox(
                    label="Input Directory",
                    value="./processed",
                    placeholder="Path to folder containing tokenized JSON files"
                )

                output_dir_aug = gr.Textbox(
                    label="Output Directory",
                    value="./processed",
                    placeholder="Path where transposed files will be saved",
                    info="Can be the same as input directory"
                )

                mode_aug = gr.Radio(
                    choices=[
                        ("All Keys (-6 to +5 semitones)", "all"),
                        ("Higher Keys Only (+1 to +5)", "higher"),
                        ("Lower Keys Only (-6 to -1)", "lower")
                    ],
                    value="all",
                    label="Transposition Mode",
                    info="Choose which transpositions to apply"
                )

                overwrite_aug = gr.Checkbox(
                    label="Overwrite existing transposed files",
                    value=False,
                    info="If unchecked, existing files will be skipped"
                )

                with gr.Row():
                    transpose_btn = gr.Button(
                        "Transpose Files",
                        variant="primary",
                        size="lg"
                    )

                with gr.Row():
                    clean_btn = gr.Button(
                        "Clean Augmented Files",
                        variant="stop",
                        size="lg"
                    )

                gr.Markdown("""
                ---
                **Clean Augmented Files** will delete all files with '-transpose' in their name
                from the input directory. Use this to reset your dataset to original files only.
                """)

            # Right column - Results and status
            with gr.Column(scale=2):

                status_aug = gr.Textbox(
                    label="Status",
                    interactive=False
                )

                summary_aug = gr.Markdown("*No files processed yet*")

                with gr.Accordion("Directory Statistics", open=True):
                    stats_display = gr.Markdown(get_directory_stats("./processed"))

        # Event handlers
        transpose_btn.click(
            fn=transpose_batch_files,
            inputs=[input_dir_aug, output_dir_aug, mode_aug, overwrite_aug],
            outputs=[status_aug, summary_aug]
        ).then(
            fn=get_directory_stats,
            inputs=[input_dir_aug],
            outputs=[stats_display]
        )

        clean_btn.click(
            fn=clean_augmented_files,
            inputs=[input_dir_aug],
            outputs=[status_aug, summary_aug]
        ).then(
            fn=get_directory_stats,
            inputs=[input_dir_aug],
            outputs=[stats_display]
        )

        input_dir_aug.change(
            fn=get_directory_stats,
            inputs=[input_dir_aug],
            outputs=[stats_display]
        )

    return tab
