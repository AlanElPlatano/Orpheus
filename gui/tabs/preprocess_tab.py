"""
Preprocess tab for Orpheus Gradio GUI.

This tab handles MIDI preprocessing operations:
- Track filtering and cleanup
- Quantization
- Note cleanup
- Structure simplification
- Backup and recovery
"""

import gradio as gr
from pathlib import Path
from typing import Tuple, List
import shutil
import time

from preprocessor import process_midi_file


# ============================================================================
# Constants
# ============================================================================

BACKUP_DIR = Path("./midi_backups")
DEFAULT_INPUT_DIR = "./source_midis"

QUANTIZATION_OPTIONS = [
    ("Off", None),
    ("1/8 notes", 8),
    ("1/12 notes", 12),
    ("1/16 notes", 16),
    ("1/24 notes", 24),
    ("1/32 notes (recommended)", 32),
    ("1/48 notes", 48),
    ("1/64 notes", 64)
]


# ============================================================================
# Backup Functions
# ============================================================================

def backup_midi_files(input_dir: str) -> Tuple[str, str]:
    """
    Create a backup of all MIDI files in the input directory.

    Args:
        input_dir: Directory containing MIDI files to backup

    Returns:
        Tuple of (status, summary)
    """
    if not input_dir:
        return "❌ No input directory selected", ""

    input_path = Path(input_dir)
    if not input_path.exists() or not input_path.is_dir():
        return "❌ Invalid input directory", ""

    # Find all MIDI files
    midi_files = list(input_path.glob("*.mid")) + list(input_path.glob("*.midi"))
    if not midi_files:
        return "❌ No MIDI files found to backup", ""

    # Create backup directory with timestamp
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    backup_path = BACKUP_DIR / timestamp
    backup_path.mkdir(parents=True, exist_ok=True)

    # Copy all MIDI files
    success_count = 0
    failed_files = []

    for midi_file in midi_files:
        try:
            shutil.copy2(midi_file, backup_path / midi_file.name)
            success_count += 1
        except Exception as e:
            failed_files.append(f"{midi_file.name}: {str(e)}")

    # Build summary
    summary = f"""
## 💾 Backup Complete

**Backup Location:** `{backup_path}`
**Total Files:** {len(midi_files)}
**Successfully Backed Up:** {success_count} ✅
**Failed:** {len(failed_files)} ❌

### Files Backed Up:
"""

    for midi_file in midi_files:
        if midi_file.name not in [f.split(':')[0] for f in failed_files]:
            summary += f"\n- ✅ {midi_file.name}"

    if failed_files:
        summary += "\n\n### Failed Files:\n"
        for failed in failed_files:
            summary += f"\n- ❌ {failed}"

    status = f"✅ Backup complete: {success_count}/{len(midi_files)} files backed up to {backup_path.name}"

    return status, summary


def recover_backup_files(backup_selection: str, input_dir: str) -> Tuple[str, str]:
    """
    Recover MIDI files from a backup folder.

    Args:
        backup_selection: Selected backup folder name
        input_dir: Target directory to restore files to

    Returns:
        Tuple of (status, summary)
    """
    if not backup_selection or backup_selection == "No backups available":
        return "❌ No backup selected", ""

    if not input_dir:
        return "❌ No input directory selected", ""

    backup_path = BACKUP_DIR / backup_selection
    if not backup_path.exists():
        return "❌ Backup folder not found", ""

    input_path = Path(input_dir)
    input_path.mkdir(parents=True, exist_ok=True)

    # Find all MIDI files in backup
    midi_files = list(backup_path.glob("*.mid")) + list(backup_path.glob("*.midi"))
    if not midi_files:
        return "❌ No MIDI files found in backup", ""

    # Copy files back
    success_count = 0
    failed_files = []

    for midi_file in midi_files:
        try:
            shutil.copy2(midi_file, input_path / midi_file.name)
            success_count += 1
        except Exception as e:
            failed_files.append(f"{midi_file.name}: {str(e)}")

    # Build summary
    summary = f"""
## Recovery Complete

**Recovered From:** `{backup_path}`
**Restored To:** `{input_path}`
**Total Files:** {len(midi_files)}
**Successfully Recovered:** {success_count} ✅
**Failed:** {len(failed_files)} ❌

### Files Recovered:
"""

    for midi_file in midi_files:
        if midi_file.name not in [f.split(':')[0] for f in failed_files]:
            summary += f"\n- ✅ {midi_file.name}"

    if failed_files:
        summary += "\n\n### Failed Files:\n"
        for failed in failed_files:
            summary += f"\n- ❌ {failed}"

    status = f"✅ Recovery complete: {success_count}/{len(midi_files)} files restored"

    return status, summary


def get_available_backups() -> List[str]:
    """
    Get list of available backup folders.

    Returns:
        List of backup folder names (sorted newest first)
    """
    if not BACKUP_DIR.exists():
        return ["No backups available yet"]

    backups = [d.name for d in BACKUP_DIR.iterdir() if d.is_dir()]

    if not backups:
        return ["No backups available yet"]

    # Sort by timestamp (newest first)
    backups.sort(reverse=True)

    return backups


# ============================================================================
# Preprocessing Functions
# ============================================================================

def preprocess_batch_files(
    input_dir: str,
    quantize_grid: int,
    cleanup_notes: bool,
    remove_empty: bool,
    remove_bass: bool,
    bass_threshold: int,
    progress=gr.Progress()
) -> Tuple[str, str]:
    """
    Process multiple MIDI files with preprocessing options.

    Args:
        input_dir: Directory containing MIDI files
        quantize_grid: Quantization grid (None to skip)
        cleanup_notes: Whether to clean up notes
        remove_empty: Whether to remove empty tracks
        remove_bass: Whether to remove bass tracks
        bass_threshold: MIDI note threshold for bass removal
        progress: Gradio progress tracker

    Returns:
        Tuple of (status, summary)
    """
    if not input_dir:
        return "❌ No input directory selected", ""

    input_path = Path(input_dir)
    if not input_path.exists() or not input_path.is_dir():
        return "❌ Invalid input directory", ""

    # Find all MIDI files
    midi_files = list(input_path.glob("*.mid")) + list(input_path.glob("*.midi"))
    if not midi_files:
        return "❌ No MIDI files found in directory", ""

    # Process all files
    results = []
    total_time = 0

    for i, midi_file in enumerate(midi_files, 1):
        progress(
            i / len(midi_files),
            desc=f"Processing {i}/{len(midi_files)}: {midi_file.name}"
        )

        start_time = time.time()

        success, processed_midi, stats = process_midi_file(
            str(midi_file),
            quantize_grid=quantize_grid,
            cleanup_notes=cleanup_notes,
            remove_empty=remove_empty,
            remove_bass=remove_bass,
            bass_threshold=bass_threshold,
            verbose=False
        )

        processing_time = time.time() - start_time
        total_time += processing_time

        if success:
            # Overwrite the original file
            try:
                processed_midi.write(str(midi_file))
                results.append({
                    "file": midi_file.name,
                    "success": True,
                    "stats": stats,
                    "time": processing_time
                })
            except Exception as e:
                results.append({
                    "file": midi_file.name,
                    "success": False,
                    "error": f"Failed to save: {str(e)}",
                    "time": processing_time
                })
        else:
            results.append({
                "file": midi_file.name,
                "success": False,
                "error": stats.get('error', 'Unknown error'),
                "time": processing_time
            })

    # Calculate statistics
    successful = sum(1 for r in results if r["success"])
    failed = len(results) - successful

    # Build summary
    summary = f"""
## 🔧 Preprocessing Complete

**Total Files:** {len(results)}
**Successful:** {successful} ✅
**Failed:** {failed} ❌
**Total Time:** {total_time:.1f}s
**Average Time:** {total_time/len(results):.1f}s per file

### Settings Applied:
- **Quantization:** {f"1/{quantize_grid} notes" if quantize_grid else "Off"}
- **Note Cleanup:** {"Enabled" if cleanup_notes else "Disabled"}
- **Remove Empty Tracks:** {"Enabled" if remove_empty else "Disabled"}
- **Remove Bass Tracks:** {"Enabled" if remove_bass else "Disabled"}
{f"- **Bass Threshold:** MIDI note {bass_threshold}" if remove_bass else ""}

### Results:
"""

    for i, result in enumerate(results, 1):
        status_icon = "✅" if result["success"] else "❌"
        summary += f"\n{i}. {status_icon} **{result['file']}**"

        if result["success"]:
            stats = result['stats']
            preprocessing = stats.get('preprocessing_applied', {})
            notes_removed = preprocessing.get('notes_removed', 0)
            tracks_removed = preprocessing.get('tracks_removed', 0)

            details = []
            if notes_removed > 0:
                details.append(f"{notes_removed} notes removed")
            if tracks_removed > 0:
                details.append(f"{tracks_removed} tracks removed")

            if details:
                summary += f" → {', '.join(details)}"
            summary += f" ({result['time']:.1f}s)"
        else:
            summary += f" → {result['error']}"

    status = f"✅ Preprocessing complete: {successful}/{len(results)} successful"

    return status, summary


def show_warning_message() -> str:
    """
    Show warning message before processing.

    Returns:
        Warning message
    """
    return """
⚠️ **WARNING: This will overwrite all MIDI files in the input directory!**

All files will be permanently modified with the selected preprocessing options.

**We strongly recommend backing up your files before proceeding.**

Click the "Backup MIDI Files" button above before processing.

If you're sure you want to continue, click "Process Files" again.
"""


# ============================================================================
# Tab Creation
# ============================================================================

def create_preprocess_tab() -> gr.Tab:
    """
    Create the preprocess tab with UI and event handlers.

    Returns:
        Gradio Tab component
    """
    with gr.Tab("Preprocess") as tab:

        gr.Markdown("""
        ## MIDI Preprocessing

        Clean and prepare MIDI files for AI training or further processing.

        ⚠️ **Important:** We strongly recommend backing up your MIDI files before preprocessing!
        """)

        # Top section - Main controls in three columns
        with gr.Row():
            # Column 1 - Backup & Recovery and Input
            with gr.Column(scale=1):
                gr.Markdown("### Backup & Recovery")

                backup_btn = gr.Button(
                    "Backup MIDI Files",
                    variant="secondary",
                    size="lg"
                )

                backup_dropdown = gr.Dropdown(
                    choices=get_available_backups(),
                    label="Select Backup to Recover",
                    info="Choose a backup to restore",
                    interactive=True
                )

                refresh_backups_btn = gr.Button(
                    "Refresh Backup List",
                    size="sm"
                )

                recover_btn = gr.Button(
                    "Recover Backup MIDIs",
                    variant="secondary",
                    size="lg"
                )

                input_dir_preprocess = gr.Textbox(
                    label="Input Directory",
                    value=DEFAULT_INPUT_DIR,
                    placeholder="Path to folder containing MIDI files"
                )

            # Column 2 - Preprocessing Options
            with gr.Column(scale=1):
                gr.Markdown("### Preprocessing Options")

                quantize_dropdown = gr.Dropdown(
                    choices=QUANTIZATION_OPTIONS,
                    value=32,
                    label="Quantization Grid",
                    info="Align notes to rhythmic grid"
                )

                cleanup_checkbox = gr.Checkbox(
                    label="Note Cleanup",
                    value=True,
                    info="Remove short/empty notes and trim overlaps"
                )

                remove_empty_checkbox = gr.Checkbox(
                    label="Remove Empty Tracks",
                    value=True,
                    info="Remove tracks with no/minimal content"
                )

                remove_bass_checkbox = gr.Checkbox(
                    label="Remove Bass Tracks",
                    value=False,
                    info="Remove tracks below threshold"
                )

                bass_threshold_slider = gr.Slider(
                    minimum=24,
                    maximum=48,
                    value=32,
                    step=1,
                    label="Bass Threshold (MIDI Note)",
                    info="C1=24, C2=36, C3=48"
                )

            # Column 3 - Execute and Status
            with gr.Column(scale=1):
                gr.Markdown("### Execute")

                process_preprocess_btn = gr.Button(
                    "Process Files",
                    variant="primary",
                    size="lg"
                )

                status_preprocess = gr.Textbox(
                    label="Status",
                    interactive=False,
                    lines=3
                )

        # Warning state to track if user has been warned
        warning_shown = gr.State(value=False)

        # Bottom section - Results (full width, not scrollable initially)
        gr.Markdown("---")

        with gr.Row():
            with gr.Column():
                gr.Markdown("### Processing Results")
                summary_preprocess = gr.Markdown("*No files processed yet*")

        # Event handlers

        def handle_backup(input_dir):
            """Handle backup button click."""
            status, summary = backup_midi_files(input_dir)
            # Refresh backup list after creating backup
            new_backups = get_available_backups()
            return status, summary, gr.update(choices=new_backups)

        backup_btn.click(
            fn=handle_backup,
            inputs=[input_dir_preprocess],
            outputs=[status_preprocess, summary_preprocess, backup_dropdown]
        )

        def handle_recover(backup_selection, input_dir):
            """Handle recover button click."""
            return recover_backup_files(backup_selection, input_dir)

        recover_btn.click(
            fn=handle_recover,
            inputs=[backup_dropdown, input_dir_preprocess],
            outputs=[status_preprocess, summary_preprocess]
        )

        refresh_backups_btn.click(
            fn=lambda: gr.update(choices=get_available_backups()),
            outputs=backup_dropdown
        )

        def handle_process_click(
            warned,
            input_dir,
            quantize_grid,
            cleanup_notes,
            remove_empty,
            remove_bass,
            bass_threshold,
            progress=gr.Progress()
        ):
            """Handle process button click with warning."""
            if not warned:
                # Show warning and set flag
                return (
                    show_warning_message(),
                    "*Click 'Process Files' again to proceed*",
                    True  # Set warning_shown to True
                )
            else:
                # Actually process the files
                status, summary = preprocess_batch_files(
                    input_dir,
                    quantize_grid,
                    cleanup_notes,
                    remove_empty,
                    remove_bass,
                    bass_threshold,
                    progress
                )
                # Reset warning flag
                return status, summary, False

        process_preprocess_btn.click(
            fn=handle_process_click,
            inputs=[
                warning_shown,
                input_dir_preprocess,
                quantize_dropdown,
                cleanup_checkbox,
                remove_empty_checkbox,
                remove_bass_checkbox,
                bass_threshold_slider
            ],
            outputs=[status_preprocess, summary_preprocess, warning_shown]
        )

    return tab
