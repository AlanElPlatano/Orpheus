"""
Batch chord enrichment for existing processed JSON files.

Adds bar_chords metadata to already-processed tokenized JSON files
without requiring re-tokenization. Can be run as a standalone script or
called from the GUI.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Callable, Optional

from midi_parser.core.chord_analyzer import enrich_json_with_chords

logger = logging.getLogger(__name__)


def enrich_single_file(file_path: Path, overwrite: bool = True) -> bool:
    """
    Enrich a single JSON file with bar_chords metadata.

    Reads the file, runs chord analysis, and writes the result back.
    Skips files that already have bar_chords unless overwrite is True.

    Args:
        file_path: Path to tokenized JSON file
        overwrite: Re-analyze even if bar_chords already exists

    Returns:
        True if file was enriched, False if skipped or failed
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        logger.error(f"Failed to read {file_path.name}: {e}")
        return False

    if 'bar_chords' in data and not overwrite:
        logger.debug(f"Skipping {file_path.name}: already enriched")
        return False

    if not data.get('global_tokens') or not data.get('vocabulary'):
        logger.warning(f"Skipping {file_path.name}: missing tokens or vocabulary")
        return False

    enrich_json_with_chords(data)

    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        return True
    except Exception as e:
        logger.error(f"Failed to write {file_path.name}: {e}")
        return False


def batch_enrich_directory(
    directory: Path,
    overwrite: bool = False,
    pattern: str = "*.json",
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> Dict:
    """
    Enrich all JSON files in a directory with bar_chords metadata.

    Args:
        directory: Directory containing tokenized JSON files
        overwrite: Re-analyze files that already have bar_chords
        pattern: Glob pattern for file matching
        progress_callback: Optional callback(progress_fraction, description)

    Returns:
        Stats dict with keys: total, enriched, skipped, failed
    """
    directory = Path(directory)
    json_files = sorted(directory.glob(pattern))

    stats = {
        'total': len(json_files),
        'enriched': 0,
        'skipped': 0,
        'failed': 0,
    }

    if not json_files:
        logger.warning(f"No files matching '{pattern}' in {directory}")
        return stats

    logger.info(f"Enriching {len(json_files)} files in {directory}")

    for i, file_path in enumerate(json_files):
        if progress_callback:
            progress_callback(
                (i + 1) / len(json_files),
                f"Analyzing chords: {file_path.name}"
            )

        result = enrich_single_file(file_path, overwrite=overwrite)

        if result:
            stats['enriched'] += 1
        elif not result and 'bar_chords' in _peek_json_keys(file_path):
            stats['skipped'] += 1
        else:
            stats['failed'] += 1

    logger.info(
        f"Enrichment complete: {stats['enriched']} enriched, "
        f"{stats['skipped']} skipped, {stats['failed']} failed "
        f"(out of {stats['total']} files)"
    )

    return stats


def _peek_json_keys(file_path: Path) -> set:
    """Quickly check top-level keys without fully parsing the JSON."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return set(data.keys())
    except Exception:
        return set()


__all__ = [
    'enrich_single_file',
    'batch_enrich_directory',
]
