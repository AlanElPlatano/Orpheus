"""
Chord voicing normalization module.

Analyzes chord notes within each bar of a tokenized sequence to identify
chord root and quality, producing per-bar chord metadata. This metadata
enables Chord-Tone Embeddings to classify melody notes by their
harmonic relationship to the active chord.

The analysis works on already-reordered token sequences (using token_reorderer.py)
where chord notes sit between CHORD_START and MELODY_START markers.
"""

import logging
from typing import Dict, List, Optional, Tuple, FrozenSet

from ml_core.data.constants import (
    CHORD_START_TOKEN_NAME,
    MELODY_START_TOKEN_NAME,
    BAR_TOKEN_ID,
    CHORD_TEMPLATES,
    PITCH_CLASS_NAMES,
)

logger = logging.getLogger(__name__)


def _build_pitch_lookup(vocabulary: Dict[str, int]) -> Dict[int, int]:
    """
    Build a mapping from Pitch token IDs to MIDI pitch values.

    Args:
        vocabulary: Token name -> ID mapping

    Returns:
        Dict mapping token ID -> MIDI pitch number
    """
    lookup = {}
    for name, token_id in vocabulary.items():
        if name.startswith('Pitch_'):
            try:
                pitch = int(name.split('_')[1])
                lookup[token_id] = pitch
            except (IndexError, ValueError):
                continue
    return lookup


def identify_chord(
    pitch_classes: FrozenSet[int],
) -> Tuple[int, str]:
    """
    Identify chord root and quality from a set of pitch classes.

    Uses a two-pass strategy:
    1. Exact match: intervals must equal a template exactly.
    2. Subset match: all chord intervals appear in a template (handles
       voicings that omit the 5th, common in guitar-based genres).

    Among matches at the same priority level, prefers the most specific
    template (most intervals). Among equal specificity, prefers the
    template with fewer missing notes.

    Args:
        pitch_classes: Set of unique pitch classes (0-11) found in the chord

    Returns:
        Tuple of (root_pitch_class, quality_string).
        Returns (-1, "unknown") if no template matches.
    """
    if len(pitch_classes) < 2:
        return -1, "unknown"

    # Pass 1: exact match (highest confidence)
    best_root = -1
    best_quality = "unknown"
    best_template_size = 0

    for candidate_root in pitch_classes:
        intervals = frozenset((pc - candidate_root) % 12 for pc in pitch_classes)

        for quality, template in CHORD_TEMPLATES.items():
            if intervals == template and len(template) > best_template_size:
                best_root = candidate_root
                best_quality = quality
                best_template_size = len(template)

    if best_root >= 0:
        return best_root, best_quality

    # Pass 2: subset match (chord is missing notes, e.g. omitted 5th)
    # Score = matched intervals / template size (higher = better fit)
    best_score = 0.0
    best_missing = 999

    for candidate_root in pitch_classes:
        intervals = frozenset((pc - candidate_root) % 12 for pc in pitch_classes)

        for quality, template in CHORD_TEMPLATES.items():
            if not intervals.issubset(template):
                continue

            missing = len(template) - len(intervals)
            score = len(intervals) / len(template)

            is_better = (
                score > best_score
                or (score == best_score and missing < best_missing)
            )

            if is_better:
                best_root = candidate_root
                best_quality = quality
                best_score = score
                best_missing = missing

    return best_root, best_quality


def analyze_bar_chords(
    tokens: List[int],
    vocabulary: Dict[str, int],
) -> List[Dict]:
    """
    Analyze chord voicings across all bars in a token sequence.

    Scans the token sequence for CHORD_START / MELODY_START boundaries,
    extracts pitch tokens from the chord section, reduces them to pitch
    classes, and identifies each bar's chord root and quality.

    Args:
        tokens: Global token sequence
        vocabulary: Token name -> ID mapping

    Returns:
        List of per-bar chord info dicts with keys:
            - bar_index: 0-based bar number
            - chord_root: pitch class (0-11) or -1 if unknown
            - chord_root_name: human-readable root name (e.g. "A") or "unknown"
            - chord_quality: quality string (e.g. "major") or "unknown"
            - pitch_classes: sorted list of unique pitch classes in the chord
    """
    chord_start_id = vocabulary.get(CHORD_START_TOKEN_NAME)
    melody_start_id = vocabulary.get(MELODY_START_TOKEN_NAME)

    if chord_start_id is None or melody_start_id is None:
        logger.warning("Structural tokens not found in vocabulary, cannot analyze chords")
        return []

    bar_id = vocabulary.get("Bar_None", BAR_TOKEN_ID)
    pitch_lookup = _build_pitch_lookup(vocabulary)

    bar_chords = []
    bar_index = -1
    in_chord_section = False
    current_chord_pitches = []

    for token_id in tokens:
        if token_id == bar_id:
            # Flush previous bar's chord if any
            if bar_index >= 0 and current_chord_pitches:
                bar_chords.append(
                    _create_bar_chord_entry(bar_index, current_chord_pitches)
                )
            elif bar_index >= 0:
                bar_chords.append(_create_empty_bar_chord_entry(bar_index))

            bar_index += 1
            current_chord_pitches = []
            in_chord_section = False
            continue

        if token_id == chord_start_id:
            in_chord_section = True
            continue

        if token_id == melody_start_id:
            in_chord_section = False
            continue

        if in_chord_section and token_id in pitch_lookup:
            current_chord_pitches.append(pitch_lookup[token_id])

    # Flush the last bar
    if bar_index >= 0 and current_chord_pitches:
        bar_chords.append(
            _create_bar_chord_entry(bar_index, current_chord_pitches)
        )
    elif bar_index >= 0 and (not bar_chords or bar_chords[-1]['bar_index'] != bar_index):
        bar_chords.append(_create_empty_bar_chord_entry(bar_index))

    logger.info(
        f"Chord analysis complete: {len(bar_chords)} bars analyzed, "
        f"{sum(1 for bc in bar_chords if bc['chord_quality'] != 'unknown')} chords identified"
    )

    return bar_chords


def _create_bar_chord_entry(bar_index: int, midi_pitches: List[int]) -> Dict:
    """Build a bar chord info dict from raw MIDI pitches."""
    pitch_classes = frozenset(p % 12 for p in midi_pitches)
    root, quality = identify_chord(pitch_classes)
    root_name = PITCH_CLASS_NAMES[root] if root >= 0 else "unknown"

    return {
        'bar_index': bar_index,
        'chord_root': root,
        'chord_root_name': root_name,
        'chord_quality': quality,
        'pitch_classes': sorted(pitch_classes),
    }


def _create_empty_bar_chord_entry(bar_index: int) -> Dict:
    """Build a bar chord info dict for a bar with no chord notes."""
    return {
        'bar_index': bar_index,
        'chord_root': -1,
        'chord_root_name': "unknown",
        'chord_quality': "unknown",
        'pitch_classes': [],
    }


def enrich_json_with_chords(json_data: Dict) -> Dict:
    """
    Add bar_chords metadata to an existing tokenized JSON structure.

    This is the main entry point for enriching both new and existing files.
    It reads global_tokens and vocabulary from the JSON, runs chord analysis,
    and stores the result under the 'bar_chords' key.

    Args:
        json_data: Parsed JSON dict (must contain 'global_tokens' and 'vocabulary')

    Returns:
        The same dict with 'bar_chords' added (modified in place and returned)
    """
    tokens = json_data.get('global_tokens', [])
    vocabulary = json_data.get('vocabulary', {})

    if not tokens or not vocabulary:
        logger.warning("Cannot enrich: missing global_tokens or vocabulary")
        return json_data

    bar_chords = analyze_bar_chords(tokens, vocabulary)
    json_data['bar_chords'] = bar_chords

    return json_data


__all__ = [
    'identify_chord',
    'analyze_bar_chords',
    'enrich_json_with_chords',
]
