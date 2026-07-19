"""
Song-form assembly for structured generation.

Splits a generated token sequence into bars and reassembles it following a
song form such as "AAAB", giving the output the literal section repetition
that makes the dataset's songs feel intentional: a core idea stated several
times, then a contrasting section.

Bars are self-contained splice units in this REMI format: melody positions
restart at every Bar token and chord notes carry no positions at all, so
whole bars can be reordered without rewriting any tokens. Sections are cut
from one continuous generation (A = first bars, B = the bars that follow),
so B always continues A's harmonic context - matching how the sections are
later placed in the assembled song.
"""

from typing import Dict, List, Tuple
import logging

from ..data.constants import BAR_TOKEN_ID, BOS_TOKEN_ID, EOS_TOKEN_ID

logger = logging.getLogger(__name__)

SONG_FORM_FREE = ""
DEFAULT_SONG_FORM = "AAAB"
DEFAULT_SECTION_BARS = 4
SECTION_LABELS = "AB"


def count_distinct_sections(form: str) -> int:
    """Number of distinct sections a form needs (e.g. 'AAAB' -> 2)."""
    return len(set(form))


def bars_needed_for_form(form: str, section_bars: int) -> int:
    """Bars of generated material required to fill every distinct section."""
    return count_distinct_sections(form) * section_bars


def validate_song_form(form: str) -> None:
    """Raise ValueError unless the form is a non-empty string of A/B labels."""
    if not form or any(label not in SECTION_LABELS for label in form):
        raise ValueError(
            f"Invalid song form '{form}': expected a non-empty combination "
            f"of the labels '{SECTION_LABELS}' (e.g. 'AAAB')"
        )


def split_preamble_and_bars(token_ids: List[int]) -> Tuple[List[int], List[List[int]]]:
    """
    Split a sequence into its pre-bar preamble and one token list per bar.

    BOS and EOS are dropped; the caller re-adds them around the assembled
    song. Each bar starts with its Bar token and runs to the next one.
    """
    preamble: List[int] = []
    bars: List[List[int]] = []

    for token in token_ids:
        if token == BOS_TOKEN_ID or token == EOS_TOKEN_ID:
            continue
        if token == BAR_TOKEN_ID:
            bars.append([token])
        elif bars:
            bars[-1].append(token)
        else:
            preamble.append(token)

    return preamble, bars


def build_structured_sequence(
    token_ids: List[int],
    form: str,
    section_bars: int
) -> List[int]:
    """
    Reassemble a generated sequence into the given song form.

    The first section_bars bars become section A, the next section_bars
    bars become section B, and so on for each distinct label in the form.
    Returns the sequence unchanged (with a warning) when it holds too few
    bars to fill every section, so callers degrade to unstructured output
    instead of failing.
    """
    validate_song_form(form)

    preamble, bars = split_preamble_and_bars(token_ids)

    bars_needed = bars_needed_for_form(form, section_bars)
    if len(bars) < bars_needed:
        logger.warning(
            f"Cannot assemble form '{form}': need {bars_needed} bars "
            f"but only {len(bars)} were generated; keeping free-form output"
        )
        return token_ids

    sections = _cut_sections(bars, form, section_bars)

    assembled = [BOS_TOKEN_ID] + preamble
    for label in form:
        for bar in sections[label]:
            assembled.extend(bar)
    assembled.append(EOS_TOKEN_ID)

    logger.info(
        f"Assembled song form '{form}': {section_bars}-bar sections, "
        f"{len(assembled)} tokens total"
    )

    return assembled


def _cut_sections(
    bars: List[List[int]],
    form: str,
    section_bars: int
) -> Dict[str, List[List[int]]]:
    """Assign consecutive bar runs to each distinct label, in label order."""
    sections = {}
    for position, label in enumerate(sorted(set(form))):
        start = position * section_bars
        sections[label] = bars[start:start + section_bars]
    return sections


__all__ = [
    'SONG_FORM_FREE',
    'DEFAULT_SONG_FORM',
    'DEFAULT_SECTION_BARS',
    'count_distinct_sections',
    'bars_needed_for_form',
    'validate_song_form',
    'split_preamble_and_bars',
    'build_structured_sequence'
]
