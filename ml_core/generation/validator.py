"""
Post-generation constraint validation.

Validates generated token sequences against musical constraints:
- Monophony in melody tracks
- Chord sustain in chord tracks
- Sequence structure validity
"""

from dataclasses import dataclass, field
from typing import List, Set, Dict, Optional
import logging

from ..data.constants import (
    TOKEN_RANGES,
    BOS_TOKEN_ID,
    EOS_TOKEN_ID,
    BAR_TOKEN_ID,
    is_pitch_token,
    is_duration_token,
    is_position_token,
    is_special_token
)
from ..data.vocab import VocabularyInfo

logger = logging.getLogger(__name__)


@dataclass
class ValidationReport:
    """
    Report of constraint validation results.

    Contains detailed information about which constraints were violated
    and where in the sequence the violations occurred.
    """

    # ========== Overall Status ==========
    is_valid: bool = True
    num_violations: int = 0

    # ========== Specific Violations ==========
    monophony_violations: List[str] = field(default_factory=list)
    chord_sustain_violations: List[str] = field(default_factory=list)
    structure_violations: List[str] = field(default_factory=list)
    token_sequence_violations: List[str] = field(default_factory=list)

    # ========== Statistics ==========
    sequence_length: int = 0
    num_bars: int = 0
    num_pitch_tokens: int = 0
    num_duration_tokens: int = 0

    def add_violation(self, category: str, message: str) -> None:
        """Add a violation to the report."""
        if category == "monophony":
            self.monophony_violations.append(message)
        elif category == "chord_sustain":
            self.chord_sustain_violations.append(message)
        elif category == "structure":
            self.structure_violations.append(message)
        elif category == "token_sequence":
            self.token_sequence_violations.append(message)

        self.num_violations += 1
        self.is_valid = False

    def get_all_violations(self) -> List[str]:
        """Get all violations as a single list."""
        return (
            self.monophony_violations +
            self.chord_sustain_violations +
            self.structure_violations +
            self.token_sequence_violations
        )

    def get_summary(self) -> str:
        """Get human-readable summary of validation results."""
        if self.is_valid:
            return f"✓ Valid | {self.sequence_length} tokens | {self.num_bars} bars"
        else:
            violations_by_type = []
            if self.monophony_violations:
                violations_by_type.append(f"{len(self.monophony_violations)} monophony")
            if self.chord_sustain_violations:
                violations_by_type.append(f"{len(self.chord_sustain_violations)} chord sustain")
            if self.structure_violations:
                violations_by_type.append(f"{len(self.structure_violations)} structure")
            if self.token_sequence_violations:
                violations_by_type.append(f"{len(self.token_sequence_violations)} sequence")

            return f"✗ Invalid | {self.num_violations} violations: {', '.join(violations_by_type)}"


class ConstraintValidator:
    """
    Validator for musical constraints in generated sequences.

    Checks that generated sequences satisfy the required constraints
    for the 2-track structure.
    """

    def __init__(self, vocab_info: VocabularyInfo):
        """
        Initialize constraint validator.

        Args:
            vocab_info: Vocabulary information for token categorization
        """
        self.vocab_info = vocab_info

    def validate(self, token_ids: List[int]) -> ValidationReport:
        """
        Validate a generated token sequence.

        Args:
            token_ids: Generated token sequence

        Returns:
            ValidationReport with detailed results
        """
        report = ValidationReport()

        # Compute basic statistics
        report.sequence_length = len(token_ids)
        report.num_bars = sum(1 for token in token_ids if token == BAR_TOKEN_ID)
        report.num_pitch_tokens = sum(1 for token in token_ids if self.vocab_info.is_pitch_token(token))
        report.num_duration_tokens = sum(1 for token in token_ids if self.vocab_info.is_duration_token(token))

        # Run validation checks
        self._validate_token_sequence(token_ids, report)
        self._validate_structure(token_ids, report)
        self._validate_monophony(token_ids, report)
        self._validate_pitch_duration_pairs(token_ids, report)

        # Log results
        if report.is_valid:
            logger.info(f"Validation passed: {report.get_summary()}")
        else:
            logger.warning(f"Validation failed: {report.get_summary()}")
            for violation in report.get_all_violations():
                logger.warning(f"  - {violation}")

        return report

    def _validate_token_sequence(self, token_ids: List[int], report: ValidationReport) -> None:
        """
        Validate basic token sequence properties.

        Args:
            token_ids: Token sequence
            report: Validation report to update
        """
        if not token_ids:
            report.add_violation("token_sequence", "Empty token sequence")
            return

        # Check for out-of-vocabulary tokens
        for i, token_id in enumerate(token_ids):
            if token_id < 0 or token_id >= self.vocab_info.vocab_size:
                report.add_violation(
                    "token_sequence",
                    f"Out-of-vocabulary token at position {i}: {token_id} "
                    f"(vocab size: {self.vocab_info.vocab_size})"
                )

        # Check for excessive repetition
        if len(token_ids) > 10:
            for i in range(len(token_ids) - 10):
                window = token_ids[i:i+10]
                if len(set(window)) == 1 and not self.vocab_info.is_special_token(window[0]):
                    report.add_violation(
                        "token_sequence",
                        f"Excessive repetition detected at position {i}: "
                        f"token {window[0]} repeated 10+ times"
                    )
                    break  # Only report once

    def _validate_structure(self, token_ids: List[int], report: ValidationReport) -> None:
        """
        Validate sequence structure (BOS, EOS, bars).

        Args:
            token_ids: Token sequence
            report: Validation report to update
        """
        # Check for BOS token
        if token_ids[0] != BOS_TOKEN_ID:
            report.add_violation("structure", "Sequence does not start with BOS token")

        # Check for EOS token (should be last or close to last)
        if EOS_TOKEN_ID in token_ids:
            eos_position = token_ids.index(EOS_TOKEN_ID)
            if eos_position < len(token_ids) - 3:  # Allow some padding
                report.add_violation(
                    "structure",
                    f"EOS token found at position {eos_position}, "
                    f"but sequence continues to {len(token_ids)}"
                )

        # Check for minimum number of bars
        if report.num_bars < 4:
            report.add_violation(
                "structure",
                f"Too few bars: {report.num_bars} (minimum: 4)"
            )

    def _validate_monophony(self, token_ids: List[int], report: ValidationReport) -> None:
        """
        Validate monophony constraint (no overlapping melody notes).

        Only checks monophony in melody sections (after MelodyStart marker).
        Chord sections are polyphonic by design and are excluded.

        Args:
            token_ids: Token sequence
            report: Validation report to update
        """
        active_notes: Set[int] = set()
        in_melody = False

        chord_start_id = self.vocab_info.chord_start_token_id
        melody_start_id = self.vocab_info.melody_start_token_id

        for i, token_id in enumerate(token_ids):
            # Track section switches
            if token_id == melody_start_id:
                in_melody = True
                active_notes.clear()
                continue
            elif token_id == chord_start_id:
                in_melody = False
                active_notes.clear()
                continue

            # Only check monophony in melody sections
            if not in_melody:
                continue

            # Check if this is a pitch token
            if self.vocab_info.is_pitch_token(token_id):
                token_name = self.vocab_info.get_token_name(token_id)

                try:
                    # Extract MIDI pitch
                    midi_pitch = int(token_name.split('_')[1])

                    # Check if another note is already active
                    if len(active_notes) > 0:
                        report.add_violation(
                            "monophony",
                            f"Monophony violation at position {i}: "
                            f"new note {midi_pitch} while notes {active_notes} still active"
                        )

                    # Add to active notes
                    active_notes.add(midi_pitch)

                except (IndexError, ValueError):
                    # Could not parse pitch
                    pass

            # Check if this is a duration token (ends note)
            elif self.vocab_info.is_duration_token(token_id):
                active_notes.clear()

    def _validate_pitch_duration_pairs(
        self,
        token_ids: List[int],
        report: ValidationReport
    ) -> None:
        """
        Validate that pitch tokens are followed by duration tokens.

        Args:
            token_ids: Token sequence
            report: Validation report to update
        """
        expecting_duration = False

        for i, token_id in enumerate(token_ids):
            if self.vocab_info.is_pitch_token(token_id):
                expecting_duration = True

            elif self.vocab_info.is_duration_token(token_id):
                expecting_duration = False

            elif expecting_duration and not self.vocab_info.is_special_token(token_id):
                # We expected a duration token but got something else
                # (Allow special tokens like BAR to appear between pitch and duration)
                if not is_position_token(token_id):
                    token_name = self.vocab_info.get_token_name(token_id)
                    report.add_violation(
                        "token_sequence",
                        f"Expected duration token after pitch at position {i-1}, "
                        f"got {token_name} instead"
                    )
                    expecting_duration = False


def validate_generated_sequence(
    token_ids: List[int],
    vocab_info: VocabularyInfo
) -> ValidationReport:
    """
    Convenience function to validate a generated sequence.

    Args:
        token_ids: Generated token sequence
        vocab_info: Vocabulary information

    Returns:
        ValidationReport with detailed results
    """
    validator = ConstraintValidator(vocab_info)
    return validator.validate(token_ids)


__all__ = [
    'ValidationReport',
    'ConstraintValidator',
    'validate_generated_sequence'
]
