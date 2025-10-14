"""
Vocabulary management for REMI tokenization.

This module loads and manages the vocabulary extracted from tokenized JSON files.
The vocabulary is frozen and consistent across all training data.

Serves as a dictionary between tokens and numbers, the model can't understand "Pitch_60" it
only understands the number 29. This module handles all those "translations". Also verifies
that all the training dataset has the EXACT same vocabular, which is critical for training.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Set
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class VocabularyInfo:
    """
    Information about the vocabulary structure.
    """
    vocab_size: int
    token_to_id: Dict[str, int]
    id_to_token: Dict[int, str]

    # Special token IDs
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2
    mask_token_id: int = 3
    bar_token_id: int = 4

    # Token categories for analysis
    pitch_tokens: Set[int] = None
    duration_tokens: Set[int] = None
    position_tokens: Set[int] = None
    velocity_tokens: Set[int] = None
    tempo_tokens: Set[int] = None
    time_sig_tokens: Set[int] = None
    program_tokens: Set[int] = None

    def __post_init__(self):
        """Initialize token category sets after dataclass creation."""
        if self.pitch_tokens is None:
            self._categorize_tokens()

    def _categorize_tokens(self):
        """
        Categorize tokens by type for constraint enforcement.
        """
        self.pitch_tokens = set()
        self.duration_tokens = set()
        self.position_tokens = set()
        self.velocity_tokens = set()
        self.tempo_tokens = set()
        self.time_sig_tokens = set()
        self.program_tokens = set()

        for token_str, token_id in self.token_to_id.items():
            token_lower = token_str.lower()

            if token_str.startswith('Pitch_'):
                self.pitch_tokens.add(token_id)
            elif token_str.startswith('Duration_'):
                self.duration_tokens.add(token_id)
            elif token_str.startswith('Position_'):
                self.position_tokens.add(token_id)
            elif token_str.startswith('Velocity_'):
                self.velocity_tokens.add(token_id)
            elif token_str.startswith('Tempo_'):
                self.tempo_tokens.add(token_id)
            elif token_str.startswith('TimeSig_'):
                self.time_sig_tokens.add(token_id)
            elif token_str.startswith('Program_'):
                self.program_tokens.add(token_id)

        logger.info(f"Categorized tokens:")
        logger.info(f"  Pitch: {len(self.pitch_tokens)}")
        logger.info(f"  Duration: {len(self.duration_tokens)}")
        logger.info(f"  Position: {len(self.position_tokens)}")
        logger.info(f"  Velocity: {len(self.velocity_tokens)}")
        logger.info(f"  Tempo: {len(self.tempo_tokens)}")
        logger.info(f"  Time Sig: {len(self.time_sig_tokens)}")
        logger.info(f"  Program: {len(self.program_tokens)}")

    def get_token_name(self, token_id: int) -> str:
        """
        Get the string name for a token ID.

        Args:
            token_id: Integer token ID

        Returns:
            Token name string, or '<UNK>' if not found
        """
        return self.id_to_token.get(token_id, '<UNK>')

    def get_token_id(self, token_name: str) -> Optional[int]:
        """
        Get the ID for a token name.

        Args:
            token_name: Token name string

        Returns:
            Token ID, or None if not found
        """
        return self.token_to_id.get(token_name)

    def is_pitch_token(self, token_id: int) -> bool:
        """Check if token is a pitch token."""
        return token_id in self.pitch_tokens

    def is_duration_token(self, token_id: int) -> bool:
        """Check if token is a duration token."""
        return token_id in self.duration_tokens

    def is_position_token(self, token_id: int) -> bool:
        """Check if token is a position token."""
        return token_id in self.position_tokens

    def is_special_token(self, token_id: int) -> bool:
        """Check if token is a special token (PAD, BOS, EOS, MASK, Bar)."""
        return token_id in {
            self.pad_token_id,
            self.bos_token_id,
            self.eos_token_id,
            self.mask_token_id,
            self.bar_token_id
        }


class VocabularyLoader:
    """
    Loads and validates vocabulary from tokenized JSON files.
    """

    def __init__(self, json_dir: Path):
        """
        Initialize vocabulary loader.

        Args:
            json_dir: Directory containing tokenized JSON files
        """
        self.json_dir = Path(json_dir)
        if not self.json_dir.exists():
            raise FileNotFoundError(f"JSON directory not found: {json_dir}")

    def load_vocabulary_from_json(self, json_path: Optional[Path] = None) -> VocabularyInfo:
        """
        Load vocabulary from a tokenized JSON file.

        Args:
            json_path: Specific JSON file to load from. If None, finds first available.

        Returns:
            VocabularyInfo object with complete vocabulary

        Raises:
            FileNotFoundError: If no JSON files found
            ValueError: If vocabulary is invalid
        """
        if json_path is None:
            # Find first JSON file in directory
            json_files = list(self.json_dir.glob('*.json'))
            if not json_files:
                raise FileNotFoundError(f"No JSON files found in {self.json_dir}")
            json_path = json_files[0]
            logger.info(f"Using vocabulary from: {json_path.name}")

        # Load JSON
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Extract vocabulary
        if 'vocabulary' not in data:
            raise ValueError(f"No vocabulary found in {json_path}")

        vocab_dict = data['vocabulary']
        vocab_size = data.get('vocabulary_size', len(vocab_dict))

        # Validate vocabulary size
        if vocab_size != len(vocab_dict):
            logger.warning(
                f"Vocabulary size mismatch: declared={vocab_size}, actual={len(vocab_dict)}"
            )
            vocab_size = len(vocab_dict)

        # Create bidirectional mappings
        token_to_id = vocab_dict
        id_to_token = {v: k for k, v in vocab_dict.items()}

        # Verify bidirectional mapping is valid
        if len(id_to_token) != len(token_to_id):
            raise ValueError("Vocabulary has duplicate token IDs")

        # Extract special token IDs
        pad_id = token_to_id.get('PAD_None', 0)
        bos_id = token_to_id.get('BOS_None', 1)
        eos_id = token_to_id.get('EOS_None', 2)
        mask_id = token_to_id.get('MASK_None', 3)
        bar_id = token_to_id.get('Bar_None', 4)

        logger.info(f"Loaded vocabulary: {vocab_size} tokens")
        logger.info(f"Special tokens: PAD={pad_id}, BOS={bos_id}, EOS={eos_id}, "
                   f"MASK={mask_id}, Bar={bar_id}")

        vocab_info = VocabularyInfo(
            vocab_size=vocab_size,
            token_to_id=token_to_id,
            id_to_token=id_to_token,
            pad_token_id=pad_id,
            bos_token_id=bos_id,
            eos_token_id=eos_id,
            mask_token_id=mask_id,
            bar_token_id=bar_id
        )

        return vocab_info

    def verify_vocabulary_consistency(
        self,
        num_files_to_check: int = 10
    ) -> tuple[bool, List[str]]:
        """
        Verify that vocabulary is consistent across multiple JSON files.

        Args:
            num_files_to_check: Number of files to check

        Returns:
            Tuple of (is_consistent, list_of_issues)
        """
        json_files = list(self.json_dir.glob('*.json'))[:num_files_to_check]

        if not json_files:
            return False, ["No JSON files found"]

        issues = []
        reference_vocab = None
        reference_file = None

        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                if 'vocabulary' not in data:
                    issues.append(f"{json_file.name}: No vocabulary field")
                    continue

                current_vocab = data['vocabulary']

                if reference_vocab is None:
                    reference_vocab = current_vocab
                    reference_file = json_file.name
                else:
                    # Check for differences
                    if current_vocab != reference_vocab:
                        issues.append(
                            f"{json_file.name}: Vocabulary differs from {reference_file}"
                        )

                        # Report specific differences
                        ref_keys = set(reference_vocab.keys())
                        cur_keys = set(current_vocab.keys())

                        missing = ref_keys - cur_keys
                        extra = cur_keys - ref_keys

                        if missing:
                            issues.append(f"  Missing tokens: {list(missing)[:5]}")
                        if extra:
                            issues.append(f"  Extra tokens: {list(extra)[:5]}")

            except Exception as e:
                issues.append(f"{json_file.name}: Error loading - {e}")

        is_consistent = len(issues) == 0

        if is_consistent:
            logger.info(f"Vocabulary is consistent across {len(json_files)} files")
        else:
            logger.warning(f"Vocabulary inconsistencies found: {len(issues)} issues")

        return is_consistent, issues


def load_vocabulary(
    json_dir: Path,
    verify_consistency: bool = True,
    num_files_to_check: int = 10
) -> VocabularyInfo:
    """
    Main entry point for loading vocabulary.

    Args:
        json_dir: Directory containing tokenized JSON files
        verify_consistency: Whether to verify vocabulary across multiple files
        num_files_to_check: Number of files to check for consistency

    Returns:
        VocabularyInfo object

    Raises:
        ValueError: If vocabulary is inconsistent across files

    Example:
        >>> vocab = load_vocabulary(Path('processed'))
        >>> print(f"Vocabulary size: {vocab.vocab_size}")
        >>> print(f"PAD token ID: {vocab.pad_token_id}")
    """
    loader = VocabularyLoader(json_dir)

    # Load vocabulary from first file
    vocab_info = loader.load_vocabulary_from_json()

    # Verify consistency if requested
    if verify_consistency:
        is_consistent, issues = loader.verify_vocabulary_consistency(num_files_to_check)

        if not is_consistent:
            error_msg = "Vocabulary inconsistency detected:\n" + "\n".join(issues)
            raise ValueError(error_msg)

    return vocab_info


__all__ = [
    'VocabularyInfo',
    'VocabularyLoader',
    'load_vocabulary'
]
