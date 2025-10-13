"""
Transpose tokenized JSON files for data augmentation.

This script transposes tokenized MIDI JSON files to all 12 chromatic keys,
creating 11 additional versions of each source file (12 total including original).
This augmentation strategy dramatically expands the training dataset and helps
the model learn music theory intrinsically.
"""

# Transpose all files to all 12 keys (-6 to +5 semitones)
# python -m augmentation.transpose_tokenized_json processed/ augmented/

# Transpose with only higher transpositions (+1 to +5)
# python -m augmentation.transpose_tokenized_json processed/ augmented/ --mode higher

# Transpose with only lower transpositions (-6 to -1)
# python -m augmentation.transpose_tokenized_json processed/ augmented/ --mode lower

# Transpose single file by specific amount
# python -m augmentation.transpose_tokenized_json processed/song.json augmented/ --single --semitones 3

# Custom semitone offsets
# python -m augmentation.transpose_tokenized_json processed/ augmented/ --offsets -2 -1 1 2

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import hashlib
from datetime import datetime
import argparse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


SEMITONE_OFFSETS = {
    'all': list(range(-6, 6)),
    'higher': [1, 2, 3, 4, 5],
    'lower': [-6, -5, -4, -3, -2, -1],
    'default': list(range(-6, 6))
}

KEY_CIRCLE = [
    'C', 'Db', 'D', 'Eb', 'E', 'F',
    'F#', 'G', 'Ab', 'A', 'Bb', 'B'
]

KEY_CIRCLE_MINOR = [
    'Cm', 'Dbm', 'Dm', 'Ebm', 'Em', 'Fm',
    'F#m', 'Gm', 'Abm', 'Am', 'Bbm', 'Bm'
]


def calculate_new_key(original_key: str, semitones: int) -> str:
    """
    Calculate the new key after transposition.

    Args:
        original_key: Original key signature (e.g., 'Fm', 'C', 'Bb')
        semitones: Number of semitones to transpose (+/- 1-6)

    Returns:
        New key signature string

    Example:
        >>> calculate_new_key('Fm', 2)
        'Gm'
        >>> calculate_new_key('C', -3)
        'A'
    """
    is_minor = original_key.endswith('m')

    if is_minor:
        key_root = original_key[:-1]
        circle = KEY_CIRCLE_MINOR
    else:
        key_root = original_key
        circle = KEY_CIRCLE

    try:
        if is_minor:
            current_index = circle.index(original_key)
        else:
            current_index = circle.index(original_key)
    except ValueError:
        logger.warning(f"Key '{original_key}' not found in circle. Returning original.")
        return original_key

    new_index = (current_index + semitones) % 12
    return circle[new_index]


def transpose_tokens(
    tokens: List[int],
    vocabulary: Dict[str, int],
    semitones: int,
    pitch_range: Tuple[int, int] = (36, 84)
) -> Tuple[List[int], bool]:
    """
    Transpose pitch tokens in a token sequence.

    Args:
        tokens: List of token IDs
        vocabulary: Token vocabulary mapping (token_name -> token_id)
        semitones: Number of semitones to transpose
        pitch_range: Valid MIDI pitch range (min, max)

    Returns:
        Tuple of (transposed_tokens, success_flag)
    """
    reverse_vocab = {v: k for k, v in vocabulary.items()}

    pitch_token_map = {}
    for token_name, token_id in vocabulary.items():
        if token_name.startswith('Pitch_'):
            try:
                pitch_value = int(token_name.split('_')[1])
                pitch_token_map[token_id] = pitch_value
            except (IndexError, ValueError):
                continue

    transposed_tokens = []
    out_of_range_count = 0

    for token_id in tokens:
        if token_id in pitch_token_map:
            original_pitch = pitch_token_map[token_id]
            new_pitch = original_pitch + semitones

            if pitch_range[0] <= new_pitch <= pitch_range[1]:
                new_token_name = f'Pitch_{new_pitch}'
                if new_token_name in vocabulary:
                    transposed_tokens.append(vocabulary[new_token_name])
                else:
                    logger.warning(
                        f"Transposed pitch {new_pitch} not in vocabulary. "
                        f"Keeping original pitch {original_pitch}"
                    )
                    transposed_tokens.append(token_id)
                    out_of_range_count += 1
            else:
                logger.warning(
                    f"Pitch {new_pitch} out of range [{pitch_range[0]}, {pitch_range[1]}]. "
                    f"Keeping original pitch {original_pitch}"
                )
                transposed_tokens.append(token_id)
                out_of_range_count += 1
        else:
            transposed_tokens.append(token_id)

    success = out_of_range_count == 0

    if out_of_range_count > 0:
        logger.info(
            f"Transposition by {semitones:+d} semitones: "
            f"{out_of_range_count} notes kept at original pitch due to range constraints"
        )

    return transposed_tokens, success


def calculate_file_hash(tokens: List[int]) -> str:
    """
    Calculate MD5 hash of token sequence for deduplication.

    Args:
        tokens: Token sequence

    Returns:
        MD5 hash string
    """
    token_bytes = ''.join(str(t) for t in tokens).encode('utf-8')
    return hashlib.md5(token_bytes).hexdigest()


def transpose_json_file(
    input_path: Path,
    output_dir: Path,
    semitones: int,
    overwrite: bool = False
) -> Optional[Path]:
    """
    Transpose a single tokenized JSON file.

    Args:
        input_path: Path to input JSON file
        output_dir: Directory to save transposed file
        semitones: Number of semitones to transpose
        overwrite: Whether to overwrite existing files

    Returns:
        Path to output file if successful, None otherwise
    """
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        logger.error(f"Failed to load {input_path}: {e}")
        return None

    tokens = data.get('global_tokens', [])
    vocabulary = data.get('vocabulary', {})
    metadata = data.get('metadata', {})

    if not tokens:
        logger.error(f"No tokens found in {input_path}")
        return None

    if not vocabulary:
        logger.error(f"No vocabulary found in {input_path}")
        return None

    original_key = metadata.get('key_signature', 'C')
    new_key = calculate_new_key(original_key, semitones)

    pitch_range = data.get('tokenizer_config', {}).get('pitch_range', [36, 84])
    if isinstance(pitch_range, list):
        pitch_range = tuple(pitch_range)

    transposed_tokens, success = transpose_tokens(
        tokens,
        vocabulary,
        semitones,
        pitch_range
    )

    stem = input_path.stem
    if '-transpose' in stem:
        base_stem = stem.split('-transpose')[0]
    else:
        base_stem = stem

    sign = '+' if semitones >= 0 else ''
    output_filename = f"{base_stem}-transpose{sign}{semitones}.json"
    output_path = output_dir / output_filename

    if output_path.exists() and not overwrite:
        logger.info(f"Skipping existing file: {output_path}")
        return output_path

    output_data = data.copy()
    output_data['global_tokens'] = transposed_tokens
    output_data['file_hash'] = calculate_file_hash(transposed_tokens)
    output_data['sequence_length'] = len(transposed_tokens)

    if 'metadata' in output_data:
        output_data['metadata'] = metadata.copy()
        output_data['metadata']['key_signature'] = new_key

    if 'processing' in output_data:
        output_data['processing'] = {
            'timestamp': datetime.now().isoformat(),
            'transposition_semitones': semitones,
            'original_file': str(input_path.name),
            'original_key': original_key,
            'transposed_key': new_key,
            'transposition_successful': success
        }

    if 'source_file' in output_data:
        original_source = output_data['source_file']
        base_name = Path(original_source).stem
        ext = Path(original_source).suffix
        output_data['source_file'] = f"{base_name}-transpose{sign}{semitones}{ext}"

    try:
        output_dir.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=4, ensure_ascii=False)

        logger.info(
            f"Transposed {input_path.name} by {semitones:+d} semitones: "
            f"{original_key} -> {new_key} -> {output_path.name}"
        )

        return output_path

    except Exception as e:
        logger.error(f"Failed to write {output_path}: {e}")
        return None


def batch_transpose_directory(
    input_dir: Path,
    output_dir: Path,
    semitone_offsets: List[int] = None,
    overwrite: bool = False,
    pattern: str = "*.json"
) -> Dict[str, any]:
    """
    Transpose all JSON files in a directory to multiple keys.

    Args:
        input_dir: Directory containing source JSON files
        output_dir: Directory to save transposed files
        semitone_offsets: List of semitone offsets to apply (default: -6 to +5)
        overwrite: Whether to overwrite existing files
        pattern: File pattern to match (default: "*.json")

    Returns:
        Dictionary with processing statistics
    """
    if semitone_offsets is None:
        semitone_offsets = SEMITONE_OFFSETS['default']

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    if not input_dir.exists():
        logger.error(f"Input directory does not exist: {input_dir}")
        return {'error': 'Input directory not found'}

    json_files = list(input_dir.glob(pattern))

    if not json_files:
        logger.warning(f"No files matching '{pattern}' found in {input_dir}")
        return {'error': 'No matching files found'}

    stats = {
        'total_input_files': len(json_files),
        'total_transpositions': 0,
        'successful': 0,
        'failed': 0,
        'skipped': 0,
        'output_files': []
    }

    logger.info(f"Processing {len(json_files)} files with {len(semitone_offsets)} transpositions each")
    logger.info(f"Transposition offsets: {semitone_offsets}")

    for json_file in json_files:
        logger.info(f"\nProcessing: {json_file.name}")

        for semitones in semitone_offsets:
            if semitones == 0:
                continue

            stats['total_transpositions'] += 1

            output_path = transpose_json_file(
                json_file,
                output_dir,
                semitones,
                overwrite
            )

            if output_path:
                stats['successful'] += 1
                stats['output_files'].append(str(output_path))
            else:
                stats['failed'] += 1

    logger.info("\n" + "="*60)
    logger.info("BATCH TRANSPOSITION COMPLETE")
    logger.info("="*60)
    logger.info(f"Input files processed: {stats['total_input_files']}")
    logger.info(f"Total transpositions: {stats['total_transpositions']}")
    logger.info(f"Successful: {stats['successful']}")
    logger.info(f"Failed: {stats['failed']}")
    logger.info(f"Output directory: {output_dir}")
    logger.info("="*60)

    return stats


def main():
    """Command-line interface for the transposition script."""
    parser = argparse.ArgumentParser(
        description="Transpose tokenized MIDI JSON files for data augmentation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Transpose all files in 'processed' directory to all 12 keys
  python transpose_tokenized_json.py processed/ augmented/

  # Transpose with only higher transpositions (+1 to +5)
  python transpose_tokenized_json.py processed/ augmented/ --mode higher

  # Transpose single file by +3 semitones
  python transpose_tokenized_json.py processed/song.json augmented/ --single --semitones 3

  # Use custom semitone offsets
  python transpose_tokenized_json.py processed/ augmented/ --offsets -2 -1 1 2
        """
    )

    parser.add_argument(
        'input',
        type=str,
        help='Input JSON file or directory'
    )

    parser.add_argument(
        'output',
        type=str,
        help='Output directory for transposed files'
    )

    parser.add_argument(
        '--mode',
        choices=['all', 'higher', 'lower', 'default'],
        default='default',
        help='Transposition mode (default: all keys except original)'
    )

    parser.add_argument(
        '--offsets',
        nargs='+',
        type=int,
        help='Custom semitone offsets (e.g., -2 -1 1 2)'
    )

    parser.add_argument(
        '--single',
        action='store_true',
        help='Process single file instead of directory'
    )

    parser.add_argument(
        '--semitones',
        type=int,
        help='Semitones to transpose (for single file mode)'
    )

    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite existing output files'
    )

    parser.add_argument(
        '--pattern',
        type=str,
        default='*.json',
        help='File pattern for directory mode (default: *.json)'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    input_path = Path(args.input)
    output_dir = Path(args.output)

    if args.single:
        if not input_path.is_file():
            logger.error(f"Input file not found: {input_path}")
            return 1

        semitones = args.semitones
        if semitones is None:
            logger.error("--semitones required for single file mode")
            return 1

        result = transpose_json_file(
            input_path,
            output_dir,
            semitones,
            args.overwrite
        )

        if result:
            logger.info(f"Successfully created: {result}")
            return 0
        else:
            logger.error("Transposition failed")
            return 1

    else:
        if not input_path.is_dir():
            logger.error(f"Input directory not found: {input_path}")
            return 1

        if args.offsets:
            semitone_offsets = args.offsets
        else:
            semitone_offsets = SEMITONE_OFFSETS[args.mode]

        stats = batch_transpose_directory(
            input_path,
            output_dir,
            semitone_offsets,
            args.overwrite,
            args.pattern
        )

        if 'error' in stats:
            return 1

        return 0


if __name__ == '__main__':
    exit(main())
