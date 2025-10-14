"""
Test script for vocabulary loading.

This script is a testing script that verifies that the vocabulary can be loaded
correctly from tokenized JSON files and that all expected tokens are present.

This is accomplished by running automated tests on the vocabulary system and
verifying that all the special tokens have correct IDs, also verifies that
token <-> mapping works both ways
"""

# python pytorch/scripts/test_vocab.py

import sys
from pathlib import Path

# Set UTF-8 encoding for Windows console
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

# Add parent directory to path to import modules
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from pytorch.data.vocab import load_vocabulary
from pytorch.data.constants import (
    PAD_TOKEN_ID, BOS_TOKEN_ID, EOS_TOKEN_ID, MASK_TOKEN_ID, BAR_TOKEN_ID,
    VOCAB_SIZE, get_token_type
)


def test_vocabulary_loading():
    """Test basic vocabulary loading."""
    print("=" * 60)
    print("Testing Vocabulary Loading")
    print("=" * 60)

    # Path to processed JSON files
    processed_dir = project_root / 'processed'

    if not processed_dir.exists():
        print(f"ERROR: Processed directory not found at {processed_dir}")
        return False

    print(f"\nLoading vocabulary from: {processed_dir}")

    try:
        vocab = load_vocabulary(processed_dir, verify_consistency=True, num_files_to_check=5)
        print(f"✓ Successfully loaded vocabulary")
    except Exception as e:
        print(f"✗ Failed to load vocabulary: {e}")
        return False

    # Test basic properties
    print(f"\n{'='*60}")
    print("Vocabulary Statistics")
    print(f"{'='*60}")
    print(f"Vocabulary size: {vocab.vocab_size}")
    print(f"Expected size: {VOCAB_SIZE}")

    if vocab.vocab_size != VOCAB_SIZE:
        print(f"⚠ WARNING: Size mismatch!")

    # Test special tokens
    print(f"\n{'='*60}")
    print("Special Tokens")
    print(f"{'='*60}")
    special_tokens = [
        ('PAD', vocab.pad_token_id, PAD_TOKEN_ID),
        ('BOS', vocab.bos_token_id, BOS_TOKEN_ID),
        ('EOS', vocab.eos_token_id, EOS_TOKEN_ID),
        ('MASK', vocab.mask_token_id, MASK_TOKEN_ID),
        ('Bar', vocab.bar_token_id, BAR_TOKEN_ID)
    ]

    all_correct = True
    for name, actual, expected in special_tokens:
        status = "✓" if actual == expected else "✗"
        print(f"{status} {name}: {actual} (expected: {expected})")
        if actual != expected:
            all_correct = False

    # Test token categories
    print(f"\n{'='*60}")
    print("Token Categories")
    print(f"{'='*60}")
    print(f"Pitch tokens: {len(vocab.pitch_tokens)}")
    print(f"Duration tokens: {len(vocab.duration_tokens)}")
    print(f"Position tokens: {len(vocab.position_tokens)}")
    print(f"Velocity tokens: {len(vocab.velocity_tokens)}")
    print(f"Tempo tokens: {len(vocab.tempo_tokens)}")
    print(f"Time signature tokens: {len(vocab.time_sig_tokens)}")
    print(f"Program tokens: {len(vocab.program_tokens)}")

    # Sample some tokens
    print(f"\n{'='*60}")
    print("Sample Tokens")
    print(f"{'='*60}")

    sample_tokens = [
        'Pitch_60',      # Middle C
        'Duration_1.0.4', # Quarter note
        'Position_0',    # Start of measure
        'Velocity_95',   # Forte
        'Tempo_123.33',  # Typical corridos tempo (closest to 125 in vocab)
        'TimeSig_6/8',   # Typical corridos time signature
        'Program_29',    # Chord program (guitar)
        'Program_98',    # Melody program (lead)
        'Bar_None',      # Bar marker
    ]

    for token_name in sample_tokens:
        token_id = vocab.get_token_id(token_name)
        if token_id is not None:
            retrieved_name = vocab.get_token_name(token_id)
            token_type = get_token_type(token_id)
            print(f"✓ {token_name:20s} → ID: {token_id:3d} → Type: {token_type}")
        else:
            print(f"✗ {token_name:20s} → NOT FOUND")

    # Test token type checking
    print(f"\n{'='*60}")
    print("Token Type Checking")
    print(f"{'='*60}")

    test_cases = [
        (vocab.get_token_id('Pitch_60'), 'pitch', vocab.is_pitch_token),
        (vocab.get_token_id('Duration_1.0.4'), 'duration', vocab.is_duration_token),
        (vocab.get_token_id('Position_0'), 'position', vocab.is_position_token),
        (vocab.get_token_id('Bar_None'), 'special', vocab.is_special_token),
    ]

    for token_id, expected_type, check_func in test_cases:
        if token_id is not None:
            result = check_func(token_id)
            status = "✓" if result else "✗"
            token_name = vocab.get_token_name(token_id)
            print(f"{status} {token_name:20s} is_{expected_type}: {result}")

    # Summary
    print(f"\n{'='*60}")
    print("Test Summary")
    print(f"{'='*60}")

    if all_correct and vocab.vocab_size == VOCAB_SIZE:
        print("✓ All tests passed!")
        return True
    else:
        print("⚠ Some tests failed")
        return False


def test_bidirectional_mapping():
    """Test that token_to_id and id_to_token are consistent."""
    print(f"\n{'='*60}")
    print("Testing Bidirectional Mapping")
    print(f"{'='*60}")

    processed_dir = project_root / 'processed'
    vocab = load_vocabulary(processed_dir, verify_consistency=False)

    errors = 0

    # Test forward and backward mapping
    for token_name, token_id in list(vocab.token_to_id.items())[:20]:  # Sample 20 tokens
        retrieved_name = vocab.get_token_name(token_id)
        if retrieved_name != token_name:
            print(f"✗ Mapping error: {token_name} → {token_id} → {retrieved_name}")
            errors += 1

    if errors == 0:
        print(f"✓ Bidirectional mapping is consistent (tested {20} tokens)")
    else:
        print(f"✗ Found {errors} mapping errors")

    return errors == 0


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("VOCABULARY LOADING TEST SUITE")
    print("=" * 60)

    # Run tests
    test1_passed = test_vocabulary_loading()
    test2_passed = test_bidirectional_mapping()

    # Final summary
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)

    if test1_passed and test2_passed:
        print("✓ All tests passed successfully!")
        return 0
    else:
        print("✗ Some tests failed")
        return 1


if __name__ == '__main__':
    sys.exit(main())
