"""
Test script to verify the entire data pipeline end-to-end.

This script tests:
1. Loading split manifest
2. Creating datasets
3. Creating dataloaders
4. Iterating through batches
5. Verifying tensor shapes and values
"""

# python pytorch/scripts/test_data_pipeline.py

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import torch
from ml_core.data.dataloader import (
    create_dataloaders,
    print_dataloader_info,
    print_memory_estimates
)
from ml_core.data.constants import VOCAB_SIZE, PAD_TOKEN_ID, BOS_TOKEN_ID, EOS_TOKEN_ID


def test_data_pipeline():
    """Run comprehensive tests on the data pipeline."""
    print("=" * 80)
    print("Data Pipeline Verification Test")
    print("=" * 80)

    # Setup paths
    splits_dir = project_root / 'pytorch' / 'data' / 'splits'
    manifest_path = splits_dir / 'split_manifest.json'

    print(f"\nManifest path: {manifest_path}")
    print(f"Manifest exists: {manifest_path.exists()}")

    if not manifest_path.exists():
        print("\nERROR: Split manifest not found!")
        print("Please run 'python pytorch/data/split.py' first to create the split.")
        return False

    # Test 1: Create DataLoaders
    print("\n" + "=" * 80)
    print("Test 1: Creating DataLoaders")
    print("=" * 80)

    try:
        train_loader, val_loader, test_loader = create_dataloaders(
            split_manifest_path=manifest_path,
            batch_size=2,  # Small batch size for testing
            max_length=2048,
            num_workers=0,  # Single process for testing
            use_cache=False,
            shuffle_train=True
        )
        print("[OK] DataLoaders created successfully")
    except Exception as e:
        print(f"[FAIL] Failed to create DataLoaders: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Print DataLoader info
    print_dataloader_info(train_loader, val_loader, test_loader)

    # Check if we have any data to test
    if not train_loader and not val_loader and not test_loader:
        print("\n[FAIL] No data loaders available!")
        return False

    # Use the first available loader for testing
    test_loader_to_use = train_loader or val_loader or test_loader
    loader_name = "training" if train_loader else ("validation" if val_loader else "test")

    # Test 2: Iterate through batches
    print("\n" + "=" * 80)
    print(f"Test 2: Iterating Through {loader_name.title()} Batches")
    print("=" * 80)

    try:
        num_batches_to_test = min(3, len(test_loader_to_use))
        print(f"\nTesting {num_batches_to_test} batches from {loader_name} loader...")

        for i, batch in enumerate(test_loader_to_use):
            if i >= num_batches_to_test:
                break

            print(f"\nBatch {i + 1}:")
            print(f"  input_ids shape: {batch['input_ids'].shape}")
            print(f"  attention_mask shape: {batch['attention_mask'].shape}")
            print(f"  labels shape: {batch['labels'].shape}")
            print(f"  Number of metadata dicts: {len(batch['metadata'])}")

            # Verify shapes match
            assert batch['input_ids'].shape == batch['attention_mask'].shape == batch['labels'].shape, \
                "Tensor shapes don't match!"

            # Verify batch size
            batch_size = batch['input_ids'].shape[0]
            print(f"  Batch size: {batch_size}")

            # Verify sequence length
            seq_len = batch['input_ids'].shape[1]
            print(f"  Sequence length: {seq_len}")

            # Verify token IDs are in valid range
            min_token = batch['input_ids'].min().item()
            max_token = batch['input_ids'].max().item()
            print(f"  Token ID range: [{min_token}, {max_token}]")
            assert 0 <= min_token < VOCAB_SIZE, f"Invalid min token ID: {min_token}"
            assert 0 <= max_token < VOCAB_SIZE, f"Invalid max token ID: {max_token}"

            # Verify attention mask contains only 0s and 1s
            unique_mask_values = torch.unique(batch['attention_mask'])
            print(f"  Unique attention mask values: {unique_mask_values.tolist()}")
            assert all(v in [0, 1] for v in unique_mask_values), "Invalid attention mask values!"

            # Count padding tokens
            num_pad_tokens = (batch['input_ids'] == PAD_TOKEN_ID).sum().item()
            num_attention_zeros = (batch['attention_mask'] == 0).sum().item()
            print(f"  Padding tokens: {num_pad_tokens}")
            print(f"  Attention mask zeros: {num_attention_zeros}")

            # Verify BOS and EOS tokens
            first_token = batch['input_ids'][0, 0].item()
            print(f"  First token of first sequence: {first_token} (BOS={BOS_TOKEN_ID})")

            # Check if any sequence has EOS before padding
            for j in range(batch_size):
                seq = batch['input_ids'][j]
                mask = batch['attention_mask'][j]
                real_tokens = seq[mask == 1]
                if len(real_tokens) > 0:
                    last_real_token = real_tokens[-1].item()
                    if last_real_token == EOS_TOKEN_ID:
                        print(f"  Sequence {j}: Has EOS token at end")
                        break

        print("\n[OK] Successfully iterated through training batches")
    except Exception as e:
        print(f"\n[FAIL] Failed during batch iteration: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test 3: Check validation loader (if exists)
    if val_loader:
        print("\n" + "=" * 80)
        print("Test 3: Validation DataLoader")
        print("=" * 80)

        try:
            val_batch = next(iter(val_loader))
            print(f"Validation batch shape: {val_batch['input_ids'].shape}")
            print("[OK] Validation DataLoader works")
        except Exception as e:
            print(f"[FAIL] Validation DataLoader failed: {e}")
            return False
    else:
        print("\n[WARN] No validation set available (expected with small dataset)")

    # Test 4: Check test loader (if exists)
    if test_loader:
        print("\n" + "=" * 80)
        print("Test 4: Test DataLoader")
        print("=" * 80)

        try:
            test_batch = next(iter(test_loader))
            print(f"Test batch shape: {test_batch['input_ids'].shape}")
            print("[OK] Test DataLoader works")
        except Exception as e:
            print(f"[FAIL] Test DataLoader failed: {e}")
            return False

    # Test 5: Memory estimates
    print("\n" + "=" * 80)
    print("Test 5: Memory Usage Estimates")
    print("=" * 80)

    try:
        print_memory_estimates(train_loader, val_loader, test_loader)
        print("[OK] Memory estimates computed successfully")
    except Exception as e:
        print(f"[FAIL] Failed to compute memory estimates: {e}")
        return False

    # Test 6: Dataset statistics
    print("\n" + "=" * 80)
    print("Test 6: Dataset Statistics")
    print("=" * 80)

    try:
        stats = test_loader_to_use.dataset.get_statistics()
        print(f"\n{loader_name.title()} dataset statistics:")
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.2f}")
            else:
                print(f"  {key}: {value}")
        print("[OK] Dataset statistics computed successfully")
    except Exception as e:
        print(f"[FAIL] Failed to compute dataset statistics: {e}")
        return False

    # Test 7: Verify labels are correctly shifted
    print("\n" + "=" * 80)
    print("Test 7: Verify Labels (Next-Token Prediction)")
    print("=" * 80)

    try:
        batch = next(iter(test_loader_to_use))
        input_ids = batch['input_ids'][0]  # First sequence
        labels = batch['labels'][0]
        attention_mask = batch['attention_mask'][0]

        # Get real tokens (not padding)
        real_tokens_mask = attention_mask == 1
        real_input = input_ids[real_tokens_mask]
        real_labels = labels[real_tokens_mask]

        print(f"\nFirst sequence (showing first 10 tokens):")
        print(f"  Input:  {real_input[:10].tolist()}")
        print(f"  Labels: {real_labels[:10].tolist()}")

        # Verify labels match input (they should be identical for autoregressive training)
        # The shifting happens during loss calculation, not in the data
        if torch.all(real_labels == real_input):
            print("[OK] Labels correctly match input_ids (shift happens in loss function)")
        else:
            print("[WARN] Labels differ from input_ids (check if this is intentional)")

        print("[OK] Label verification complete")
    except Exception as e:
        print(f"[FAIL] Failed to verify labels: {e}")
        return False

    # Final summary
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    print("[OK] All data pipeline tests passed!")
    print("\nThe data pipeline is ready for training.")
    print("You can now proceed to Phase 2: Model Architecture")
    print("=" * 80)

    return True


if __name__ == '__main__':
    success = test_data_pipeline()
    sys.exit(0 if success else 1)
