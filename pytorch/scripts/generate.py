"""
CLI script for music generation.

Provides command-line interface for generating music from trained models.

Usage:

    # Generate with defaults
    python -m pytorch.scripts.generate --checkpoint pytorch/checkpoints/best_model.pt

    # Generate with custom settings
    python -m pytorch.scripts.generate \
        --checkpoint pytorch/checkpoints/best_model.pt \
        --num_files 10 \
        --output_dir ./generated \
        --mode creative \
        --temperature 1.1
"""

import argparse
import sys
from pathlib import Path
import logging

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from pytorch.generation import (
    GenerationConfig,
    MusicGenerator,
    create_quality_config,
    create_creative_config,
    create_custom_config
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate music from trained transformer models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Model loading
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help="Path to model checkpoint file"
    )

    # Generation settings
    parser.add_argument(
        '--num_files',
        type=int,
        default=5,
        help="Number of files to generate"
    )

    parser.add_argument(
        '--output_dir',
        type=str,
        default="./generated",
        help="Output directory for generated files"
    )

    # Mode selection
    parser.add_argument(
        '--mode',
        type=str,
        choices=['quality', 'creative', 'custom'],
        default='quality',
        help="Generation mode preset"
    )

    # Custom sampling parameters (used when mode='custom')
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.8,
        help="Sampling temperature (used in custom mode)"
    )

    parser.add_argument(
        '--top_p',
        type=float,
        default=0.95,
        help="Nucleus sampling threshold (used in custom mode)"
    )

    parser.add_argument(
        '--top_k',
        type=int,
        default=None,
        help="Top-k filtering (optional)"
    )

    parser.add_argument(
        '--repetition_penalty',
        type=float,
        default=1.1,
        help="Repetition penalty"
    )

    # Generation constraints
    parser.add_argument(
        '--max_length',
        type=int,
        default=2048,
        help="Maximum sequence length"
    )

    parser.add_argument(
        '--max_bars',
        type=int,
        default=32,
        help="Maximum number of bars to generate"
    )

    parser.add_argument(
        '--min_bars',
        type=int,
        default=8,
        help="Minimum number of bars for valid sequence"
    )

    # Retry settings
    parser.add_argument(
        '--max_retries',
        type=int,
        default=2,
        help="Maximum retry attempts on constraint violations"
    )

    parser.add_argument(
        '--retry_temperature_decay',
        type=float,
        default=0.95,
        help="Temperature decay factor on retry"
    )

    # Output settings
    parser.add_argument(
        '--save_tokens',
        action='store_true',
        help="Save intermediate token sequences as JSON"
    )

    parser.add_argument(
        '--filename_template',
        type=str,
        default="generated_{timestamp}_{index}",
        help="Template for output filenames"
    )

    # Validation
    parser.add_argument(
        '--skip_validation',
        action='store_true',
        help="Skip constraint validation"
    )

    # Random seed
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help="Random seed for reproducibility"
    )

    # Device
    parser.add_argument(
        '--device',
        type=str,
        choices=['cuda', 'cpu', 'auto'],
        default='auto',
        help="Device to use for generation"
    )

    return parser.parse_args()


def create_config_from_args(args) -> GenerationConfig:
    """
    Create GenerationConfig from parsed arguments.

    Args:
        args: Parsed command-line arguments

    Returns:
        GenerationConfig object
    """
    # Determine device
    if args.device == 'auto':
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device

    # Create config based on mode
    if args.mode == 'quality':
        config = create_quality_config()
    elif args.mode == 'creative':
        config = create_creative_config()
    else:  # custom
        config = create_custom_config(
            temperature=args.temperature,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty
        )

    # Override with CLI arguments
    config.checkpoint_path = Path(args.checkpoint)
    config.output_dir = Path(args.output_dir)
    config.device = device
    config.top_k = args.top_k
    config.max_length = args.max_length
    config.max_generation_bars = args.max_bars
    config.min_generation_bars = args.min_bars
    config.max_retries = args.max_retries
    config.retry_temperature_decay = args.retry_temperature_decay
    config.save_intermediate_tokens = args.save_tokens
    config.filename_template = args.filename_template
    config.validate_output = not args.skip_validation
    config.seed = args.seed

    return config


def main():
    """Main entry point for CLI generation."""
    args = parse_args()

    # Print header
    print("=" * 70)
    print("ORPHEUS MUSIC GENERATOR")
    print("=" * 70)
    print()

    # Create config
    config = create_config_from_args(args)

    # Print configuration
    print("Configuration:")
    print(f"  Checkpoint: {config.checkpoint_path}")
    print(f"  Output directory: {config.output_dir}")
    print(f"  Device: {config.device}")
    print(f"  Mode: {args.mode}")
    print(f"  Temperature: {config.temperature}")
    print(f"  Top-p: {config.top_p}")
    print(f"  Top-k: {config.top_k or 'None'}")
    print(f"  Repetition penalty: {config.repetition_penalty}")
    print(f"  Max length: {config.max_length}")
    print(f"  Max bars: {config.max_generation_bars}")
    print(f"  Max retries: {config.max_retries}")
    print(f"  Validation: {'Enabled' if config.validate_output else 'Disabled'}")
    print(f"  Seed: {config.seed or 'Random'}")
    print()

    # Check checkpoint exists
    if not config.checkpoint_path.exists():
        print(f"ERROR: Checkpoint not found: {config.checkpoint_path}")
        print("Please provide a valid checkpoint path.")
        sys.exit(1)

    # Create generator
    print("Loading model...")
    generator = MusicGenerator(config)

    if not generator.load_checkpoint(config.checkpoint_path):
        print("ERROR: Failed to load checkpoint")
        sys.exit(1)

    print("Model loaded successfully!")
    print()

    # Generate files
    print(f"Generating {args.num_files} files...")
    print()

    def progress_callback(current, total, result):
        """Print progress updates."""
        print(f"\n[{current}/{total}] {result.get_summary()}")
        if result.midi_path:
            print(f"  → {result.midi_path}")

    results = generator.generate_batch(
        num_files=args.num_files,
        progress_callback=progress_callback
    )

    # Print summary
    print()
    print("=" * 70)
    print("GENERATION COMPLETE")
    print("=" * 70)

    successful = sum(1 for r in results if r.success)
    failed = len(results) - successful
    total_time = sum(r.generation_time for r in results)
    avg_time = total_time / len(results) if results else 0
    total_violations = sum(r.num_violations for r in results)

    print(f"Total files: {len(results)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Total violations: {total_violations}")
    print(f"Total time: {total_time:.1f}s")
    print(f"Average time: {avg_time:.1f}s per file")
    print()

    if successful > 0:
        print(f"Generated files saved to: {config.output_dir}")
        print()

    # Print failed generations if any
    if failed > 0:
        print("Failed generations:")
        for i, result in enumerate(results, 1):
            if not result.success:
                print(f"  {i}. {result.error_message}")
        print()

    # Exit with appropriate code
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
