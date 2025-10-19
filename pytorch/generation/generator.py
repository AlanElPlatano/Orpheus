"""
Main music generator orchestrator.

Ties together all generation components: loading models, two-stage generation,
validation, MIDI export, and retry logic.
"""

import torch
import time
from pathlib import Path
from typing import Optional, List, Dict, Any
import logging
from datetime import datetime

from ..model.transformer import MusicTransformer
from ..data.vocab import load_vocabulary, VocabularyInfo
from .generation_config import GenerationConfig, GenerationResult
from .two_stage import TwoStageGenerator
from .validator import ConstraintValidator
from .midi_export import tokens_to_midi, save_token_sequence

logger = logging.getLogger(__name__)


class MusicGenerator:
    """
    Main music generator orchestrator.

    Handles model loading, generation, validation, and export for
    complete end-to-end music generation.
    """

    def __init__(self, config: GenerationConfig):
        """
        Initialize music generator.

        Args:
            config: Generation configuration
        """
        self.config = config
        self.model: Optional[MusicTransformer] = None
        self.vocab_info: Optional[VocabularyInfo] = None
        self.tokenizer_config: Optional[Dict[str, Any]] = None
        self.device = torch.device(config.device)

        self.two_stage_generator: Optional[TwoStageGenerator] = None
        self.validator: Optional[ConstraintValidator] = None

        self.is_loaded = False

        logger.info(f"MusicGenerator initialized (device: {self.device})")

    def load_checkpoint(self, checkpoint_path: Path) -> bool:
        """
        Load model and related data from checkpoint.

        Args:
            checkpoint_path: Path to model checkpoint

        Returns:
            True if loading succeeded
        """
        try:
            logger.info(f"Loading checkpoint from {checkpoint_path}")

            if not checkpoint_path.exists():
                logger.error(f"Checkpoint not found: {checkpoint_path}")
                return False

            # Load checkpoint
            checkpoint = torch.load(checkpoint_path, map_location=self.device)

            # Extract model configuration
            model_config = checkpoint.get('model_config', {})

            if not model_config:
                logger.warning("No model_config in checkpoint, using defaults")
                # Use default values from constants
                from ..data.constants import (
                    VOCAB_SIZE, HIDDEN_DIM, NUM_LAYERS, NUM_HEADS,
                    FF_DIM, CONTEXT_LENGTH, DROPOUT
                )
                model_config = {
                    'vocab_size': VOCAB_SIZE,
                    'hidden_dim': HIDDEN_DIM,
                    'num_layers': NUM_LAYERS,
                    'num_heads': NUM_HEADS,
                    'ff_dim': FF_DIM,
                    'max_len': CONTEXT_LENGTH,
                    'dropout': DROPOUT
                }

            # Create model
            from ..model.transformer import create_model

            self.model = create_model(
                vocab_size=model_config['vocab_size'],
                hidden_dim=model_config['hidden_dim'],
                num_layers=model_config['num_layers'],
                num_heads=model_config['num_heads'],
                ff_dim=model_config['ff_dim'],
                max_len=model_config['max_len'],
                dropout=model_config.get('dropout', 0.1),
                use_track_embeddings=model_config.get('use_track_embeddings', False),
                num_track_types=model_config.get('num_track_types', 2)
            )

            # Load model weights
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()

            logger.info(f"Model loaded: {self.model.get_num_params() / 1e6:.1f}M parameters")

            # Load vocabulary
            # Try to load from checkpoint first, fall back to processed directory
            if 'vocab_info' in checkpoint:
                self.vocab_info = checkpoint['vocab_info']
                logger.info("Loaded vocab_info from checkpoint")
            else:
                logger.info("Loading vocabulary from processed/ directory")
                from ..data.vocab import load_vocabulary
                self.vocab_info = load_vocabulary(Path("processed"))

            logger.info(f"Vocabulary size: {self.vocab_info.vocab_size}")

            # Load tokenizer config (for MIDI export)
            if 'tokenizer_config' in checkpoint:
                self.tokenizer_config = checkpoint['tokenizer_config']
                logger.info("Loaded tokenizer_config from checkpoint")
            else:
                logger.warning("No tokenizer_config in checkpoint, will use defaults for export")
                self.tokenizer_config = {
                    'pitch_range': (36, 84),
                    'beat_resolution': 4,
                    'num_velocities': 8,
                    'additional_tokens': {
                        'Chord': True,
                        'Rest': True,
                        'Tempo': True,
                        'TimeSignature': True
                    }
                }

            # Initialize two-stage generator
            # Enable track-aware sampling if model supports it
            use_track_aware = model_config.get('use_track_embeddings', False)

            self.two_stage_generator = TwoStageGenerator(
                model=self.model,
                vocab_info=self.vocab_info,
                config=self.config,
                device=str(self.device),
                use_track_aware_sampling=use_track_aware
            )

            logger.info(f"Two-stage generator track-aware sampling: {use_track_aware}")

            # Initialize validator
            self.validator = ConstraintValidator(self.vocab_info)

            self.is_loaded = True

            logger.info("Generator fully loaded and ready")

            return True

        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}", exc_info=True)
            return False

    def generate_single(
        self,
        prompt_tokens: Optional[List[int]] = None,
        seed: Optional[int] = None,
        index: int = 0
    ) -> GenerationResult:
        """
        Generate a single music file.

        Args:
            prompt_tokens: Optional conditioning tokens
            seed: Optional random seed
            index: Index for filename generation

        Returns:
            GenerationResult with all generation information
        """
        if not self.is_loaded:
            return GenerationResult(
                success=False,
                error_message="Generator not loaded. Call load_checkpoint() first."
            )

        result = GenerationResult(config=self.config)

        start_time = time.time()

        # Use provided seed or config seed
        if seed is not None:
            generation_seed = seed
        elif self.config.seed is not None:
            generation_seed = self.config.seed + index
        else:
            generation_seed = None

        # Attempt generation with retry logic
        for attempt in range(1, self.config.max_retries + 1):
            try:
                logger.info(f"Generation attempt {attempt}/{self.config.max_retries}")

                # Generate token sequence
                token_ids = self.two_stage_generator.generate_complete_sequence(
                    prompt_tokens=prompt_tokens,
                    seed=generation_seed
                )

                result.token_ids = token_ids
                result.sequence_length = len(token_ids)
                result.num_attempts = attempt
                result.temperature_used = self.config.temperature

                # Count bars
                from ..data.constants import BAR_TOKEN_ID
                result.num_bars = sum(1 for t in token_ids if t == BAR_TOKEN_ID)

                logger.info(f"Generated {len(token_ids)} tokens ({result.num_bars} bars)")

                # Validate if enabled
                if self.config.validate_output:
                    validation_report = self.validator.validate(token_ids)

                    result.is_valid = validation_report.is_valid
                    result.constraint_violations = validation_report.get_all_violations()
                    result.num_violations = validation_report.num_violations

                    if not validation_report.is_valid:
                        logger.warning(f"Validation failed: {validation_report.get_summary()}")

                        # Retry with lower temperature if configured
                        if attempt < self.config.max_retries:
                            logger.info("Retrying with lower temperature...")
                            self.config.temperature *= self.config.retry_temperature_decay
                            continue
                else:
                    result.is_valid = True

                # Generate filenames
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                base_filename = self.config.filename_template.format(
                    timestamp=timestamp,
                    index=index
                )

                # Save token sequence if configured
                if self.config.save_intermediate_tokens:
                    token_path = self.config.output_dir / f"{base_filename}_tokens.json"

                    save_success, save_error = save_token_sequence(
                        token_ids,
                        token_path,
                        self.vocab_info,
                        metadata={
                            'timestamp': timestamp,
                            'index': index,
                            'attempt': attempt,
                            'temperature': result.temperature_used,
                            'num_bars': result.num_bars,
                            'validation': {
                                'is_valid': result.is_valid,
                                'num_violations': result.num_violations
                            }
                        }
                    )

                    if save_success:
                        result.token_sequence_path = token_path

                # Convert to MIDI
                midi_path = self.config.output_dir / f"{base_filename}.mid"

                logger.info("Converting to MIDI...")

                midi_success, midi_output_path, midi_error = tokens_to_midi(
                    token_ids,
                    self.vocab_info,
                    midi_path,
                    self.tokenizer_config
                )

                if not midi_success:
                    result.success = False
                    result.error_message = f"MIDI conversion failed: {midi_error}"
                    logger.error(result.error_message)
                    return result

                result.midi_path = midi_output_path
                result.success = True

                break  # Success, exit retry loop

            except Exception as e:
                logger.error(f"Generation attempt {attempt} failed: {e}", exc_info=True)

                if attempt < self.config.max_retries:
                    logger.info("Retrying...")
                    self.config.temperature *= self.config.retry_temperature_decay
                else:
                    result.success = False
                    result.error_message = f"All {self.config.max_retries} attempts failed. Last error: {str(e)}"

        result.generation_time = time.time() - start_time

        # Log final result
        if result.success:
            logger.info(f"Generation successful: {result.get_summary()}")
        else:
            logger.error(f"Generation failed: {result.get_summary()}")

        return result

    def generate_batch(
        self,
        num_files: int,
        progress_callback=None
    ) -> List[GenerationResult]:
        """
        Generate multiple music files.

        Args:
            num_files: Number of files to generate
            progress_callback: Callback function(current, total, result)

        Returns:
            List of GenerationResult objects
        """
        if not self.is_loaded:
            return [GenerationResult(
                success=False,
                error_message="Generator not loaded. Call load_checkpoint() first."
            )]

        results = []

        logger.info(f"Starting batch generation: {num_files} files")

        for i in range(num_files):
            logger.info(f"\n{'='*60}")
            logger.info(f"Generating file {i+1}/{num_files}")
            logger.info(f"{'='*60}")

            result = self.generate_single(index=i+1)
            results.append(result)

            # Call progress callback if provided
            if progress_callback:
                progress_callback(i+1, num_files, result)

            # Reset temperature for next generation
            # (it may have been modified by retry logic)
            self.config.temperature = (
                self.config.temperature  # Will be overridden from original config if needed
            )

        # Log batch summary
        successful = sum(1 for r in results if r.success)
        failed = len(results) - successful

        total_time = sum(r.generation_time for r in results)
        avg_time = total_time / len(results) if results else 0

        logger.info(f"\n{'='*60}")
        logger.info(f"Batch Generation Complete")
        logger.info(f"{'='*60}")
        logger.info(f"Total files: {len(results)}")
        logger.info(f"Successful: {successful}")
        logger.info(f"Failed: {failed}")
        logger.info(f"Total time: {total_time:.1f}s")
        logger.info(f"Average time: {avg_time:.1f}s per file")
        logger.info(f"{'='*60}")

        return results


def load_generator_from_checkpoint(
    checkpoint_path: Path,
    config: Optional[GenerationConfig] = None
) -> Optional[MusicGenerator]:
    """
    Convenience function to load generator from checkpoint.

    Args:
        checkpoint_path: Path to model checkpoint
        config: Generation configuration (uses default if None)

    Returns:
        Loaded MusicGenerator, or None if loading failed
    """
    if config is None:
        from .generation_config import create_quality_config
        config = create_quality_config()

    config.checkpoint_path = checkpoint_path

    generator = MusicGenerator(config)

    if generator.load_checkpoint(checkpoint_path):
        return generator
    else:
        return None


__all__ = [
    'MusicGenerator',
    'load_generator_from_checkpoint'
]
