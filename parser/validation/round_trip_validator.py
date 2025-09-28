"""
Main round-trip validator orchestrator.

This module provides the simplified main validator class that coordinates
all the specialized components for comprehensive MIDI round-trip validation.
"""

import logging
import time
from typing import Dict, Any, Optional, List, Tuple
from miditoolkit import MidiFile

# MidiTok imports
try:
    from miditok import MIDITokenizer
    MIDITOK_AVAILABLE = True
except ImportError:
    MIDITOK_AVAILABLE = False
    logging.warning("MidiTok not installed. Round-trip validation unavailable.")

from parser.config.defaults import MidiParserConfig, DEFAULT_CONFIG
from parser.core.midi_loader import ValidationResult
from parser.core.tokenizer_manager import TokenizerManager, TokenizationResult
from parser.core.track_analyzer import TrackInfo

from .validation_metrics import RoundTripMetrics, TokenizationError, DetokenizationError
from .note_matcher import NoteMatcher
from .midi_comparator import MidiComparator
from .tolerance_checker import ToleranceChecker
from .batch_processor import BatchValidator

logger = logging.getLogger(__name__)


class RoundTripValidator:
    """
    Simplified main orchestrator for MIDI round-trip validation.
    
    This class coordinates specialized components to perform comprehensive
    validation of MIDI tokenization fidelity through round-trip conversion.
    """
    
    def __init__(self, config: Optional[MidiParserConfig] = None):
        """
        Initialize the round-trip validator.
        
        Args:
            config: Parser configuration with validation tolerances
        """
        if not MIDITOK_AVAILABLE:
            raise ImportError("MidiTok is required for round-trip validation")
        
        self.config = config or DEFAULT_CONFIG
        self.validation_config = self.config.validation
        
        # Initialize core components
        self.tokenizer_manager = TokenizerManager(self.config)
        
        # Initialize specialized components
        tolerances = self.validation_config.tolerances
        self.note_matcher = NoteMatcher(tolerances)
        self.midi_comparator = MidiComparator(self.note_matcher, tolerances)
        self.tolerance_checker = ToleranceChecker(self.validation_config)
        
        # Initialize batch processor
        self.batch_processor = BatchValidator(self)
        
        # Cache for tokenizers
        self._tokenizer_cache = {}
        
    def validate_round_trip(
        self,
        midi: MidiFile,
        strategy: Optional[str] = None,
        track_infos: Optional[List[TrackInfo]] = None,
        detailed_report: bool = True
    ) -> Tuple[ValidationResult, RoundTripMetrics]:
        """
        Perform complete round-trip validation on a MIDI file.
        
        This is the main validation function that orchestrates the entire process.
        
        Args:
            midi: Original MIDI file
            strategy: Tokenization strategy to test
            track_infos: Optional track analysis information
            detailed_report: Whether to generate detailed comparison data
            
        Returns:
            Tuple of (ValidationResult, RoundTripMetrics)
        """
        start_time = time.time()
        strategy = strategy or self.config.tokenization
        
        logger.info(f"Starting round-trip validation with {strategy}")
        
        try:
            # Step 1: Tokenize the original MIDI
            tokenization_result = self._tokenize_midi(midi, strategy, track_infos)
            
            if not tokenization_result.success:
                return self._create_error_result(
                    f"Tokenization failed: {tokenization_result.error_message}",
                    start_time
                )
            
            # Step 2: Detokenize back to MIDI
            reconstructed_midi = self._detokenize_tokens(
                tokenization_result.tokens,
                strategy,
                midi.ticks_per_beat
            )
            
            if reconstructed_midi is None:
                return self._create_error_result("Detokenization failed", start_time)
            
            # Step 3: Compare original and reconstructed MIDI
            metrics = self.midi_comparator.compare_files(
                midi,
                reconstructed_midi,
                strategy,
                detailed_report
            )
            
            # Set additional metrics
            metrics.token_count = len(tokenization_result.tokens)
            metrics.processing_time = time.time() - start_time
            
            # Step 4: Apply tolerance checks
            validation_result = self.tolerance_checker.check_tolerances(metrics)
            
            # Log summary
            if validation_result.is_valid:
                logger.info(f"Round-trip validation PASSED for {strategy} "
                          f"(accuracy: {metrics.overall_accuracy:.2%})")
            else:
                logger.warning(f"Round-trip validation FAILED for {strategy}: "
                             f"{validation_result.errors}")
            
            return validation_result, metrics
            
        except Exception as e:
            logger.error(f"Round-trip validation error: {e}")
            return self._create_error_result(f"Validation error: {str(e)}", start_time)
    
    def _tokenize_midi(
        self,
        midi: MidiFile,
        strategy: str,
        track_infos: Optional[List[TrackInfo]] = None
    ) -> TokenizationResult:
        """
        Tokenize MIDI file using specified strategy.
        
        Args:
            midi: MIDI file to tokenize
            strategy: Tokenization strategy
            track_infos: Optional track information
            
        Returns:
            TokenizationResult with tokens
        """
        try:
            return self.tokenizer_manager.tokenize_midi(
                midi,
                strategy=strategy,
                track_infos=track_infos,
                max_seq_length=None  # No truncation for validation
            )
        except Exception as e:
            logger.error(f"Tokenization failed: {e}")
            raise TokenizationError(f"Failed to tokenize MIDI: {str(e)}")
    
    def _detokenize_tokens(
        self,
        tokens: List[int],
        strategy: str,
        ticks_per_beat: int = 480
    ) -> Optional[MidiFile]:
        """
        Detokenize tokens back to MIDI format.
        
        Args:
            tokens: Token sequence
            strategy: Tokenization strategy used
            ticks_per_beat: PPQ for the MIDI file
            
        Returns:
            Reconstructed MidiFile or None if failed
        """
        try:
            # Get or create tokenizer
            tokenizer = self.tokenizer_manager.create_tokenizer(strategy)
            
            # MidiTok detokenization - handle different API versions
            if hasattr(tokenizer, 'tokens_to_midi'):
                reconstructed = tokenizer.tokens_to_midi(tokens)
            elif hasattr(tokenizer, 'detokenize'):
                reconstructed = tokenizer.detokenize(tokens)
            else:
                # Fallback for older versions
                reconstructed = tokenizer(tokens, _=None)
            
            # Ensure we have a MidiFile object
            if isinstance(reconstructed, MidiFile):
                return reconstructed
            elif hasattr(reconstructed, 'to_midi'):
                return reconstructed.to_midi()
            else:
                logger.error(f"Unexpected detokenization output type: {type(reconstructed)}")
                return None
                
        except Exception as e:
            logger.error(f"Detokenization failed: {e}")
            raise DetokenizationError(f"Failed to detokenize: {str(e)}")
    
    def _create_error_result(
        self,
        error_message: str,
        start_time: float
    ) -> Tuple[ValidationResult, RoundTripMetrics]:
        """Create error result with timing information."""
        validation_result = ValidationResult(is_valid=False)
        validation_result.add_error(error_message)
        
        metrics = RoundTripMetrics()
        metrics.processing_time = time.time() - start_time
        
        return validation_result, metrics
    
    # Delegate batch operations to BatchValidator
    def validate_batch(
        self,
        midi_files: List[MidiFile],
        strategy: Optional[str] = None,
        parallel: bool = False,
        stop_on_failure: bool = False,
        max_workers: Optional[int] = None
    ) -> List[Tuple[ValidationResult, RoundTripMetrics]]:
        """
        Validate multiple MIDI files in batch.
        
        Args:
            midi_files: List of MIDI files to validate
            strategy: Tokenization strategy (uses config default if None)
            parallel: Whether to use parallel processing
            stop_on_failure: Whether to stop on first validation failure
            max_workers: Maximum number of parallel workers
            
        Returns:
            List of (ValidationResult, RoundTripMetrics) tuples
        """
        return self.batch_processor.validate_batch(
            midi_files, strategy, parallel, stop_on_failure, max_workers
        )
    
    def validate_chunked_midi(
        self,
        chunks: List[MidiFile],
        strategy: Optional[str] = None,
        original_midi: Optional[MidiFile] = None
    ) -> Tuple[ValidationResult, List[RoundTripMetrics]]:
        """
        Validate chunked MIDI files.
        
        Args:
            chunks: List of MIDI chunks
            strategy: Tokenization strategy
            original_midi: Original MIDI for reference
            
        Returns:
            Tuple of (overall ValidationResult, list of chunk metrics)
        """
        return self.batch_processor.validate_chunked_midi(
            chunks, strategy, original_midi
        )
    
    def validate_with_fallback(
        self,
        midi: MidiFile,
        primary_strategy: str,
        fallback_strategies: Optional[List[str]] = None
    ) -> Tuple[ValidationResult, RoundTripMetrics, str]:
        """
        Validate with automatic fallback to alternative strategies.
        
        Args:
            midi: MIDI file to validate
            primary_strategy: Primary tokenization strategy
            fallback_strategies: Alternative strategies to try
            
        Returns:
            Tuple of (ValidationResult, RoundTripMetrics, successful_strategy)
        """
        return self.batch_processor.validate_with_fallback(
            midi, primary_strategy, fallback_strategies
        )
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """Get summary of current validation configuration."""
        return {
            "tokenization_strategy": self.config.tokenization,
            "tolerances": self.tolerance_checker.get_tolerance_summary(),
            "quality_threshold": self.tolerance_checker.quality_threshold,
            "miditok_available": MIDITOK_AVAILABLE,
            "components": {
                "note_matcher": type(self.note_matcher).__name__,
                "midi_comparator": type(self.midi_comparator).__name__,
                "tolerance_checker": type(self.tolerance_checker).__name__,
                "batch_processor": type(self.batch_processor).__name__
            }
        }
    
    def update_tolerance(self, key: str, value: float) -> None:
        """Update tolerance settings across all components."""
        self.tolerance_checker.update_tolerance(key, value)
        self.note_matcher.set_tolerance(key, value)
        logger.info(f"Updated tolerance {key} to {value} across all components")
    
    def set_quality_threshold(self, threshold: float) -> None:
        """Set the overall quality threshold."""
        self.tolerance_checker.set_quality_threshold(threshold)
        logger.info(f"Quality threshold set to {threshold}")


# Export main classes and functions
__all__ = [
    'RoundTripValidator',
    'RoundTripMetrics',
    'ValidationResult',
    'TokenizationError',
    'DetokenizationError'
]