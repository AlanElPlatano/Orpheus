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
MIDITOK_AVAILABLE = False
MIDITOK_VERSION = None
MIDITOK_IMPORT_ERROR = None

try:
    from miditok import REMI, TSD, Structured, CPWord, Octuple
    
    # Check if we can access a tokenizer to verify installation
    _test_tokenizer = REMI()
    
    MIDITOK_AVAILABLE = True
    
    # Try to get version
    try:
        import miditok
        MIDITOK_VERSION = getattr(miditok, '__version__', 'unknown')
    except:
        MIDITOK_VERSION = 'unknown'
    
    logging.info(f"MidiTok {MIDITOK_VERSION} loaded for round-trip validation")
    
except ImportError as e:
    MIDITOK_IMPORT_ERROR = f"ImportError: {str(e)}"
    logging.warning("MidiTok not installed. Round-trip validation unavailable.")
    logging.warning(f"Import error: {e}")
    
except Exception as e:
    MIDITOK_IMPORT_ERROR = f"{type(e).__name__}: {str(e)}"
    logging.error(f"MidiTok import failed: {e}")

from midi_parser.config.defaults import MidiParserConfig, DEFAULT_CONFIG
from midi_parser.core.midi_loader import ValidationResult
from midi_parser.core.tokenizer_manager import TokenizerManager, TokenizationResult
from midi_parser.core.track_analyzer import TrackInfo

from .validation_metrics import RoundTripMetrics, TokenizationError, DetokenizationError
from .note_matcher import NoteMatcher
from .midi_comparator import MidiComparator
from .tolerance_checker import ToleranceChecker
from .batch_processor import BatchValidator

logger = logging.getLogger(__name__)

# Log availability status
if not MIDITOK_AVAILABLE:
    logger.warning("=" * 60)
    logger.warning("ROUND-TRIP VALIDATION UNAVAILABLE")
    logger.warning(f"Reason: {MIDITOK_IMPORT_ERROR}")
    logger.warning("=" * 60)


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
            error_msg = (
                f"MidiTok is required for round-trip validation.\n"
                f"Error: {MIDITOK_IMPORT_ERROR}\n"
                f"Install with: pip install miditok"
            )
            raise ImportError(error_msg)
        
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
        
        # Cache for tokenizers and reconstructed MIDI
        self._tokenizer_cache = {}
        self._last_reconstructed_midi = None
        
        logger.info(f"RoundTripValidator initialized with MidiTok {MIDITOK_VERSION}")
        
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

        Caches the reconstructed MIDI for later use in quality analysis.
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
            
            # CACHE THE RECONSTRUCTED MIDI for later use in quality analysis
            self._last_reconstructed_midi = reconstructed_midi
            
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
            import traceback
            logger.error(traceback.format_exc())
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
            import traceback
            logger.error(traceback.format_exc())
            raise TokenizationError(f"Failed to tokenize MIDI: {str(e)}")
    
    def _detokenize_tokens(
        self,
        tokens: List[int],
        strategy: str,
        ticks_per_beat: int = 480
    ) -> Optional[MidiFile]:
        """
        Detokenize tokens back to MIDI format.
        
        MidiTok 3.x uses a different detokenization API than 2.x.
        
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
            
            logger.debug(f"Detokenizing {len(tokens)} tokens with {strategy}")
            
            # MidiTok 3.x detokenization API
            # Try different methods based on version
            reconstructed = None
            
            # Method 1: Direct call (miditok v3.x standard)
            try:
                # tokenizers can be called with tokens to detokenize
                reconstructed = tokenizer(tokens)
            except Exception as e1:
                logger.debug(f"Direct call failed: {e1}")
                
                # Method 2: tokens_to_midi method
                try:
                    if hasattr(tokenizer, 'tokens_to_midi'):
                        reconstructed = tokenizer.tokens_to_midi(tokens)
                except Exception as e2:
                    logger.debug(f"tokens_to_midi failed: {e2}")
                    
                    # Method 3: decode method
                    try:
                        if hasattr(tokenizer, 'decode'):
                            reconstructed = tokenizer.decode(tokens)
                    except Exception as e3:
                        logger.debug(f"decode failed: {e3}")
                        raise e1  # Raise the original error
            
            # Convert to MidiFile if needed
            if reconstructed is None:
                logger.error("All detokenization methods returned None")
                return None
                
            # Handle different return types
            if isinstance(reconstructed, MidiFile):
                logger.debug("Detokenization returned MidiFile directly")
                return reconstructed
                
            elif hasattr(reconstructed, 'to_midi'):
                logger.debug("Converting to MidiFile using .to_midi()")
                return reconstructed.to_midi()
                
            elif hasattr(reconstructed, 'dump'):
                # Some versions return a Score object
                logger.debug("Converting Score to MidiFile")
                temp_path = "/tmp/temp_detokenized.mid"
                reconstructed.dump(temp_path)
                return MidiFile(temp_path)
                
            else:
                logger.error(f"Unexpected detokenization output type: {type(reconstructed)}")
                logger.error(f"Available methods: {dir(reconstructed)}")
                return None
                
        except Exception as e:
            logger.error(f"Detokenization failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
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
    
    def get_last_reconstructed_midi(self) -> Optional[MidiFile]:
        """
        Get the last reconstructed MIDI from validation.
        
        This is useful for quality analysis that needs both original
        and reconstructed MIDI files.
        
        Returns:
            Last reconstructed MidiFile or None
        """
        return self._last_reconstructed_midi
    
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
            "miditok_version": MIDITOK_VERSION,
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


# Utility function to check if round-trip validation is available
def is_round_trip_available() -> bool:
    """
    Check if round-trip validation is available.
    
    Returns:
        True if MidiTok is properly installed
    """
    return MIDITOK_AVAILABLE


def get_round_trip_status() -> Dict[str, Any]:
    """
    Get detailed status of round-trip validation availability.
    
    Returns:
        Dictionary with availability status and details
    """
    return {
        "available": MIDITOK_AVAILABLE,
        "version": MIDITOK_VERSION,
        "error": MIDITOK_IMPORT_ERROR if not MIDITOK_AVAILABLE else None
    }


# Export main classes and functions
__all__ = [
    'RoundTripValidator',
    'RoundTripMetrics',
    'ValidationResult',
    'TokenizationError',
    'DetokenizationError',
    'is_round_trip_available',
    'get_round_trip_status',
    'MIDITOK_AVAILABLE',
    'MIDITOK_VERSION',
]
