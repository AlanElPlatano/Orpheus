"""
MIDI tokenization manager module using MidiTok 3.x.

This module provides the core tokenization functionality, managing multiple
tokenization strategies and handling the conversion from MIDI to token sequences
suitable for transformer models.
"""

import logging
from typing import Dict, Any, Optional, List, Tuple, Union, TYPE_CHECKING
from dataclasses import dataclass, field
from pathlib import Path
from functools import lru_cache
import json
import pickle
import hashlib
import sys

MIDITOK_AVAILABLE = False
MIDITOK_VERSION = None
MIDITOK_IMPORT_ERROR = None

try:
    # Import the tokenizer classes
    from miditok import REMI, TSD, MusicTokenizer, CPWord, Octuple
    from miditok.classes import TokSequence
    
    # Try to import symusic (required for MidiTok 3.x)
    try:
        import symusic
        SYMUSIC_AVAILABLE = True
    except ImportError:
        SYMUSIC_AVAILABLE = False
        logger = logging.getLogger(__name__)
        logger.warning("symusic not installed. MidiTok 3.x requires symusic for best performance.")
    
    MIDITOK_AVAILABLE = True
    
    # Get version
    try:
        import miditok
        MIDITOK_VERSION = getattr(miditok, '__version__', '3.0.6+')
    except:
        MIDITOK_VERSION = '3.0.6+'
    
    logging.info(f"MidiTok imported successfully (version: {MIDITOK_VERSION})")
    
except ImportError as e:
    MIDITOK_IMPORT_ERROR = f"ImportError: {str(e)}"
    TokSequence = Any
    REMI = None
    TSD = None
    MusicTokenizer = None
    CPWord = None
    Octuple = None
    logging.warning(f"MidiTok not installed. Install with: pip install miditok")
    logging.warning(f"Import error details: {e}")
    
except Exception as e:
    MIDITOK_IMPORT_ERROR = f"{type(e).__name__}: {str(e)}"
    TokSequence = Any
    REMI = None
    TSD = None
    MusicTokenizer = None
    CPWord = None
    Octuple = None
    logging.error(f"MidiTok import failed with unexpected error: {e}")
    import traceback
    logging.error(f"Traceback: {traceback.format_exc()}")

from miditoolkit import MidiFile

from midi_parser.config.defaults import (
    MidiParserConfig,
    TokenizerConfig,
    DEFAULT_CONFIG,
    ERROR_HANDLING_STRATEGIES,
    get_strategy_info,
    get_available_strategies
)
from midi_parser.core.track_analyzer import TrackInfo

logger = logging.getLogger(__name__)

if not MIDITOK_AVAILABLE:
    logger.warning("=" * 60)
    logger.warning("MIDITOK NOT AVAILABLE")
    logger.warning("=" * 60)
    logger.warning(f"Error: {MIDITOK_IMPORT_ERROR}")
    logger.warning("Python path:")
    for i, path in enumerate(sys.path[:5], 1):
        logger.warning(f"  {i}. {path}")
    logger.warning("=" * 60)
else:
    logger.info(f"MidiTok version {MIDITOK_VERSION} loaded successfully")


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class TokenizationResult:
    """Result of tokenizing a MIDI file."""
    tokens: List[int] = field(default_factory=list)
    token_types: Optional[List[str]] = None
    vocabulary: Optional[Dict[str, int]] = None
    vocabulary_size: int = 0
    sequence_length: int = 0
    tokenization_strategy: str = "REMI"
    success: bool = True
    error_message: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    processing_time: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "tokens": self.tokens,
            "vocabulary_size": self.vocabulary_size,
            "sequence_length": self.sequence_length,
            "tokenization_strategy": self.tokenization_strategy,
            "success": self.success,
            "error_message": self.error_message,
            "warnings": self.warnings,
            "processing_time": round(self.processing_time, 3)
        }


@dataclass
class TokenizerCache:
    """Cache for tokenizer instances to avoid recreation."""
    tokenizers: Dict[str, Any] = field(default_factory=dict)
    configs: Dict[str, str] = field(default_factory=dict)
    max_cache_size: int = 10
    
    def get_cache_key(self, strategy: str, config: TokenizerConfig) -> str:
        """Generate unique cache key for tokenizer configuration."""
        config_str = json.dumps({
            "strategy": strategy,
            "pitch_range": config.pitch_range,
            "beat_resolution": config.beat_resolution,
            "num_velocities": config.num_velocities,
            "additional_tokens": config.additional_tokens,
        }, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()
    
    def get(self, strategy: str, config: TokenizerConfig) -> Optional[Any]:
        """Get cached tokenizer if available."""
        key = self.get_cache_key(strategy, config)
        return self.tokenizers.get(key)
    
    def put(self, strategy: str, config: TokenizerConfig, tokenizer: Any) -> None:
        """Cache a tokenizer instance."""
        key = self.get_cache_key(strategy, config)
        
        if len(self.tokenizers) >= self.max_cache_size:
            oldest_key = next(iter(self.tokenizers))
            del self.tokenizers[oldest_key]
            if oldest_key in self.configs.values():
                self.configs = {k: v for k, v in self.configs.items() if v != oldest_key}
        
        self.tokenizers[key] = tokenizer
        self.configs[key] = key


# ============================================================================
# Tokenizer Manager Class
# ============================================================================

class TokenizerManager:
    """
    Manager for MidiTok 3.x tokenizers with multi-strategy support.
    """
    
    STRATEGY_CLASSES = {
        "REMI": REMI,
        "TSD": TSD,
        "Structured": MusicTokenizer,  # MusicTokenizer is the base for Structured in 3.x
        "CPWord": CPWord,
        "Octuple": Octuple
    } if MIDITOK_AVAILABLE else {}
    
    def __init__(self, config: Optional[MidiParserConfig] = None):
        """
        Initialize the TokenizerManager.
        
        Args:
            config: Parser configuration (uses default if not provided)
        """
        if not MIDITOK_AVAILABLE:
            error_msg = (
                f"MidiTok is required but not available.\n"
                f"Error: {MIDITOK_IMPORT_ERROR}\n"
                f"Install with: pip install miditok symusic"
            )
            logger.error(error_msg)
            raise ImportError(error_msg)
        
        self.config = config or DEFAULT_CONFIG
        self.cache = TokenizerCache()
        self._fallback_strategies = ["REMI", "TSD", "Structured"]
        
        logger.info(f"TokenizerManager initialized with MidiTok {MIDITOK_VERSION}")
        
    def create_tokenizer(
        self,
        strategy: str,
        tokenizer_config: Optional[TokenizerConfig] = None
    ) -> Any:
        """
        Create or retrieve a cached tokenizer instance.
        
        Args:
            strategy: Tokenization strategy name
            tokenizer_config: Optional tokenizer configuration override
            
        Returns:
            MIDITokenizer instance
            
        Raises:
            ValueError: If strategy is not supported
        """
        if strategy not in self.STRATEGY_CLASSES:
            available = list(self.STRATEGY_CLASSES.keys())
            raise ValueError(f"Unsupported strategy '{strategy}'. Available: {available}")
        
        tok_config = tokenizer_config or self.config.tokenizer
        
        # Check cache first
        cached = self.cache.get(strategy, tok_config)
        if cached is not None:
            logger.debug(f"Using cached {strategy} tokenizer")
            return cached
        
        logger.info(f"Creating new {strategy} tokenizer")
        
        # Build MidiTok 3.x configuration
        miditok_config = self._build_miditok_config(strategy, tok_config)
        
        # Create tokenizer instance
        tokenizer_class = self.STRATEGY_CLASSES[strategy]
        
        try:
            # MidiTok 3.x uses TokenizerConfig objects
            from miditok import TokenizerConfig as MidiTokConfig
            tokenizer = tokenizer_class(tokenizer_config=miditok_config)
        except Exception as e:
            logger.warning(f"Failed with full config: {e}")
            # Try minimal config
            try:
                tokenizer = tokenizer_class()
            except Exception as e2:
                logger.error(f"Failed to create tokenizer: {e2}")
                raise
        
        # Cache the tokenizer
        self.cache.put(strategy, tok_config, tokenizer)
        
        return tokenizer
    
    def _build_miditok_config(
        self,
        strategy: str,
        config: TokenizerConfig
    ) -> Any:
        """
        Build TokenizerConfig for MidiTok 3.x.
        
        MidiTok 3.x uses TokenizerConfig dataclass instead of kwargs.
        """
        from miditok import TokenizerConfig as MidiTokConfig
        
        # Build config dict
        config_dict = {
            "pitch_range": config.pitch_range,
            "beat_res": {(0, 4): config.beat_resolution},
            "num_velocities": config.num_velocities,
        }
        
        # Add special tokens based on our config
        special_tokens = []
        if config.additional_tokens.get("Chord"):
            special_tokens.append("Chord")
        if config.additional_tokens.get("Rest"):
            special_tokens.append("Rest")
        if config.additional_tokens.get("Tempo"):
            special_tokens.append("Tempo")
        if config.additional_tokens.get("TimeSignature"):
            special_tokens.append("TimeSignature")
        if config.additional_tokens.get("Program"):
            special_tokens.append("Program")
        
        if special_tokens:
            config_dict["special_tokens"] = special_tokens
        
        # Create TokenizerConfig
        try:
            miditok_config = MidiTokConfig(**config_dict)
        except Exception as e:
            logger.warning(f"Failed to create config with special tokens: {e}")
            # Try without special tokens
            minimal_dict = {
                "pitch_range": config.pitch_range,
                "beat_res": {(0, 4): config.beat_resolution},
                "num_velocities": config.num_velocities,
            }
            miditok_config = MidiTokConfig(**minimal_dict)
        
        return miditok_config
    
    def tokenize_midi(
        self,
        midi: MidiFile,
        strategy: Optional[str] = None,
        track_infos: Optional[List[TrackInfo]] = None,
        max_seq_length: Optional[int] = None,
        auto_select: bool = False
    ) -> TokenizationResult:
        """
        Tokenize a MIDI file using the specified strategy.

        Args:
            midi: MidiFile object to tokenize
            strategy: Tokenization strategy (uses config default if None)
            track_infos: Optional track analysis results for optimization
            max_seq_length: Maximum sequence length (uses config default if None)
            auto_select: Whether to auto-select best strategy based on content
            
        Returns:
            TokenizationResult with tokens and metadata

        - Returns List[TokSequence] instead of List[int]
        - Uses symusic.Score as backend instead of miditoolkit
        - Need to extract .ids from TokSequence objects
        """
        import time
        start_time = time.time()
        
        # Determine strategy
        if auto_select and track_infos:
            strategy = self._auto_select_strategy(midi, track_infos)
            logger.info(f"Auto-selected strategy: {strategy}")
        elif strategy is None:
            strategy = self.config.tokenization
        
        max_seq_length = max_seq_length or self.config.tokenizer.max_seq_length
        
        # Initialize result
        result = TokenizationResult(tokenization_strategy=strategy)
        
        try:
            # Get or create tokenizer
            tokenizer = self.create_tokenizer(strategy)
            
            # Tokenize the MIDI file
            tokens = self._tokenize_with_strategy(midi, tokenizer, strategy)
            
            # Handle sequence length constraints
            if len(tokens) > max_seq_length:
                result.warnings.append(
                    f"Token sequence ({len(tokens)}) exceeds max length ({max_seq_length}). "
                    "Truncating."
                )
                tokens = tokens[:max_seq_length]
            
            # Build result
            result.tokens = tokens
            result.sequence_length = len(tokens)
            
            # Get vocabulary - MidiTok 3.x stores vocab differently
            vocab = self._get_vocabulary(tokenizer)
            result.vocabulary_size = len(vocab)
            result.vocabulary = vocab
            result.success = True
            
            logger.info(f"Successfully tokenized with {strategy}: {len(tokens)} tokens")
            
        except Exception as e:
            logger.error(f"Tokenization failed with {strategy}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            
            # Try fallback strategies
            fallback_result = self._try_fallback_tokenization(
                midi, strategy, max_seq_length, str(e)
            )
            if fallback_result:
                result = fallback_result
            else:
                result.success = False
                result.error_message = f"All tokenization strategies failed. Last error: {e}"
        
        result.processing_time = time.time() - start_time
        return result
    
    def _tokenize_with_strategy(
        self,
        midi: MidiFile,
        tokenizer: Any,
        strategy: str
    ) -> List[int]:
        """
        Apply tokenization with MidiTok 3.x API.

        Args:
            midi: MIDI file to tokenize
            tokenizer: Tokenizer instance
            strategy: Strategy name
            
        Returns:
            List of integer tokens
        
        MidiTok 3.x returns TokSequence objects that need proper handling.
        Some tokenizers use multi-dimensional tokens (track, token_id).
        """
        try:
            # Convert miditoolkit.MidiFile to symusic.Score if available
            if SYMUSIC_AVAILABLE:
                import symusic
                try:
                    score = symusic.Score(midi)
                except:
                    score = midi
            else:
                score = midi
            
            # Call tokenizer - returns TokSequence or List[TokSequence]
            tok_sequences = tokenizer(score)
            
            # Store the raw TokSequence for later detokenization
            # This is important because some tokenizers need the full structure
            self._last_tok_sequence = tok_sequences
            
            # Extract tokens while preserving structure information
            tokens = []
            
            if isinstance(tok_sequences, list):
                # Multiple tracks/sequences
                for seq in tok_sequences:
                    if hasattr(seq, 'ids'):
                        # TokSequence object - get ids
                        seq_ids = seq.ids
                        # Handle both 1D and 2D token structures
                        if seq_ids and isinstance(seq_ids[0], (list, tuple)):
                            # 2D structure - flatten but this loses info
                            # Better to keep the sequence structure
                            tokens.extend([id for sublist in seq_ids for id in sublist])
                        else:
                            tokens.extend(seq_ids)
                    elif isinstance(seq, (list, tuple)):
                        tokens.extend(seq)
            elif hasattr(tok_sequences, 'ids'):
                # Single TokSequence
                seq_ids = tok_sequences.ids
                if seq_ids and isinstance(seq_ids[0], (list, tuple)):
                    tokens = [id for sublist in seq_ids for id in sublist]
                else:
                    tokens = list(seq_ids)
            
            # Ensure all are integers (handle any remaining nested structures)
            flat_tokens = []
            for t in tokens:
                if isinstance(t, int):
                    flat_tokens.append(t)
                elif isinstance(t, (list, tuple)):
                    # Should not happen, but handle it
                    flat_tokens.extend([int(x) for x in t if isinstance(x, int)])
                else:
                    try:
                        flat_tokens.append(int(t))
                    except (ValueError, TypeError):
                        logger.warning(f"Skipping non-integer token: {t}")
            
            return flat_tokens
            
        except Exception as e:
            logger.error(f"Tokenization failed: {e}")
            raise
    
    def _get_vocabulary(self, tokenizer: Any) -> Dict[str, int]:
        """
        Extract vocabulary from MidiTok 3.x tokenizer.
        
        MidiTok 3.x stores vocabulary in tokenizer.vocab attribute.
        """
        try:
            if hasattr(tokenizer, 'vocab'):
                vocab = tokenizer.vocab
                # Convert to dict if needed
                if hasattr(vocab, '__iter__') and not isinstance(vocab, dict):
                    return {str(k): v for k, v in enumerate(vocab)}
                return dict(vocab) if isinstance(vocab, dict) else {}
            return {}
        except Exception as e:
            logger.warning(f"Failed to extract vocabulary: {e}")
            return {}
    
    def _auto_select_strategy(
        self,
        midi: MidiFile,
        track_infos: List[TrackInfo]
    ) -> str:
        """
        Auto-select best tokenization strategy based on MIDI content.
        
        Args:
            midi: MIDI file being processed
            track_infos: Analyzed track information
            
        Returns:
            Recommended strategy name
        """
        total_notes = sum(t.statistics.total_notes for t in track_infos)
        avg_polyphony = sum(
            t.statistics.avg_polyphony * t.statistics.total_notes 
            for t in track_infos
        ) / max(total_notes, 1)
        
        has_drums = any(t.type == "drums" for t in track_infos)
        has_complex_harmony = any(
            t.type == "chord" and t.statistics.avg_polyphony > 4 
            for t in track_infos
        )
        
        duration_seconds = midi.get_tick_to_time_mapping()[-1] if midi.max_tick > 0 else 0
        
        if duration_seconds > 600:
            return "CPWord"
        elif has_complex_harmony and avg_polyphony > 3:
            return "Structured"
        elif total_notes < 500 and not has_drums:
            return "TSD"
        elif has_drums or len(track_infos) > 4:
            return "REMI"
        else:
            return "REMI"
    
    def _try_fallback_tokenization(
        self,
        midi: MidiFile,
        failed_strategy: str,
        max_seq_length: int,
        error_message: str
    ) -> Optional[TokenizationResult]:
        """
        Try fallback tokenization strategies after primary failure.
        
        Args:
            midi: MIDI file to tokenize
            failed_strategy: Strategy that failed
            max_seq_length: Maximum sequence length
            error_message: Error from primary attempt
            
        Returns:
            TokenizationResult if successful, None if all fallbacks fail
        """
        fallback_strategies = [s for s in self._fallback_strategies if s != failed_strategy]
        
        logger.info(f"Trying fallback strategies: {fallback_strategies}")
        
        for strategy in fallback_strategies:
            try:
                logger.info(f"Attempting fallback with {strategy}")
                
                tokenizer = self.create_tokenizer(strategy)
                tokens = self._tokenize_with_strategy(midi, tokenizer, strategy)
                
                if len(tokens) > max_seq_length:
                    tokens = tokens[:max_seq_length]
                
                vocab = self._get_vocabulary(tokenizer)
                
                result = TokenizationResult(
                    tokens=tokens,
                    sequence_length=len(tokens),
                    vocabulary_size=len(vocab),
                    vocabulary=vocab,
                    tokenization_strategy=strategy,
                    success=True,
                    warnings=[
                        f"Used fallback strategy {strategy} after {failed_strategy} failed: {error_message}"
                    ]
                )
                
                logger.info(f"Fallback successful with {strategy}")
                return result
                
            except Exception as e:
                logger.warning(f"Fallback {strategy} also failed: {e}")
                continue
        
        return None
    
    def batch_tokenize(
        self,
        midi_files: List[MidiFile],
        track_infos_list: Optional[List[List[TrackInfo]]] = None,
        strategy: Optional[str] = None,
        parallel: bool = False
    ) -> List[TokenizationResult]:
        """
        Tokenize multiple MIDI files efficiently.
        
        Args:
            midi_files: List of MidiFile objects
            track_infos_list: Optional list of track analysis results
            strategy: Tokenization strategy (can be different per file if None)
            parallel: Whether to use parallel processing
            
        Returns:
            List of TokenizationResult objects
        """
        results = []
        
        if parallel and len(midi_files) > 1:
            import concurrent.futures
            import multiprocessing
            
            max_workers = self.config.processing.max_workers or multiprocessing.cpu_count()
            
            with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = []
                for i, midi in enumerate(midi_files):
                    track_infos = track_infos_list[i] if track_infos_list else None
                    future = executor.submit(
                        self.tokenize_midi,
                        midi,
                        strategy,
                        track_infos,
                        self.config.tokenizer.max_seq_length,
                        auto_select=(strategy is None)
                    )
                    futures.append(future)
                
                for future in concurrent.futures.as_completed(futures):
                    try:
                        result = future.result(timeout=30)
                        results.append(result)
                    except Exception as e:
                        logger.error(f"Batch tokenization error: {e}")
                        results.append(TokenizationResult(
                            success=False,
                            error_message=str(e)
                        ))
        else:
            for i, midi in enumerate(midi_files):
                track_infos = track_infos_list[i] if track_infos_list else None
                result = self.tokenize_midi(
                    midi,
                    strategy,
                    track_infos,
                    auto_select=(strategy is None)
                )
                results.append(result)
        
        return results
    
    def get_vocabulary_info(self, strategy: Optional[str] = None) -> Dict[str, Any]:
        """
        Get information about the vocabulary for a strategy.
        
        Args:
            strategy: Tokenization strategy (uses config default if None)
            
        Returns:
            Dictionary with vocabulary information
        """
        strategy = strategy or self.config.tokenization
        tokenizer = self.create_tokenizer(strategy)
        
        vocab = self._get_vocabulary(tokenizer)
        
        if not vocab:
            return {
                "strategy": strategy,
                "total_tokens": 0,
                "categories": {},
                "sample_tokens": {}
            }
        
        # Analyze vocabulary
        token_categories = {
            "note": 0, "velocity": 0, "duration": 0, "position": 0,
            "bar": 0, "chord": 0, "rest": 0, "tempo": 0,
            "time_signature": 0, "program": 0, "special": 0
        }
        
        for token_str in vocab.keys():
            token_lower = str(token_str).lower()
            if "note" in token_lower or "pitch" in token_lower:
                token_categories["note"] += 1
            elif "velocity" in token_lower or "vel" in token_lower:
                token_categories["velocity"] += 1
            elif "duration" in token_lower or "dur" in token_lower:
                token_categories["duration"] += 1
            elif "position" in token_lower or "pos" in token_lower:
                token_categories["position"] += 1
            elif "bar" in token_lower:
                token_categories["bar"] += 1
            elif "chord" in token_lower:
                token_categories["chord"] += 1
            elif "rest" in token_lower:
                token_categories["rest"] += 1
            elif "tempo" in token_lower:
                token_categories["tempo"] += 1
            elif "time" in token_lower and "signature" in token_lower:
                token_categories["time_signature"] += 1
            elif "program" in token_lower:
                token_categories["program"] += 1
            else:
                token_categories["special"] += 1
        
        return {
            "strategy": strategy,
            "total_tokens": len(vocab),
            "categories": token_categories,
            "sample_tokens": dict(list(vocab.items())[:20])
        }


# ============================================================================
# Utility Functions
# ============================================================================

def create_adaptive_tokenizer(
    midi: MidiFile,
    track_infos: List[TrackInfo],
    base_config: Optional[MidiParserConfig] = None
) -> Tuple[TokenizerManager, str]:
    """
    Create a tokenizer with adaptive configuration based on MIDI content.
    
    This implements the adaptive resolution feature from Section 19.
    
    Args:
        midi: MIDI file to analyze
        track_infos: Track analysis results
        base_config: Base configuration to adapt from
        
    Returns:
        Tuple of (TokenizerManager instance, recommended strategy)
    """
    config = base_config or DEFAULT_CONFIG
    
    total_notes = sum(t.statistics.total_notes for t in track_infos)
    note_density = total_notes / len(track_infos) if track_infos else 0
    max_polyphony = max((t.statistics.max_polyphony for t in track_infos), default=1)
    
    adapted_config = MidiParserConfig(
        tokenization=config.tokenization,
        tokenizer=TokenizerConfig(
            pitch_range=config.tokenizer.pitch_range,
            beat_resolution=config.tokenizer.beat_resolution,
            num_velocities=config.tokenizer.num_velocities,
            additional_tokens=dict(config.tokenizer.additional_tokens),
            max_seq_length=config.tokenizer.max_seq_length
        ),
        track_classification=config.track_classification,
        output=config.output,
        processing=config.processing,
        validation=config.validation
    )
    
    if note_density > 100:
        adapted_config.tokenizer.beat_resolution = 6
    elif note_density < 20:
        adapted_config.tokenizer.beat_resolution = 2
    
    velocity_range = max(
        t.statistics.avg_velocity for t in track_infos
    ) - min(
        t.statistics.avg_velocity for t in track_infos
    ) if track_infos else 0
    
    if velocity_range > 60:
        adapted_config.tokenizer.num_velocities = 32
    elif velocity_range < 20:
        adapted_config.tokenizer.num_velocities = 8
    
    if max_polyphony > 6:
        recommended_strategy = "Octuple"
    elif total_notes > 10000:
        recommended_strategy = "CPWord"
    elif note_density < 10:
        recommended_strategy = "TSD"
    else:
        recommended_strategy = "REMI"
    
    manager = TokenizerManager(adapted_config)
    
    logger.info(f"Created adaptive tokenizer: strategy={recommended_strategy}, "
               f"beat_res={adapted_config.tokenizer.beat_resolution}, "
               f"vel_bins={adapted_config.tokenizer.num_velocities}")
    
    return manager, recommended_strategy


def validate_token_sequence(
    tokens: List[int],
    vocabulary_size: int,
    strategy: str
) -> Tuple[bool, List[str]]:
    """
    Validate a token sequence for common issues.
    
    Args:
        tokens: Token sequence to validate
        vocabulary_size: Size of the vocabulary
        strategy: Tokenization strategy used
        
    Returns:
        Tuple of (is_valid, list of issues)
    """
    issues = []
    
    if not tokens:
        issues.append("Empty token sequence")
        return False, issues
    
    oov_tokens = [t for t in tokens if t >= vocabulary_size]
    if oov_tokens:
        issues.append(f"Found {len(oov_tokens)} out-of-vocabulary tokens")
    
    if len(tokens) > 10:
        for i in range(len(tokens) - 10):
            if len(set(tokens[i:i+10])) == 1:
                issues.append(f"Repetitive pattern detected at position {i}")
                break
    
    if strategy == "REMI":
        bar_token_likely = any(t < 10 for t in tokens[:20])
        if not bar_token_likely:
            issues.append("REMI sequence might be missing bar markers")
    
    elif strategy == "TSD":
        if len(set(tokens)) < 10:
            issues.append("TSD sequence has unusually low token diversity")
    
    is_valid = len(issues) == 0
    return is_valid, issues


def tokenize_midi(
    midi: MidiFile,
    config: Optional[MidiParserConfig] = None,
    track_infos: Optional[List[TrackInfo]] = None,
    strategy: Optional[str] = None,
    auto_select: bool = False
) -> TokenizationResult:
    """
    Main entry point for MIDI tokenization.
    """
    config = config or DEFAULT_CONFIG
    manager = TokenizerManager(config)
    
    return manager.tokenize_midi(
        midi,
        strategy=strategy,
        track_infos=track_infos,
        max_seq_length=config.tokenizer.max_seq_length,
        auto_select=auto_select
    )


__all__ = [
    'TokenizerManager',
    'TokenizationResult',
    'tokenize_midi',
    'create_adaptive_tokenizer',
    'validate_token_sequence',
    'MIDITOK_AVAILABLE',
    'MIDITOK_VERSION',
]