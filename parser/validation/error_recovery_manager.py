"""
Error recovery manager for intelligent fallback strategy implementation.

This module provides intelligent error recovery and fallback strategies using
the ERROR_HANDLING_STRATEGIES from the configuration, with pattern recognition
and automatic configuration adjustment capabilities.
"""

import logging
from dataclasses import dataclass, field
import time
from typing import Dict, Any, Optional, List, Tuple
from enum import Enum
from pathlib import Path

from parser.config.defaults import (
    ERROR_HANDLING_STRATEGIES,
    MidiParserConfig,
    DEFAULT_CONFIG
)

logger = logging.getLogger(__name__)


class ErrorType(Enum):
    """Types of errors that can occur during processing."""
    CORRUPTED_MIDI = "corrupted_midi"
    EMPTY_TRACKS = "empty_tracks"
    EXTREME_DURATION = "extreme_duration"
    UNSUPPORTED_EVENTS = "unsupported_events"
    MEMORY_OVERFLOW = "memory_overflow"
    TOKENIZATION_FAILURE = "tokenization_failure"
    INVALID_METADATA = "invalid_metadata"
    UNKNOWN = "unknown"


class RecoveryStrategy(Enum):
    """Recovery strategies available."""
    SKIP = "skip"
    REMOVE = "remove"
    TRUNCATE = "truncate"
    FILTER = "filter"
    CHUNK = "chunk"
    FALLBACK = "fallback"
    DEFAULT = "default"
    RETRY = "retry"
    CONTINUE = "continue"


@dataclass
class ErrorPattern:
    """Pattern for recognizing specific error types."""
    error_type: ErrorType
    keywords: List[str]
    exception_types: List[type]
    confidence: float = 0.0
    
    def matches(self, error: Exception, message: str = "") -> Tuple[bool, float]:
        """Check if error matches this pattern."""
        confidence = 0.0
        
        # Check exception type
        if any(isinstance(error, exc_type) for exc_type in self.exception_types):
            confidence += 0.5
        
        # Check message keywords
        error_msg = (str(error) + " " + message).lower()
        keyword_matches = sum(1 for kw in self.keywords if kw.lower() in error_msg)
        if keyword_matches > 0:
            confidence += 0.5 * (keyword_matches / len(self.keywords))
        
        return confidence > 0.3, confidence


@dataclass
class RecoveryAction:
    """Action to take for recovery."""
    strategy: RecoveryStrategy
    config_adjustments: Dict[str, Any] = field(default_factory=dict)
    message: str = ""
    retry_allowed: bool = True
    max_retries: int = 3
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "strategy": self.strategy.value,
            "config_adjustments": self.config_adjustments,
            "message": self.message,
            "retry_allowed": self.retry_allowed,
            "max_retries": self.max_retries
        }


@dataclass
class RecoveryResult:
    """Result of error recovery attempt."""
    success: bool = False
    strategy_used: Optional[RecoveryStrategy] = None
    config_changes: Dict[str, Any] = field(default_factory=dict)
    message: str = ""
    should_retry: bool = False
    should_skip: bool = False
    recommendations: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "strategy_used": self.strategy_used.value if self.strategy_used else None,
            "config_changes": self.config_changes,
            "message": self.message,
            "should_retry": self.should_retry,
            "should_skip": self.should_skip,
            "recommendations": self.recommendations
        }


class ErrorRecoveryManager:
    """
    Manages intelligent error recovery and fallback strategies.
    
    This class recognizes error patterns and applies appropriate recovery
    strategies based on the ERROR_HANDLING_STRATEGIES configuration.
    """
    
    # Error pattern definitions
    ERROR_PATTERNS = [
        ErrorPattern(
            error_type=ErrorType.CORRUPTED_MIDI,
            keywords=["corrupt", "invalid", "malformed", "bad file", "cannot read"],
            exception_types=[IOError, ValueError],
        ),
        ErrorPattern(
            error_type=ErrorType.EMPTY_TRACKS,
            keywords=["empty", "no notes", "no events", "zero notes"],
            exception_types=[ValueError],
        ),
        ErrorPattern(
            error_type=ErrorType.EXTREME_DURATION,
            keywords=["duration", "too long", "timeout", "excessive"],
            exception_types=[TimeoutError, ValueError],
        ),
        ErrorPattern(
            error_type=ErrorType.UNSUPPORTED_EVENTS,
            keywords=["unsupported", "unknown event", "unrecognized"],
            exception_types=[NotImplementedError, ValueError],
        ),
        ErrorPattern(
            error_type=ErrorType.MEMORY_OVERFLOW,
            keywords=["memory", "overflow", "too large", "oom"],
            exception_types=[MemoryError],
        ),
        ErrorPattern(
            error_type=ErrorType.TOKENIZATION_FAILURE,
            keywords=["tokeniz", "token", "vocabulary", "encode"],
            exception_types=[ValueError, KeyError],
        ),
        ErrorPattern(
            error_type=ErrorType.INVALID_METADATA,
            keywords=["metadata", "missing", "invalid ppq", "tempo"],
            exception_types=[AttributeError, KeyError, ValueError],
        ),
    ]
    
    # Recovery strategy mappings
    RECOVERY_STRATEGIES = {
        ErrorType.CORRUPTED_MIDI: RecoveryAction(
            strategy=RecoveryStrategy.SKIP,
            message="Skipping corrupted MIDI file",
            retry_allowed=False
        ),
        ErrorType.EMPTY_TRACKS: RecoveryAction(
            strategy=RecoveryStrategy.REMOVE,
            config_adjustments={"min_notes_per_track": 0},
            message="Removing empty tracks and continuing",
            retry_allowed=True
        ),
        ErrorType.EXTREME_DURATION: RecoveryAction(
            strategy=RecoveryStrategy.TRUNCATE,
            config_adjustments={
                "max_duration_seconds": 300.0,
                "chunk_size_seconds": 30.0
            },
            message="Truncating excessive duration",
            retry_allowed=True
        ),
        ErrorType.UNSUPPORTED_EVENTS: RecoveryAction(
            strategy=RecoveryStrategy.FILTER,
            config_adjustments={"filter_unsupported": True},
            message="Filtering unsupported events",
            retry_allowed=True
        ),
        ErrorType.MEMORY_OVERFLOW: RecoveryAction(
            strategy=RecoveryStrategy.CHUNK,
            config_adjustments={
                "enable_chunking": True,
                "chunk_size_seconds": 15.0,
                "max_seq_length": 1024
            },
            message="Enabling chunked processing for large file",
            retry_allowed=True
        ),
        ErrorType.TOKENIZATION_FAILURE: RecoveryAction(
            strategy=RecoveryStrategy.FALLBACK,
            config_adjustments={"tokenization": "REMI"},
            message="Falling back to alternative tokenization strategy",
            retry_allowed=True,
            max_retries=2
        ),
        ErrorType.INVALID_METADATA: RecoveryAction(
            strategy=RecoveryStrategy.DEFAULT,
            config_adjustments={
                "use_default_ppq": True,
                "default_ppq": 480,
                "use_default_tempo": True,
                "default_tempo": 120
            },
            message="Using default metadata values",
            retry_allowed=True
        ),
    }
    
    # Fallback tokenization strategies
    TOKENIZATION_FALLBACK_SEQUENCE = [
        "REMI",
        "TSD", 
        "Structured",
        "MIDILike",
        "CPWord"
    ]
    
    def __init__(self, config: Optional[MidiParserConfig] = None):
        """
        Initialize the error recovery manager.
        
        Args:
            config: Parser configuration
        """
        self.config = config or DEFAULT_CONFIG
        self.error_history: List[Dict[str, Any]] = []
        self.recovery_attempts: Dict[str, int] = {}
        
    def handle_error(
        self,
        error: Exception,
        context: Optional[Dict[str, Any]] = None,
        retry_count: int = 0
    ) -> RecoveryResult:
        """
        Handle an error with intelligent recovery.
        
        Args:
            error: The exception that occurred
            context: Additional context about the error
            retry_count: Number of retries already attempted
            
        Returns:
            RecoveryResult with recovery action
        """
        logger.info(f"Handling error: {type(error).__name__}: {str(error)}")
        
        # Identify error type
        error_type = self._identify_error_type(error, context)
        
        # Record error in history
        self._record_error(error, error_type, context)
        
        # Get recovery strategy from configuration
        recovery_strategy = self._get_recovery_strategy(error_type)
        
        # Apply recovery action
        result = self._apply_recovery(
            error_type,
            recovery_strategy,
            error,
            context,
            retry_count
        )
        
        logger.info(f"Recovery result: {result.message}")
        
        return result
    
    def _identify_error_type(
        self,
        error: Exception,
        context: Optional[Dict[str, Any]] = None
    ) -> ErrorType:
        """
        Identify the type of error using pattern matching.
        
        Args:
            error: The exception
            context: Additional context
            
        Returns:
            Identified ErrorType
        """
        best_match = ErrorType.UNKNOWN
        best_confidence = 0.0
        
        context_msg = str(context) if context else ""
        
        for pattern in self.ERROR_PATTERNS:
            matches, confidence = pattern.matches(error, context_msg)
            if matches and confidence > best_confidence:
                best_match = pattern.error_type
                best_confidence = confidence
        
        logger.debug(f"Identified error type: {best_match.value} (confidence: {best_confidence:.2f})")
        
        return best_match
    
    def _get_recovery_strategy(self, error_type: ErrorType) -> str:
        """
        Get recovery strategy from configuration.
        
        Args:
            error_type: Type of error
            
        Returns:
            Recovery strategy string
        """
        # First check ERROR_HANDLING_STRATEGIES from config
        strategy = ERROR_HANDLING_STRATEGIES.get(error_type.value, "skip")
        
        # Override with custom recovery if defined
        if error_type in self.RECOVERY_STRATEGIES:
            recovery_action = self.RECOVERY_STRATEGIES[error_type]
            strategy = recovery_action.strategy.value
        
        return strategy
    
    def _apply_recovery(
        self,
        error_type: ErrorType,
        strategy: str,
        error: Exception,
        context: Optional[Dict[str, Any]],
        retry_count: int
    ) -> RecoveryResult:
        """
        Apply recovery action based on strategy.
        
        Args:
            error_type: Type of error
            strategy: Recovery strategy
            error: The original exception
            context: Error context
            retry_count: Number of retries
            
        Returns:
            RecoveryResult
        """
        result = RecoveryResult()
        
        # Get recovery action
        recovery_action = self.RECOVERY_STRATEGIES.get(
            error_type,
            RecoveryAction(strategy=RecoveryStrategy.SKIP)
        )
        
        # Check retry limit
        if retry_count >= recovery_action.max_retries:
            result.should_skip = True
            result.message = f"Max retries ({recovery_action.max_retries}) exceeded"
            return result
        
        # Apply strategy
        if strategy == "skip":
            result.should_skip = True
            result.message = recovery_action.message
            
        elif strategy == "retry":
            if recovery_action.retry_allowed and retry_count < recovery_action.max_retries:
                result.should_retry = True
                result.config_changes = recovery_action.config_adjustments
                result.message = f"Retrying with adjusted configuration (attempt {retry_count + 1})"
            else:
                result.should_skip = True
                result.message = "Retry not allowed or limit reached"
                
        elif strategy == "remove":
            result.config_changes = recovery_action.config_adjustments
            result.should_retry = True
            result.message = recovery_action.message
            
        elif strategy == "truncate":
            result.config_changes = recovery_action.config_adjustments
            result.should_retry = True
            result.message = recovery_action.message
            
        elif strategy == "filter":
            result.config_changes = recovery_action.config_adjustments
            result.should_retry = True
            result.message = recovery_action.message
            
        elif strategy == "chunk":
            result.config_changes = recovery_action.config_adjustments
            result.should_retry = True
            result.message = recovery_action.message
            
        elif strategy == "fallback":
            # Special handling for tokenization fallback
            if error_type == ErrorType.TOKENIZATION_FAILURE:
                result.config_changes = self._get_fallback_tokenization(context, retry_count)
                result.should_retry = True
                result.message = f"Trying fallback tokenization: {result.config_changes.get('tokenization')}"
            else:
                result.config_changes = recovery_action.config_adjustments
                result.should_retry = True
                result.message = recovery_action.message
                
        elif strategy == "default":
            result.config_changes = recovery_action.config_adjustments
            result.should_retry = True
            result.message = recovery_action.message
            
        else:
            # Unknown strategy - skip
            result.should_skip = True
            result.message = f"Unknown recovery strategy: {strategy}"
        
        # Add recommendations
        result.recommendations = self._generate_recommendations(error_type, error, context)
        
        # Set success if we have a recovery path
        result.success = result.should_retry or result.should_skip
        result.strategy_used = RecoveryStrategy(strategy) if result.success else None
        
        return result
    
    def _get_fallback_tokenization(
        self,
        context: Optional[Dict[str, Any]],
        retry_count: int
    ) -> Dict[str, Any]:
        """
        Get next fallback tokenization strategy.
        
        Args:
            context: Error context
            retry_count: Current retry count
            
        Returns:
            Configuration changes with new tokenization strategy
        """
        current_strategy = context.get("tokenization", "REMI") if context else "REMI"
        
        # Find current position in fallback sequence
        try:
            current_idx = self.TOKENIZATION_FALLBACK_SEQUENCE.index(current_strategy)
            next_idx = (current_idx + 1) % len(self.TOKENIZATION_FALLBACK_SEQUENCE)
        except ValueError:
            # Current strategy not in sequence, start from beginning
            next_idx = min(retry_count, len(self.TOKENIZATION_FALLBACK_SEQUENCE) - 1)
        
        next_strategy = self.TOKENIZATION_FALLBACK_SEQUENCE[next_idx]
        
        return {"tokenization": next_strategy}
    
    def _record_error(
        self,
        error: Exception,
        error_type: ErrorType,
        context: Optional[Dict[str, Any]]
    ) -> None:
        """Record error in history for pattern learning."""
        self.error_history.append({
            "error_type": error_type.value,
            "exception_type": type(error).__name__,
            "message": str(error),
            "context": context,
            "timestamp": time.time()
        })
        
        # Keep history size manageable
        if len(self.error_history) > 100:
            self.error_history = self.error_history[-100:]
    
    def _generate_recommendations(
        self,
        error_type: ErrorType,
        error: Exception,
        context: Optional[Dict[str, Any]]
    ) -> List[str]:
        """
        Generate recommendations based on error pattern.
        
        Args:
            error_type: Type of error
            error: The exception
            context: Error context
            
        Returns:
            List of recommendations
        """
        recommendations = []
        
        if error_type == ErrorType.CORRUPTED_MIDI:
            recommendations.append("Verify MIDI file integrity")
            recommendations.append("Try re-exporting from source DAW")
            
        elif error_type == ErrorType.MEMORY_OVERFLOW:
            recommendations.append("Reduce max sequence length")
            recommendations.append("Enable chunked processing")
            recommendations.append("Process in smaller batches")
            
        elif error_type == ErrorType.TOKENIZATION_FAILURE:
            recommendations.append("Try simpler tokenization strategy")
            recommendations.append("Check vocabulary configuration")
            recommendations.append("Reduce additional token features")
            
        elif error_type == ErrorType.EXTREME_DURATION:
            recommendations.append("Split long MIDI files before processing")
            recommendations.append("Adjust max_duration_seconds parameter")
            
        elif error_type == ErrorType.EMPTY_TRACKS:
            recommendations.append("Pre-filter empty tracks")
            recommendations.append("Check MIDI export settings")
            
        elif error_type == ErrorType.UNSUPPORTED_EVENTS:
            recommendations.append("Update to latest MidiTok version")
            recommendations.append("Filter non-standard MIDI events")
            
        elif error_type == ErrorType.INVALID_METADATA:
            recommendations.append("Verify MIDI file format")
            recommendations.append("Check for required metadata fields")
        
        # Add context-specific recommendations
        if context:
            if context.get("file_size_mb", 0) > 5:
                recommendations.append("Consider splitting large files")
            if context.get("track_count", 0) > 16:
                recommendations.append("Reduce number of tracks")
        
        return recommendations[:5]  # Return top 5 recommendations
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about error patterns.
        
        Returns:
            Dictionary with error statistics
        """
        stats = {
            "total_errors": len(self.error_history),
            "error_types": {},
            "recovery_success_rate": 0.0,
            "most_common_errors": [],
            "recommendations": []
        }
        
        # Count error types
        for entry in self.error_history:
            error_type = entry["error_type"]
            stats["error_types"][error_type] = stats["error_types"].get(error_type, 0) + 1
        
        # Find most common errors
        if stats["error_types"]:
            sorted_errors = sorted(
                stats["error_types"].items(),
                key=lambda x: x[1],
                reverse=True
            )
            stats["most_common_errors"] = [
                {"type": err, "count": count}
                for err, count in sorted_errors[:3]
            ]
        
        # Generate overall recommendations
        if stats["most_common_errors"]:
            most_common_type = ErrorType(stats["most_common_errors"][0]["type"])
            stats["recommendations"] = self._generate_recommendations(
                most_common_type, None, None
            )
        
        return stats
    
    def suggest_configuration_adjustments(
        self,
        error_history: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Suggest configuration adjustments based on error patterns.
        
        Args:
            error_history: Optional external error history
            
        Returns:
            Suggested configuration changes
        """
        history = error_history or self.error_history
        suggestions = {}
        
        if not history:
            return suggestions
        
        # Analyze patterns
        error_counts = {}
        for entry in history:
            error_type = entry.get("error_type", "unknown")
            error_counts[error_type] = error_counts.get(error_type, 0) + 1
        
        # Generate suggestions based on patterns
        if error_counts.get("memory_overflow", 0) > 2:
            suggestions["max_seq_length"] = 1024
            suggestions["chunk_size_seconds"] = 20.0
            suggestions["enable_chunking"] = True
        
        if error_counts.get("tokenization_failure", 0) > 3:
            suggestions["tokenization"] = "REMI"  # Most robust
            suggestions["simplify_vocabulary"] = True
        
        if error_counts.get("extreme_duration", 0) > 1:
            suggestions["max_duration_seconds"] = 300.0
            suggestions["auto_truncate"] = True
        
        if error_counts.get("corrupted_midi", 0) > 5:
            suggestions["strict_validation"] = False
            suggestions["skip_corrupted"] = True
        
        return suggestions
    
    def reset_error_history(self) -> None:
        """Clear error history and recovery attempts."""
        self.error_history.clear()
        self.recovery_attempts.clear()
        logger.info("Error history reset")


# Convenience function
def create_error_handler(config: Optional[MidiParserConfig] = None) -> ErrorRecoveryManager:
    """
    Create an error recovery manager instance.
    
    Args:
        config: Optional parser configuration
        
    Returns:
        Configured ErrorRecoveryManager
    """
    return ErrorRecoveryManager(config)


# Export main classes
__all__ = [
    'ErrorRecoveryManager',
    'ErrorType',
    'RecoveryStrategy',
    'RecoveryAction',
    'RecoveryResult',
    'create_error_handler'
]