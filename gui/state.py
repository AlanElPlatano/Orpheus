"""
Shared state management for Orpheus Gradio GUI.

This module provides global state that is shared across all tabs
in the Gradio interface.
"""

from typing import Optional, List, Dict, Any
from datetime import datetime

from midi_parser.interface import MidiParserGUI
from gui.config_presets import get_simple_mode_config, get_advanced_mode_config


class AppState:
    """
    Global application state shared across all tabs.

    Manages parser instances, configuration, logging, and training sessions.
    """

    def __init__(self):
        # Parser state
        self.parser: Optional[MidiParserGUI] = None
        self.current_mode = "simple"
        self.current_compression = True
        self.processing = False
        self.logs = []
        self.results = []

        # Training state
        self.trainer: Optional[Any] = None  # GradioTrainer instance
        self.training_logs: List[str] = []
        self.training_metrics_history: Dict[str, List[float]] = {
            'train_loss': [],
            'val_loss': [],
            'perplexity': [],
            'learning_rate': [],
            'steps': []
        }
        self.training_active = False
    
    def initialize_parser(self, mode: str, compress: bool) -> None:
        """
        Initialize parser with appropriate configuration.
        
        Args:
            mode: Processing mode ('simple' or 'advanced')
            compress: Whether to compress output files
        """
        config = (get_simple_mode_config(compress) if mode == "simple" 
                  else get_advanced_mode_config(compress))
        
        self.parser = MidiParserGUI(
            config=config,
            log_callback=self.log_handler
        )
        self.current_mode = mode
        self.current_compression = compress
    
    def log_handler(self, level: str, message: str) -> None:
        """
        Handle log messages from parser.
        
        Args:
            level: Log level (info, warning, error, debug)
            message: Log message
        """
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] [{level.upper()}] {message}"
        self.logs.append(log_entry)
    
    def clear_logs(self) -> None:
        """Clear the log buffer."""
        self.logs.clear()
    
    def get_recent_logs(self, count: int = 20) -> str:
        """
        Get recent log entries as a formatted string.

        Args:
            count: Number of recent entries to return

        Returns:
            Formatted log string
        """
        return "\n".join(self.logs[-count:])

    def initialize_trainer(self, trainer: Any) -> None:
        """
        Initialize or replace trainer instance.

        Args:
            trainer: GradioTrainer instance
        """
        self.trainer = trainer
        self.training_active = True
        self.training_logs.clear()
        self.training_metrics_history = {
            'train_loss': [],
            'val_loss': [],
            'perplexity': [],
            'learning_rate': [],
            'steps': []
        }

    def update_training_metrics(self, metrics: Any) -> None:
        """
        Update training metrics history.

        Args:
            metrics: TrainingMetrics instance
        """
        if metrics.step > 0:
            self.training_metrics_history['steps'].append(metrics.step)
            self.training_metrics_history['train_loss'].append(metrics.train_loss)
            if metrics.val_loss is not None:
                self.training_metrics_history['val_loss'].append(metrics.val_loss)
            self.training_metrics_history['perplexity'].append(metrics.perplexity)
            self.training_metrics_history['learning_rate'].append(metrics.learning_rate)

    def add_training_log(self, log_message: str) -> None:
        """
        Add a training log message.

        Args:
            log_message: Log message to add
        """
        self.training_logs.append(log_message)

    def get_recent_training_logs(self, count: int = 50) -> str:
        """
        Get recent training log entries.

        Args:
            count: Number of recent entries to return

        Returns:
            Formatted log string
        """
        return "\n".join(self.training_logs[-count:])

    def clear_training_state(self) -> None:
        """Clear training state."""
        self.trainer = None
        self.training_active = False
        self.training_logs.clear()
        self.training_metrics_history = {
            'train_loss': [],
            'val_loss': [],
            'perplexity': [],
            'learning_rate': [],
            'steps': []
        }


# Global state instance shared across the application
app_state = AppState()