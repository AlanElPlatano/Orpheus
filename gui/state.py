"""
Shared state management for Orpheus Gradio GUI.

This module provides global state that is shared across all tabs
in the Gradio interface.
"""

from typing import Optional, List
from datetime import datetime

from midi_parser.interface import MidiParserGUI
from gui.config_presets import get_simple_mode_config, get_advanced_mode_config


class AppState:
    """
    Global application state shared across all tabs.
    
    Manages parser instances, configuration, and logging.
    """
    
    def __init__(self):
        self.parser: Optional[MidiParserGUI] = None
        self.current_mode = "simple"
        self.current_compression = True
        self.processing = False
        self.logs = []
        self.results = []
    
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


# Global state instance shared across the application
app_state = AppState()