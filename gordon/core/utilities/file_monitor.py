"""
Quick Buy/Sell File Monitor (Day 51)
=====================================
Monitors a text file for token symbols and triggers rapid buy/sell execution.
"""

import os
import time
import logging
from pathlib import Path
from typing import Set, Optional, Callable, Dict, Any
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler


class TokenFileHandler(FileSystemEventHandler):
    """
    File system event handler for monitoring token address file.
    
    Monitors a text file for token symbols and triggers callbacks when new entries are added.
    Format:
    - Token symbol only = BUY command
    - Token symbol + 'x' or 'c' = SELL command
    """
    
    def __init__(self, file_path: str, on_token_added: Callable[[str, Optional[str]], None]):
        """
        Initialize file handler.
        
        Args:
            file_path: Path to file to monitor
            on_token_added: Callback function(token_symbol, command)
        """
        self.file_path = Path(file_path)
        self.on_token_added = on_token_added
        self.processed_lines: Set[str] = set()
        self.logger = logging.getLogger(__name__)
        
        # Ensure file exists
        self._ensure_file_exists()
        
        # Load existing lines
        self._load_existing_lines()
    
    def _ensure_file_exists(self):
        """Ensure the token file exists."""
        if not self.file_path.exists():
            self.file_path.parent.mkdir(parents=True, exist_ok=True)
            self.file_path.touch()
            self.logger.info(f"Created token file: {self.file_path}")
    
    def _load_existing_lines(self):
        """Load existing lines from file to avoid reprocessing."""
        try:
            if self.file_path.exists():
                with open(self.file_path, 'r') as f:
                    for line in f:
                        cleaned = line.strip()
                        if cleaned:
                            self.processed_lines.add(cleaned)
                self.logger.info(f"Loaded {len(self.processed_lines)} existing lines")
        except Exception as e:
            self.logger.error(f"Error loading existing lines: {e}")
    
    def on_modified(self, event):
        """Handle file modification events."""
        if event.is_directory:
            return
        
        if Path(event.src_path) == self.file_path:
            self._process_file()
    
    def _process_file(self):
        """Process the token file for new entries."""
        try:
            if not self.file_path.exists():
                return
            
            with open(self.file_path, 'r') as f:
                lines = f.readlines()
            
            for line in lines:
                line = line.strip()
                
                # Skip empty lines and comments
                if not line or line.startswith('#'):
                    continue
                
                # Skip already processed lines
                if line in self.processed_lines:
                    continue
                
                # Mark as processed
                self.processed_lines.add(line)
                
                # Parse command
                parts = line.split()
                if not parts:
                    continue
                
                token_symbol = parts[0]
                command = None
                
                # Check for sell command ('x' or 'c')
                if len(parts) > 1:
                    cmd_char = parts[1].lower()
                    if cmd_char in ['x', 'c']:
                        command = 'SELL'
                
                # If no command, default to BUY
                if not command:
                    command = 'BUY'
                
                # Trigger callback
                try:
                    self.on_token_added(token_symbol, command)
                except Exception as e:
                    self.logger.error(f"Error processing token {token_symbol}: {e}")
        
        except Exception as e:
            self.logger.error(f"Error processing file: {e}")


class QuickBuySellMonitor:
    """
    Monitor for quick buy/sell token file.
    
    Monitors a text file and triggers rapid execution when tokens are added.
    """
    
    def __init__(self, file_path: str, on_token_added: Callable[[str, Optional[str]], None],
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize monitor.
        
        Args:
            file_path: Path to token file to monitor
            on_token_added: Callback function(token_symbol, command)
            config: Configuration dictionary
        """
        self.file_path = file_path
        self.on_token_added = on_token_added
        self.config = config or {}
        self.observer: Optional[Observer] = None
        self.event_handler: Optional[TokenFileHandler] = None
        self.logger = logging.getLogger(__name__)
        self.is_running = False
    
    def start(self):
        """Start monitoring the file."""
        if self.is_running:
            self.logger.warning("Monitor is already running")
            return
        
        try:
            self.event_handler = TokenFileHandler(self.file_path, self.on_token_added)
            self.observer = Observer()
            self.observer.schedule(
                self.event_handler,
                path=str(Path(self.file_path).parent),
                recursive=False
            )
            self.observer.start()
            self.is_running = True
            self.logger.info(f"Started monitoring file: {self.file_path}")
        except Exception as e:
            self.logger.error(f"Failed to start monitor: {e}")
            raise
    
    def stop(self):
        """Stop monitoring the file."""
        if not self.is_running:
            return
        
        if self.observer:
            self.observer.stop()
            self.observer.join(timeout=5)
        
        self.is_running = False
        self.logger.info("Stopped monitoring file")
    
    def check_now(self):
        """Manually check the file for new entries."""
        if self.event_handler:
            self.event_handler._process_file()
    
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()


def add_token_to_file(file_path: str, token_symbol: str, command: Optional[str] = None):
    """
    Add a token symbol to the monitoring file.
    
    Args:
        file_path: Path to token file
        token_symbol: Token symbol to add
        command: Optional command ('BUY' or 'SELL'). Defaults to 'BUY'
    """
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    line = token_symbol
    if command and command.upper() == 'SELL':
        line = f"{token_symbol} x"
    
    try:
        with open(path, 'a') as f:
            f.write(f"{line}\n")
    except Exception as e:
        raise IOError(f"Failed to write to file {file_path}: {e}")

