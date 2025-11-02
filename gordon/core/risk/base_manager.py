"""
Base Risk Manager
=================
Base classes and common utilities for risk management.
"""

import logging
import json
import os
from typing import Dict, Any, Optional


class BaseRiskManager:
    """
    Base class for all risk management components.

    Provides common functionality like:
    - Configuration management
    - Settings persistence
    - Demo mode support
    - Event bus integration
    - Exchange connection management
    """

    def __init__(self, event_bus: Any, config_manager: Any, demo_mode: bool = False):
        """
        Initialize base risk manager.

        Args:
            event_bus: Event bus for communication
            config_manager: Configuration manager
            demo_mode: Whether to run in demo mode
        """
        self.event_bus = event_bus
        self.config_manager = config_manager
        self.logger = logging.getLogger(self.__class__.__name__)
        self.demo_mode = demo_mode

        if demo_mode:
            self.logger.info(f"⚠️ {self.__class__.__name__} running in DEMO MODE")

        # Load risk configuration
        self.risk_config = config_manager.get_risk_config()

        # Exchange connections for direct queries
        self.exchange_connections = {}

    def set_exchange_connection(self, exchange: str, connection: Any):
        """
        Set exchange connection for direct queries.

        Args:
            exchange: Exchange name
            connection: Exchange connection object
        """
        self.exchange_connections[exchange] = connection
        self.logger.info(f"Exchange connection set for {exchange}")

    def save_settings(self, settings: Dict, filename: str):
        """
        Save settings to JSON file.

        Args:
            settings: Settings dictionary
            filename: File to save to
        """
        try:
            with open(filename, 'w') as f:
                json.dump(settings, f, indent=4)
            self.logger.info(f"Settings saved to {filename}")
        except Exception as e:
            self.logger.error(f"Error saving settings to {filename}: {e}")

    def load_settings(self, filename: str) -> Dict:
        """
        Load settings from JSON file.

        Args:
            filename: File to load from

        Returns:
            Settings dictionary or empty dict if file doesn't exist
        """
        try:
            if os.path.exists(filename):
                with open(filename, 'r') as f:
                    settings = json.load(f)
                self.logger.info(f"Settings loaded from {filename}")
                return settings
            return {}
        except Exception as e:
            self.logger.error(f"Error loading settings from {filename}: {e}")
            return {}

    async def emit_event(self, event_name: str, data: Dict):
        """
        Emit an event through the event bus.

        Args:
            event_name: Name of the event
            data: Event data
        """
        try:
            await self.event_bus.emit(event_name, data)
        except Exception as e:
            self.logger.error(f"Error emitting event {event_name}: {e}")
