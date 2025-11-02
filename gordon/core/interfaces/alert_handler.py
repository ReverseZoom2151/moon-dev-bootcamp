"""
Alert Handler Interface
========================
Abstract interface for sending alerts and notifications.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from enum import Enum


class AlertSeverity(Enum):
    """Alert severity levels."""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertChannel(Enum):
    """Alert delivery channels."""
    LOG = "log"
    EMAIL = "email"
    SMS = "sms"
    TELEGRAM = "telegram"
    DISCORD = "discord"
    WEBHOOK = "webhook"
    CONSOLE = "console"


class AlertHandlerInterface(ABC):
    """
    Abstract interface for handling alerts and notifications.
    Supports multiple channels and severity levels.
    """

    @abstractmethod
    async def send_alert(self, message: str, severity: AlertSeverity,
                        channels: Optional[List[AlertChannel]] = None,
                        metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Send an alert message.

        Args:
            message: Alert message
            severity: Alert severity level
            channels: List of channels to send to (None for default)
            metadata: Additional metadata for the alert

        Returns:
            Success status
        """
        pass

    @abstractmethod
    async def send_trade_alert(self, trade_info: Dict[str, Any]) -> bool:
        """
        Send a trade execution alert.

        Args:
            trade_info: Trade information including:
                - symbol: Trading symbol
                - action: BUY/SELL
                - size: Position size
                - price: Execution price
                - pnl: Profit/loss (if closing)
                - timestamp: Execution time

        Returns:
            Success status
        """
        pass

    @abstractmethod
    async def send_risk_alert(self, risk_info: Dict[str, Any]) -> bool:
        """
        Send a risk management alert.

        Args:
            risk_info: Risk information including:
                - type: Risk type (drawdown, exposure, etc.)
                - level: Risk level
                - action_taken: Any automatic action taken
                - recommendation: Recommended action

        Returns:
            Success status
        """
        pass

    @abstractmethod
    async def send_system_alert(self, system_info: Dict[str, Any]) -> bool:
        """
        Send a system status alert.

        Args:
            system_info: System information including:
                - component: System component
                - status: Current status
                - error: Error message (if any)
                - action: Action required

        Returns:
            Success status
        """
        pass

    @abstractmethod
    def configure_channel(self, channel: AlertChannel,
                         config: Dict[str, Any]) -> bool:
        """
        Configure an alert channel.

        Args:
            channel: Channel to configure
            config: Channel-specific configuration

        Returns:
            Success status
        """
        pass

    @abstractmethod
    def set_severity_filter(self, min_severity: AlertSeverity):
        """
        Set minimum severity level for alerts.

        Args:
            min_severity: Minimum severity to send
        """
        pass

    @abstractmethod
    def enable_channel(self, channel: AlertChannel):
        """Enable an alert channel."""
        pass

    @abstractmethod
    def disable_channel(self, channel: AlertChannel):
        """Disable an alert channel."""
        pass

    @abstractmethod
    def get_alert_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get recent alert history.

        Args:
            limit: Number of alerts to retrieve

        Returns:
            List of recent alerts
        """
        pass