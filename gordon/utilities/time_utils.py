"""
Time Utilities Module
=====================
Contains time and date related utilities for trading operations.
Handles timestamps, date conversions, and time-based calculations.
"""

from datetime import datetime, timedelta
from typing import Optional


class TimeUtils:
    """Utilities for time and date operations."""

    @staticmethod
    def get_current_timestamp() -> str:
        """
        Get current timestamp in ISO format.

        Returns:
            Current timestamp as ISO string
        """
        return datetime.now().isoformat()

    @staticmethod
    def get_unix_timestamp(dt: Optional[datetime] = None) -> int:
        """
        Get Unix timestamp.

        Args:
            dt: Datetime object (default: current time)

        Returns:
            Unix timestamp in seconds
        """
        if dt is None:
            dt = datetime.now()
        return int(dt.timestamp())

    @staticmethod
    def unix_to_datetime(timestamp: int) -> datetime:
        """
        Convert Unix timestamp to datetime.

        Args:
            timestamp: Unix timestamp in seconds

        Returns:
            Datetime object
        """
        return datetime.fromtimestamp(timestamp)

    @staticmethod
    def get_time_ago(hours: int = 0, days: int = 0, minutes: int = 0) -> datetime:
        """
        Get datetime for time in the past.

        Args:
            hours: Hours ago
            days: Days ago
            minutes: Minutes ago

        Returns:
            Datetime object
        """
        return datetime.now() - timedelta(hours=hours, days=days, minutes=minutes)

    @staticmethod
    def format_datetime(dt: datetime, fmt: str = "%Y-%m-%d %H:%M:%S") -> str:
        """
        Format datetime to string.

        Args:
            dt: Datetime object
            fmt: Format string

        Returns:
            Formatted datetime string
        """
        return dt.strftime(fmt)

    @staticmethod
    def parse_datetime(date_string: str, fmt: str = "%Y-%m-%d %H:%M:%S") -> datetime:
        """
        Parse string to datetime.

        Args:
            date_string: Date string
            fmt: Format string

        Returns:
            Datetime object
        """
        return datetime.strptime(date_string, fmt)

    @staticmethod
    def is_market_hours(dt: Optional[datetime] = None,
                       start_hour: int = 9,
                       end_hour: int = 17) -> bool:
        """
        Check if current time is within market hours.

        Args:
            dt: Datetime to check (default: current time)
            start_hour: Market open hour (24h format)
            end_hour: Market close hour (24h format)

        Returns:
            True if within market hours
        """
        if dt is None:
            dt = datetime.now()
        return start_hour <= dt.hour < end_hour

    @staticmethod
    def get_weekday(dt: Optional[datetime] = None) -> str:
        """
        Get weekday name.

        Args:
            dt: Datetime object (default: current time)

        Returns:
            Weekday name (e.g., 'Monday')
        """
        if dt is None:
            dt = datetime.now()
        return dt.strftime("%A")

    @staticmethod
    def is_weekend(dt: Optional[datetime] = None) -> bool:
        """
        Check if date is a weekend.

        Args:
            dt: Datetime object (default: current time)

        Returns:
            True if weekend (Saturday or Sunday)
        """
        if dt is None:
            dt = datetime.now()
        return dt.weekday() >= 5  # 5=Saturday, 6=Sunday


# Create singleton instance
time_utils = TimeUtils()
