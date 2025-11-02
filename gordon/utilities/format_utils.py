"""
Format Utilities Module
=======================
Contains formatting and display utilities for trading data.
Handles number formatting, logging, and data presentation.
"""

import json
import logging
from datetime import datetime
from typing import Dict


class FormatUtils:
    """Utilities for formatting and display operations."""

    @staticmethod
    def format_number(value: float, decimals: int = 2) -> str:
        """
        Format number for display with K, M, B suffixes.

        Args:
            value: Number to format
            decimals: Decimal places

        Returns:
            Formatted string
        """
        if abs(value) >= 1e9:
            return f"{value/1e9:.{decimals}f}B"
        elif abs(value) >= 1e6:
            return f"{value/1e6:.{decimals}f}M"
        elif abs(value) >= 1e3:
            return f"{value/1e3:.{decimals}f}K"
        else:
            return f"{value:.{decimals}f}"

    @staticmethod
    def format_price(price: float, decimals: int = 2) -> str:
        """
        Format price with currency symbol.

        Args:
            price: Price value
            decimals: Decimal places

        Returns:
            Formatted price string
        """
        return f"${price:,.{decimals}f}"

    @staticmethod
    def format_percentage(value: float, decimals: int = 2, include_sign: bool = True) -> str:
        """
        Format percentage value.

        Args:
            value: Percentage value
            decimals: Decimal places
            include_sign: Whether to include +/- sign

        Returns:
            Formatted percentage string
        """
        sign = "+" if value > 0 and include_sign else ""
        return f"{sign}{value:.{decimals}f}%"

    @staticmethod
    def format_volume(volume: float) -> str:
        """
        Format volume with appropriate suffix.

        Args:
            volume: Volume value

        Returns:
            Formatted volume string
        """
        return FormatUtils.format_number(volume, decimals=2)

    @staticmethod
    def format_trade_info(trade_data: Dict) -> str:
        """
        Format trade information for display.

        Args:
            trade_data: Trade dictionary

        Returns:
            Formatted trade string
        """
        lines = []
        lines.append("=" * 50)
        lines.append("TRADE INFORMATION")
        lines.append("=" * 50)

        for key, value in trade_data.items():
            if isinstance(value, float):
                if 'price' in key.lower():
                    formatted_value = FormatUtils.format_price(value)
                elif 'percent' in key.lower() or 'rate' in key.lower():
                    formatted_value = FormatUtils.format_percentage(value)
                else:
                    formatted_value = FormatUtils.format_number(value)
            else:
                formatted_value = str(value)

            lines.append(f"{key.replace('_', ' ').title()}: {formatted_value}")

        lines.append("=" * 50)
        return "\n".join(lines)

    @staticmethod
    def format_orderbook_level(price: float, volume: float, total: float = None) -> str:
        """
        Format orderbook level for display.

        Args:
            price: Price level
            volume: Volume at level
            total: Cumulative total (optional)

        Returns:
            Formatted orderbook string
        """
        if total is not None:
            return f"{price:>12.2f} | {volume:>15.4f} | {total:>15.4f}"
        else:
            return f"{price:>12.2f} | {volume:>15.4f}"

    @staticmethod
    def format_pnl_summary(pnl_data: Dict) -> str:
        """
        Format PnL summary for display.

        Args:
            pnl_data: PnL dictionary

        Returns:
            Formatted PnL summary
        """
        lines = []
        lines.append("\n" + "=" * 50)
        lines.append("PnL SUMMARY")
        lines.append("=" * 50)

        gross_pnl = pnl_data.get('gross_pnl', 0)
        net_pnl = pnl_data.get('net_pnl', 0)
        fees = pnl_data.get('fees', 0)
        pnl_percent = pnl_data.get('pnl_percent', 0)

        lines.append(f"Gross PnL:     {FormatUtils.format_price(gross_pnl)}")
        lines.append(f"Fees:          {FormatUtils.format_price(fees)}")
        lines.append(f"Net PnL:       {FormatUtils.format_price(net_pnl)}")
        lines.append(f"Return:        {FormatUtils.format_percentage(pnl_percent)}")
        lines.append("=" * 50)

        return "\n".join(lines)

    @staticmethod
    def format_position_summary(position_data: Dict) -> str:
        """
        Format position summary for display.

        Args:
            position_data: Position dictionary

        Returns:
            Formatted position summary
        """
        lines = []
        lines.append("\n" + "=" * 50)
        lines.append("POSITION SUMMARY")
        lines.append("=" * 50)

        for key, value in position_data.items():
            formatted_key = key.replace('_', ' ').title()
            if isinstance(value, float):
                if 'price' in key.lower():
                    formatted_value = FormatUtils.format_price(value)
                elif any(x in key.lower() for x in ['percent', 'rate', 'pnl']):
                    formatted_value = FormatUtils.format_percentage(value)
                else:
                    formatted_value = FormatUtils.format_number(value, decimals=4)
            else:
                formatted_value = str(value)

            lines.append(f"{formatted_key}: {formatted_value}")

        lines.append("=" * 50)
        return "\n".join(lines)

    # ===========================================
    # LOGGING
    # ===========================================

    @staticmethod
    def log_trade(trade_data: Dict, filepath: str = "trades.json"):
        """
        Log trade to file.

        Args:
            trade_data: Trade information
            filepath: Path to log file
        """
        try:
            # Read existing trades
            try:
                with open(filepath, 'r') as f:
                    trades = json.load(f)
            except FileNotFoundError:
                trades = []

            # Add timestamp if not present
            if 'timestamp' not in trade_data:
                trade_data['timestamp'] = datetime.now().isoformat()

            # Append new trade
            trades.append(trade_data)

            # Write back
            with open(filepath, 'w') as f:
                json.dump(trades, f, indent=2)

        except Exception as e:
            logging.error(f"Failed to log trade: {e}")

    @staticmethod
    def log_error(error_message: str, context: Dict = None):
        """
        Log error with context.

        Args:
            error_message: Error message
            context: Additional context dictionary
        """
        log_data = {
            "timestamp": datetime.now().isoformat(),
            "error": error_message,
            "context": context or {}
        }

        logging.error(json.dumps(log_data, indent=2))

    @staticmethod
    def create_table_row(columns: list, widths: list = None) -> str:
        """
        Create formatted table row.

        Args:
            columns: List of column values
            widths: List of column widths (optional)

        Returns:
            Formatted row string
        """
        if widths is None:
            widths = [15] * len(columns)

        row_parts = []
        for col, width in zip(columns, widths):
            col_str = str(col)
            if isinstance(col, float):
                col_str = f"{col:.2f}"
            row_parts.append(col_str.ljust(width))

        return " | ".join(row_parts)

    @staticmethod
    def create_table(headers: list, rows: list, widths: list = None) -> str:
        """
        Create formatted table.

        Args:
            headers: List of header strings
            rows: List of row lists
            widths: List of column widths (optional)

        Returns:
            Formatted table string
        """
        lines = []

        # Create header
        header_line = FormatUtils.create_table_row(headers, widths)
        lines.append(header_line)
        lines.append("-" * len(header_line))

        # Create rows
        for row in rows:
            lines.append(FormatUtils.create_table_row(row, widths))

        return "\n".join(lines)


# Create singleton instance
format_utils = FormatUtils()
