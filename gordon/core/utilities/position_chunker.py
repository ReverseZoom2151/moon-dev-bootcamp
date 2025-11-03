"""
Position Chunking Utilities
============================
Day 25: Utilities for gradual position building and closing in chunks.

Features:
- Calculate chunk sizes for gradual position building
- Manage position sizes in chunks
- Close positions gradually
- Position target management
"""

import logging
from typing import Dict, Optional, Tuple
from decimal import Decimal, ROUND_DOWN

logger = logging.getLogger(__name__)


class PositionChunker:
    """
    Utilities for chunking positions during entry and exit.
    
    Helps manage large positions by breaking them into smaller chunks,
    reducing market impact and allowing for better execution.
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize position chunker.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.max_chunk_usd = self.config.get('max_chunk_usd', 1000)
        self.min_chunk_usd = self.config.get('min_chunk_usd', 100)
        self.chunk_tolerance = self.config.get('chunk_tolerance', 0.03)  # 3% tolerance

    def calculate_buy_chunk(
        self,
        target_usd_size: float,
        current_pos_usd: float,
        max_chunk_usd: Optional[float] = None
    ) -> float:
        """
        Calculate the size of the next buy chunk in USD.
        
        Args:
            target_usd_size: Target position size in USD
            current_pos_usd: Current position size in USD
            max_chunk_usd: Maximum chunk size (uses config if None)
            
        Returns:
            Chunk size in USD (0 if target reached)
        """
        max_chunk_usd = max_chunk_usd or self.max_chunk_usd
        size_needed_usd = target_usd_size - current_pos_usd
        
        if size_needed_usd <= 0:
            return 0.0  # Target reached
        
        chunk_usd = min(size_needed_usd, max_chunk_usd)
        
        # Ensure chunk meets minimum size
        if chunk_usd < self.min_chunk_usd:
            return 0.0  # Chunk too small, target likely reached
        
        return chunk_usd

    def calculate_sell_chunk(
        self,
        current_pos_usd: float,
        target_usd_size: float = 0.0,
        max_chunk_usd: Optional[float] = None
    ) -> float:
        """
        Calculate the size of the next sell chunk in USD.
        
        Args:
            current_pos_usd: Current position size in USD
            target_usd_size: Target position size (0 for full close)
            max_chunk_usd: Maximum chunk size (uses config if None)
            
        Returns:
            Chunk size in USD (0 if target reached)
        """
        max_chunk_usd = max_chunk_usd or self.max_chunk_usd
        size_to_close_usd = current_pos_usd - target_usd_size
        
        if size_to_close_usd <= 0:
            return 0.0  # Target reached
        
        chunk_usd = min(size_to_close_usd, max_chunk_usd)
        
        # Ensure chunk meets minimum size
        if chunk_usd < self.min_chunk_usd:
            return size_to_close_usd  # Close remaining if below minimum
        
        return chunk_usd

    def is_target_reached(
        self,
        current_pos_usd: float,
        target_usd_size: float,
        tolerance: Optional[float] = None
    ) -> bool:
        """
        Check if target position size has been reached.
        
        Args:
            current_pos_usd: Current position size in USD
            target_usd_size: Target position size in USD
            tolerance: Tolerance percentage (uses config if None)
            
        Returns:
            True if target reached within tolerance
        """
        tolerance = tolerance or self.chunk_tolerance
        
        if target_usd_size == 0:
            return current_pos_usd < self.min_chunk_usd
        
        ratio = current_pos_usd / target_usd_size if target_usd_size > 0 else 0
        return ratio >= (1 - tolerance)

    def calculate_chunk_size_in_base_currency(
        self,
        chunk_usd: float,
        price: float,
        decimals: int = 8
    ) -> float:
        """
        Convert chunk size from USD to base currency.
        
        Args:
            chunk_usd: Chunk size in USD
            price: Current price
            decimals: Number of decimals for rounding
            
        Returns:
            Chunk size in base currency
        """
        if price <= 0:
            return 0.0
        
        chunk_size = chunk_usd / price
        
        # Round down to specified decimals
        factor = 10 ** decimals
        chunk_size = int(chunk_size * factor) / factor
        
        return chunk_size

    def get_position_status(
        self,
        current_pos_usd: float,
        target_usd_size: float
    ) -> Dict[str, float]:
        """
        Get status of position relative to target.
        
        Args:
            current_pos_usd: Current position size in USD
            target_usd_size: Target position size in USD
            
        Returns:
            Dictionary with status information
        """
        size_needed = target_usd_size - current_pos_usd
        percent_complete = (current_pos_usd / target_usd_size * 100) if target_usd_size > 0 else 0
        target_reached = self.is_target_reached(current_pos_usd, target_usd_size)
        
        return {
            'current_usd': current_pos_usd,
            'target_usd': target_usd_size,
            'size_needed_usd': size_needed,
            'percent_complete': percent_complete,
            'target_reached': target_reached
        }

