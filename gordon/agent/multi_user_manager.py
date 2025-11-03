"""
Multi-User Conversation Manager
=================================
Day 30 Enhancement: Support multiple users with separate conversation memories.

Allows multiple users to have isolated conversation histories.
"""

import logging
from typing import Dict, Optional
from pathlib import Path

from .conversation_memory import ConversationMemory, ExchangeConversationMemory

logger = logging.getLogger(__name__)


class MultiUserConversationManager:
    """
    Manages conversations for multiple users.
    
    Each user gets their own conversation memory directory.
    """
    
    def __init__(
        self,
        base_memory_dir: str = './conversation_memory',
        default_user: str = 'default'
    ):
        """
        Initialize multi-user conversation manager.
        
        Args:
            base_memory_dir: Base directory for user memories
            default_user: Default user ID
        """
        self.base_memory_dir = Path(base_memory_dir)
        self.default_user = default_user
        self.current_user = default_user
        
        # User-specific memories
        self.user_memories: Dict[str, ExchangeConversationMemory] = {}
        
        logger.info(f"Multi-user conversation manager initialized (default: {default_user})")
    
    def get_user_memory_dir(self, user_id: str) -> Path:
        """Get memory directory for a user."""
        return self.base_memory_dir / f"user_{user_id}"
    
    def switch_user(self, user_id: str):
        """
        Switch to a different user.
        
        Args:
            user_id: User identifier
        """
        self.current_user = user_id
        logger.info(f"Switched to user: {user_id}")
    
    def get_memory(
        self,
        exchange_name: str = "binance",
        symbol: str = "BTCUSDT",
        user_id: Optional[str] = None
    ) -> ExchangeConversationMemory:
        """
        Get or create memory for a user.
        
        Args:
            exchange_name: Exchange name
            symbol: Trading symbol
            user_id: User ID (uses current user if None)
            
        Returns:
            Exchange conversation memory instance
        """
        user_id = user_id or self.current_user
        
        # Create cache key
        cache_key = f"{user_id}_{exchange_name}_{symbol}"
        
        # Return cached memory if exists
        if cache_key in self.user_memories:
            return self.user_memories[cache_key]
        
        # Create new memory
        user_memory_dir = self.get_user_memory_dir(user_id)
        memory = ExchangeConversationMemory(
            exchange_name=exchange_name,
            symbol=symbol,
            memory_dir=str(user_memory_dir),
            max_memory_tokens=6000
        )
        
        # Cache it
        self.user_memories[cache_key] = memory
        
        return memory
    
    def list_users(self) -> list:
        """
        List all users with conversation memories.
        
        Returns:
            List of user IDs
        """
        users = []
        
        if not self.base_memory_dir.exists():
            return users
        
        for item in self.base_memory_dir.iterdir():
            if item.is_dir() and item.name.startswith('user_'):
                user_id = item.name.replace('user_', '')
                users.append(user_id)
        
        return sorted(users)
    
    def get_user_stats(self, user_id: str) -> Dict:
        """
        Get statistics for a user's conversations.
        
        Args:
            user_id: User identifier
            
        Returns:
            Dictionary with user statistics
        """
        user_dir = self.get_user_memory_dir(user_id)
        
        if not user_dir.exists():
            return {
                'user_id': user_id,
                'total_conversations': 0,
                'total_size_bytes': 0,
                'files': []
            }
        
        files = list(user_dir.glob('*.txt'))
        total_size = sum(f.stat().st_size for f in files)
        
        return {
            'user_id': user_id,
            'total_conversations': len(files),
            'total_size_bytes': total_size,
            'total_size_mb': round(total_size / (1024 * 1024), 2),
            'files': [f.name for f in files]
        }
    
    def delete_user_memory(self, user_id: str, exchange_name: Optional[str] = None):
        """
        Delete conversation memory for a user.
        
        Args:
            user_id: User identifier
            exchange_name: Specific exchange to delete (None = all)
        """
        user_dir = self.get_user_memory_dir(user_id)
        
        if not user_dir.exists():
            logger.warning(f"User directory does not exist: {user_dir}")
            return
        
        if exchange_name:
            # Delete specific exchange files
            pattern = f"{exchange_name}_*.txt"
            for file in user_dir.glob(pattern):
                file.unlink()
                logger.info(f"Deleted {file}")
        else:
            # Delete all user files
            for file in user_dir.glob('*.txt'):
                file.unlink()
                logger.info(f"Deleted {file}")
            
            # Try to remove directory if empty
            try:
                user_dir.rmdir()
            except OSError:
                pass  # Directory not empty
        
        # Clear cache
        cache_keys_to_remove = [
            key for key in self.user_memories.keys()
            if key.startswith(f"{user_id}_")
        ]
        for key in cache_keys_to_remove:
            del self.user_memories[key]
        
        logger.info(f"Deleted memory for user {user_id}")

