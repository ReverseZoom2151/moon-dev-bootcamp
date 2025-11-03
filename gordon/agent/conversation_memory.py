"""
Conversation Memory Manager
===========================
Day 30: Persistent conversation memory for Gordon's conversational interface.

Manages conversation history storage and retrieval with token limits
and automatic trimming to maintain context while preventing overflow.
"""

import os
import logging
from typing import Optional, Dict
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


class ConversationMemory:
    """
    Manages persistent conversation memory for Gordon's conversational interface.
    
    Features:
    - Saves conversation history to files
    - Automatic memory trimming to prevent token overflow
    - Exchange-specific memory files
    - Timestamp tracking
    """

    def __init__(
        self,
        memory_file: str = "gordon_memory.txt",
        memory_dir: str = "./conversation_memory",
        max_memory_tokens: int = 6000,
        start_delimiter: str = "#### START GORDON MEMORY ####",
        end_delimiter: str = "#### END GORDON MEMORY ####"
    ):
        """
        Initialize conversation memory manager.
        
        Args:
            memory_file: Name of memory file
            memory_dir: Directory to store memory files
            max_memory_tokens: Maximum tokens in memory (approximate)
            start_delimiter: Delimiter marking start of memory
            end_delimiter: Delimiter marking end of memory
        """
        self.memory_dir = Path(memory_dir)
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        
        self.memory_file = memory_file
        self.memory_path = self.memory_dir / memory_file
        self.max_memory_tokens = max_memory_tokens
        self.start_delimiter = start_delimiter
        self.end_delimiter = end_delimiter
        
    def read_memory(self, initial_context: Optional[str] = None) -> str:
        """
        Read conversation memory from file.
        
        Args:
            initial_context: Initial context if memory file doesn't exist
            
        Returns:
            Memory string with delimiters
        """
        try:
            if self.memory_path.exists():
                content = self.memory_path.read_text(encoding="utf-8").strip()
                if content:
                    return content
                    
            # Return initial context if file doesn't exist
            if initial_context:
                return self._create_initial_memory(initial_context)
            return ""
            
        except Exception as e:
            logger.error(f"Error reading memory file: {e}")
            return initial_context or ""
    
    def _create_initial_memory(self, context: str) -> str:
        """Create initial memory with context."""
        return f"""{self.start_delimiter}
{context}

Session Started: {datetime.now().isoformat()}
{self.end_delimiter}"""
    
    def update_memory(
        self,
        user_prompt: str,
        ai_response: str,
        current_memory: str,
        include_timestamp: bool = True
    ) -> str:
        """
        Update conversation memory with new interaction.
        
        Args:
            user_prompt: User's message
            ai_response: AI's response
            current_memory: Current memory content
            include_timestamp: Whether to include timestamps
            
        Returns:
            Updated memory string
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        if include_timestamp:
            memory_update = f"\n\n[{timestamp}] User: {user_prompt}\n[{timestamp}] Gordon: {ai_response}"
        else:
            memory_update = f"\n\nUser: {user_prompt}\nGordon: {ai_response}"
        
        new_memory = current_memory + memory_update
        
        # Trim memory if too long
        new_memory = self._trim_memory(new_memory)
        
        # Save to file
        try:
            self.memory_path.write_text(new_memory.strip(), encoding="utf-8")
            logger.debug(f"Memory updated in {self.memory_path}")
        except Exception as e:
            logger.error(f"Error saving memory: {e}")
        
        return new_memory
    
    def _trim_memory(self, memory: str) -> str:
        """
        Trim memory if it exceeds token limit.
        
        Keeps initial context and recent conversation.
        """
        # Rough token estimate: ~1 token per word
        word_count = len(memory.split())
        
        if word_count <= self.max_memory_tokens:
            return memory
        
        logger.info(f"Trimming memory (current: {word_count} words, max: {self.max_memory_tokens})")
        
        # Split into lines
        lines = memory.splitlines()
        
        # Keep header lines (delimiters, context)
        header_lines = []
        recent_lines = []
        
        in_header = True
        for line in lines:
            if self.start_delimiter in line or self.end_delimiter in line:
                header_lines.append(line)
                in_header = True
            elif in_header and ("=" in line or "CONTEXT" in line or "Session Started" in line):
                header_lines.append(line)
            else:
                in_header = False
                recent_lines.append(line)
        
        # Keep last ~150 lines of conversation
        recent_lines = recent_lines[-150:]
        
        # Reconstruct memory
        trimmed = "\n".join(header_lines + ["...\n"] + recent_lines)
        
        return trimmed
    
    def clear_memory(self):
        """Clear conversation memory."""
        try:
            if self.memory_path.exists():
                self.memory_path.unlink()
                logger.info(f"Memory cleared: {self.memory_path}")
        except Exception as e:
            logger.error(f"Error clearing memory: {e}")
    
    def get_memory_path(self) -> Path:
        """Get path to memory file."""
        return self.memory_path


class ExchangeConversationMemory(ConversationMemory):
    """
    Exchange-specific conversation memory.
    
    Creates separate memory files for different exchanges.
    """
    
    def __init__(
        self,
        exchange_name: str,
        symbol: str = "BTCUSDT",
        memory_dir: str = "./conversation_memory",
        max_memory_tokens: int = 6000
    ):
        """
        Initialize exchange-specific memory.
        
        Args:
            exchange_name: Name of exchange (e.g., 'binance', 'bitfinex')
            symbol: Primary trading symbol
            memory_dir: Directory to store memory files
            max_memory_tokens: Maximum tokens in memory
        """
        memory_file = f"{exchange_name}_trading_memory_{symbol.lower()}.txt"
        
        start_delimiter = f"#### START {exchange_name.upper()} TRADING MEMORY ####"
        end_delimiter = f"#### END {exchange_name.upper()} TRADING MEMORY ####"
        
        super().__init__(
            memory_file=memory_file,
            memory_dir=memory_dir,
            max_memory_tokens=max_memory_tokens,
            start_delimiter=start_delimiter,
            end_delimiter=end_delimiter
        )
        
        self.exchange_name = exchange_name
        self.symbol = symbol

