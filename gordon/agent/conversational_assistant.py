"""
Conversational Trading Assistant
=================================
Day 30: Conversational interface wrapper around Gordon's agent.

Provides a MoonGPT-style conversational interface with:
- Persistent conversation memory
- Real-time market context
- Quick command shortcuts
- Exchange-specific assistants
"""

import logging
from typing import Dict, Optional, Any
from datetime import datetime

from .conversation_memory import ExchangeConversationMemory
from .market_context import MarketContextProvider, QuickCommandHandler
from .market_stream_manager import MarketStreamManager
from ..utilities.ui import Logger

logger = logging.getLogger(__name__)


class ConversationalAssistant:
    """
    Conversational trading assistant wrapper around Gordon's agent.
    
    Provides MoonGPT-style conversational interface with memory and context.
    """

    def __init__(
        self,
        gordon_agent: Any,
        exchange_name: str = "binance",
        symbol: str = "BTCUSDT",
        config: Optional[Dict] = None
    ):
        """
        Initialize conversational assistant.
        
        Args:
            gordon_agent: GordonAgent instance
            exchange_name: Exchange name for context
            symbol: Primary trading symbol
            config: Configuration dictionary
        """
        self.gordon_agent = gordon_agent
        self.exchange_name = exchange_name
        self.symbol = symbol
        self.config = config or {}
        
        # Initialize memory
        memory_config = self.config.get('conversation', {})
        self.memory = ExchangeConversationMemory(
            exchange_name=exchange_name,
            symbol=symbol,
            memory_dir=memory_config.get('memory_dir', './conversation_memory'),
            max_memory_tokens=memory_config.get('max_memory_tokens', 6000)
        )
        
        # Initialize market context provider
        exchange_adapter = gordon_agent.exchanges.get(exchange_name)
        default_symbols = memory_config.get('default_symbols', ["BTCUSDT", "ETHUSDT", "BNBUSDT"])
        
        self.market_context = MarketContextProvider(
            exchange_adapter=exchange_adapter,
            default_symbols=default_symbols
        )
        
        # Initialize market stream manager (optional)
        self.market_stream = MarketStreamManager(
            exchange_adapter=exchange_adapter,
            symbols=default_symbols
        )
        
        # Initialize quick command handler
        self.command_handler = QuickCommandHandler(self.market_context)
        
        # Logger
        self.logger = Logger()
        
    async def chat(self, user_prompt: str) -> str:
        """
        Process user message and return response.
        
        Args:
            user_prompt: User's message
            
        Returns:
            Assistant's response
        """
        # Handle quick commands
        processed_prompt = await self.command_handler.handle_command(user_prompt)
        
        # Load current memory
        current_memory = self.memory.read_memory(
            initial_context=self._create_initial_context()
        )
        
        # Build prompt with memory and market context
        full_prompt = self._build_conversation_prompt(
            memory=current_memory,
            user_prompt=processed_prompt
        )
        
        # Get market context (prefer stream if available)
        if self.market_stream.is_streaming:
            market_summary = self.market_stream.get_market_summary_from_stream()
        else:
            market_summary = await self.market_context.get_market_summary()
        
        # Add market context to prompt
        if market_summary:
            full_prompt = f"{market_summary}\n\n{full_prompt}"
        
        # Get response from Gordon agent
        try:
            # Use Gordon's general assistance for conversational queries
            # This uses the LLM directly for conversational responses
            ai_response = await self.gordon_agent.general_assistance(processed_prompt)
            
            # Update memory
            self.memory.update_memory(
                user_prompt=processed_prompt,
                ai_response=ai_response,
                current_memory=current_memory
            )
            
            return ai_response
            
        except Exception as e:
            logger.error(f"Error in conversational assistant: {e}")
            return f"I encountered an error: {str(e)}. Please try again."
    
    def _create_initial_context(self) -> str:
        """Create initial context for new conversations."""
        context = [
            f"GORDON TRADING ASSISTANT CONTEXT - {self.exchange_name.upper()}",
            "="*50,
            f"Exchange: {self.exchange_name.capitalize()}",
            f"Primary Symbol: {self.symbol}",
            f"Focus: Trading analysis, market data, strategy discussion",
            f"Session Started: {datetime.now().isoformat()}",
            "",
            "Available Commands:",
            "- 'price SYMBOL': Get current price",
            "- 'analyze SYMBOL': Get detailed analysis",
            "- 'market': Get market summary",
            "- Ask about trading strategies and technical analysis",
            "- Get real-time market data and insights",
            "- Trading education and risk management advice",
            "",
            "Ready to assist with your trading needs!"
        ]
        return "\n".join(context)
    
    def _build_conversation_prompt(self, memory: str, user_prompt: str) -> str:
        """Build full prompt with memory context."""
        system_prompt = f"""You are Gordon, a specialized {self.exchange_name} trading assistant and market analyst. You have access to real-time market data and help users with:

🎯 CORE CAPABILITIES:
- Real-time price analysis and market insights
- Technical analysis and chart pattern recognition  
- Trading strategy development and optimization
- Risk management and position sizing guidance
- Market sentiment analysis and trend identification
- Educational content about trading concepts

📊 MARKET DATA ACCESS:
- Live prices, 24h changes, volume data
- Order book depth and liquidity analysis
- Historical price patterns and trends
- Cross-symbol analysis and correlations

⚠️ IMPORTANT GUIDELINES:
- Always emphasize risk management and position sizing
- Provide educational context with trading advice
- Never guarantee profits or trading outcomes
- Encourage users to do their own research (DYOR)
- Focus on probability-based analysis, not predictions

Current market context will be provided. Always consider recent market conditions when giving advice.

{self.memory.start_delimiter}
{memory}
{self.memory.end_delimiter}

User: {user_prompt}"""
        
        return system_prompt
    
    def clear_memory(self):
        """Clear conversation memory."""
        self.memory.clear_memory()
        self.logger._log("Conversation memory cleared")

