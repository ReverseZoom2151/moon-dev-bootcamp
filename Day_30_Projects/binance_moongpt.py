#!/usr/bin/env python3
"""
Binance Trading Assistant GPT

Interactive script to chat with an OpenAI model specialized for Binance trading,
using conversation memory and real-time market data integration.
"""

import os
import requests
import sys
from datetime import datetime
from openai import OpenAI, OpenAIError
from typing import Dict, Optional, List

# Import Binance configuration
try:
    from Day_26_Projects.binance_config import (
        API_KEY, API_SECRET, PRIMARY_SYMBOL
    )
except ImportError:
    print("Warning: binance_config not found, using default values")
    API_KEY = ""
    API_SECRET = ""
    PRIMARY_SYMBOL = "BTCUSDT"

# --- Configuration ---
CONFIG = {
    # Memory and model settings
    "MEMORY_FILE": f"binance_trading_memory_{PRIMARY_SYMBOL.lower()}.txt",
    "OPENAI_MODEL": "gpt-4o",
    "MEMORY_START_DELIMITER": "#### START BINANCE TRADING MEMORY ####",
    "MEMORY_END_DELIMITER": "#### END BINANCE TRADING MEMORY ####",
    "MAX_MEMORY_TOKENS": 6000,
    
    # Binance API settings
    "BINANCE_API_BASE": "https://api.binance.com/api/v3",
    "PRIMARY_SYMBOL": PRIMARY_SYMBOL,
    "PRICE_PRECISION": 8,
    
    # Trading context
    "EXCHANGE_NAME": "Binance",
    "DEFAULT_SYMBOLS": ["BTCUSDT", "ETHUSDT", "BNBUSDT", "ADAUSDT", "SOLUSDT"],
}

# --- OpenAI Client Initialization ---
try:
    import dontshare as d
    if not hasattr(d, 'openai_key') or not d.openai_key:
        raise ImportError("Variable 'openai_key' not found or empty in dontshare.py")
    OPENAI_API_KEY = d.openai_key
    print("OpenAI API key loaded for Binance Trading Assistant.")
except ImportError as e:
    print(f"Error loading OpenAI API key: {e}")
    print("Please ensure dontshare.py exists and contains 'openai_key'.")
    sys.exit(1)

client = OpenAI(api_key=OPENAI_API_KEY)

# --- Binance Market Data Functions ---

def get_binance_ticker(symbol: str) -> Optional[Dict]:
    """Get current ticker price for Binance symbol."""
    try:
        url = f"{CONFIG['BINANCE_API_BASE']}/ticker/24hr"
        response = requests.get(url, params={'symbol': symbol.upper()}, timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Error fetching Binance ticker for {symbol}: {e}")
        return None

def get_binance_orderbook(symbol: str, limit: int = 10) -> Optional[Dict]:
    """Get order book depth for Binance symbol."""
    try:
        url = f"{CONFIG['BINANCE_API_BASE']}/depth"
        params = {'symbol': symbol.upper(), 'limit': limit}
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Error fetching Binance order book for {symbol}: {e}")
        return None

def get_binance_klines(symbol: str, interval: str = "1h", limit: int = 24) -> Optional[List]:
    """Get recent klines/candlestick data for Binance symbol."""
    try:
        url = f"{CONFIG['BINANCE_API_BASE']}/klines"
        params = {
            'symbol': symbol.upper(),
            'interval': interval,
            'limit': limit
        }
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Error fetching Binance klines for {symbol}: {e}")
        return None

def get_market_summary() -> str:
    """Get a summary of key market data for context."""
    try:
        summary_lines = [f"\n🟠 BINANCE MARKET SNAPSHOT - {datetime.now().strftime('%H:%M:%S')} UTC 🟠"]
        
        for symbol in CONFIG["DEFAULT_SYMBOLS"]:
            ticker = get_binance_ticker(symbol)
            if ticker:
                price = float(ticker['lastPrice'])
                change_pct = float(ticker['priceChangePercent'])
                volume = float(ticker['volume'])
                
                change_indicator = "📈" if change_pct > 0 else "📉" if change_pct < 0 else "➡️"
                
                summary_lines.append(
                    f"{change_indicator} {symbol}: ${price:,.8g} ({change_pct:+.2f}%) | "
                    f"Vol: {volume:,.0f}"
                )
        
        summary_lines.append("="*70)
        return "\n".join(summary_lines)
        
    except Exception as e:
        return f"\n⚠️ Error fetching market summary: {e}\n"

def get_symbol_analysis(symbol: str) -> str:
    """Get detailed analysis for a specific symbol."""
    try:
        ticker = get_binance_ticker(symbol)
        if not ticker:
            return f"❌ Could not fetch data for {symbol}"
        
        price = float(ticker['lastPrice'])
        change_24h = float(ticker['priceChange'])
        change_pct = float(ticker['priceChangePercent'])
        high_24h = float(ticker['highPrice'])
        low_24h = float(ticker['lowPrice'])
        volume = float(ticker['volume'])
        trades = int(ticker['count'])
        
        analysis = [
            f"\n📊 {symbol.upper()} DETAILED ANALYSIS",
            f"{'='*40}",
            f"💰 Current Price: ${price:,.8g}",
            f"📈 24h Change: ${change_24h:+,.8g} ({change_pct:+.2f}%)",
            f"🔺 24h High: ${high_24h:,.8g}",
            f"🔻 24h Low: ${low_24h:,.8g}",
            f"📊 24h Volume: {volume:,.0f} {symbol[:3]}",
            f"🔢 24h Trades: {trades:,}",
            f"💹 Avg Trade Size: {volume/trades:.2f} {symbol[:3]}" if trades > 0 else "",
        ]
        
        return "\n".join(analysis) + "\n"
        
    except Exception as e:
        return f"❌ Error analyzing {symbol}: {e}"

# --- Core Functions ---

def read_file_content(filepath: str) -> str:
    """Safely reads content from a file, returning empty string if not found/error."""
    try:
        if os.path.exists(filepath):
            with open(filepath, "r", encoding="utf-8") as file:
                content = file.read().strip()
                if content:
                    return content
                else:
                    return create_initial_trading_context()
        else:
            print(f"Note: Binance memory file '{filepath}' not found. Starting new trading conversation.")
            return create_initial_trading_context()
    except IOError as e:
        print(f"Error reading file {filepath}: {e}")
        return create_initial_trading_context()

def create_initial_trading_context() -> str:
    """Creates initial trading context for new conversations."""
    context = [
        "BINANCE TRADING ASSISTANT CONTEXT",
        "="*50,
        f"Exchange: Binance",
        f"Primary Symbol: {CONFIG['PRIMARY_SYMBOL']}",
        f"Focus: Trading analysis, market data, strategy discussion",
        f"Session Started: {datetime.now().isoformat()}",
        "",
        "Available Commands:",
        "- Ask about market prices and analysis",
        "- Discuss trading strategies and technical analysis", 
        "- Get real-time market data and insights",
        "- Trading education and risk management advice",
        "",
        get_market_summary(),
        "",
        "Ready to assist with your Binance trading needs!"
    ]
    return "\n".join(context)

def write_file_content(filepath: str, content: str, mode: str = "w") -> bool:
    """Safely writes content to a file."""
    try:
        with open(filepath, mode, encoding="utf-8") as file:
            file.write(content)
        return True
    except IOError as e:
        print(f"Error writing to file {filepath}: {e}")
        return False

def get_prompt_from_console() -> str:
    """Gets the user's prompt from console input with Binance-specific commands."""
    print("\n" + "="*60)
    print("🟠 BINANCE TRADING ASSISTANT 🟠")
    print("="*60)
    print("Commands: 'price SYMBOL', 'analyze SYMBOL', 'market', 'exit'")
    print("Or just chat about trading strategies and analysis!")
    print("\nEnter your prompt (press Ctrl+D or Ctrl+Z then Enter to send):")
    
    lines = []
    while True:
        try:
            line = input()
            lines.append(line)
        except EOFError:
            break
    
    prompt = "\n".join(lines).strip()
    return handle_special_commands(prompt)

def handle_special_commands(prompt: str) -> str:
    """Handle special Binance trading commands."""
    if not prompt:
        return prompt
    
    lower_prompt = prompt.lower().strip()
    
    # Price command
    if lower_prompt.startswith('price '):
        symbol = lower_prompt.split('price ', 1)[1].strip().upper()
        ticker = get_binance_ticker(symbol)
        if ticker:
            price = float(ticker['lastPrice'])
            change_pct = float(ticker['priceChangePercent'])
            return f"Current {symbol} price is ${price:,.8g} ({change_pct:+.2f}% 24h). What would you like to know about this price action?"
        else:
            return f"Could not fetch price for {symbol}. Can you help me analyze this symbol anyway?"
    
    # Analyze command
    elif lower_prompt.startswith('analyze '):
        symbol = lower_prompt.split('analyze ', 1)[1].strip().upper()
        analysis = get_symbol_analysis(symbol)
        return f"{analysis}\n\nBased on this data, what's your analysis or what should I focus on?"
    
    # Market command
    elif lower_prompt == 'market':
        summary = get_market_summary()
        return f"{summary}\n\nWhat do you think about these market conditions? Any trading opportunities you see?"
    
    return prompt

def build_trading_prompt(memory: str, prompt: str, config: Dict) -> str:
    """Builds the full prompt with trading-specific context."""
    # Trim memory if too long
    if len(memory.split()) > config.get("MAX_MEMORY_TOKENS", 6000):
        print("📝 Trimming conversation memory...")
        memory_lines = memory.splitlines()
        # Keep initial context + recent conversation
        header_lines = [line for line in memory_lines[:20] if "BINANCE" in line or "CONTEXT" in line or "=" in line]
        recent_lines = memory_lines[-150:]  # Keep last 150 lines
        memory = "\n".join(header_lines + ["...\n"] + recent_lines)

    # Add current market context
    current_market_info = get_market_summary()
    
    system_prompt = f"""You are a specialized Binance trading assistant and market analyst. You have access to real-time Binance market data and help users with:

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

Current market context will be provided in your memory. Always consider recent market conditions when giving advice.

{current_market_info}"""

    full_prompt = (
        f"{system_prompt}\n\n"
        f"{config['MEMORY_START_DELIMITER']}\n"
        f"{memory}\n"
        f"{config['MEMORY_END_DELIMITER']}\n\n"
        f"User: {prompt}"
    )
    
    return full_prompt

def call_openai_model(prompt: str, config: Dict) -> Optional[str]:
    """Calls OpenAI model with trading-optimized settings."""
    print(f"\n🤖 Consulting Binance Trading AI ({config['OPENAI_MODEL']})...")
    try:
        response = client.chat.completions.create(
            model=config["OPENAI_MODEL"],
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,  # Balanced creativity for trading insights
            max_tokens=2000,  # Allow detailed responses
        )
        message_content = response.choices[0].message.content
        return message_content.strip() if message_content else None
    except OpenAIError as e:
        print(f"OpenAI API error: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred during AI consultation: {e}")
        return None

def update_trading_memory(prompt: str, response: str, current_memory: str, config: Dict) -> str:
    """Updates trading conversation memory with timestamp."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    memory_update = f"\n\n[{timestamp}] User: {prompt}\n[{timestamp}] Binance AI: {response}"
    new_memory = current_memory + memory_update

    if write_file_content(config["MEMORY_FILE"], new_memory.strip()):
        print(f"📝 Trading conversation updated in {config['MEMORY_FILE']}")
    else:
        print(f"⚠️ Warning: Failed to update memory file {config['MEMORY_FILE']}")

    return new_memory

# --- Main Execution ---

def main() -> None:
    """Main function for Binance Trading Assistant."""
    print("🚀" + "="*70 + "🚀")
    print("🟠           BINANCE TRADING ASSISTANT POWERED BY GPT-4O           🟠")
    print("🚀" + "="*70 + "🚀")
    print(f"📊 Primary Symbol: {CONFIG['PRIMARY_SYMBOL']}")
    print(f"🤖 AI Model: {CONFIG['OPENAI_MODEL']}")
    print(f"💾 Memory: {CONFIG['MEMORY_FILE']}")
    print(f"⚠️  Remember: This is for educational purposes. Trade responsibly!")
    print("\n🎯 Ready to help with Binance trading analysis and strategies!")

    # Load initial memory with trading context
    current_memory = read_file_content(CONFIG["MEMORY_FILE"])

    while True:
        try:
            # Get user input with command handling
            user_prompt = get_prompt_from_console()
            
            # Check for exit commands
            if not user_prompt or user_prompt.lower() in ["exit", "quit", "bye"]:
                print("\n👋 Thanks for using Binance Trading Assistant!")
                print("📊 Happy trading and stay safe out there! 🚀")
                break

            # Build full prompt with trading context
            full_prompt = build_trading_prompt(current_memory, user_prompt, CONFIG)

            # Get AI response
            ai_response = call_openai_model(full_prompt, CONFIG)

            # Display response and update memory
            if ai_response:
                print("\n" + "🤖" + "="*60 + "🤖")
                print("BINANCE TRADING AI RESPONSE:")
                print("="*70)
                print(ai_response)
                print("="*70 + "\n")
                
                # Update memory
                current_memory = update_trading_memory(user_prompt, ai_response, current_memory, CONFIG)
            else:
                print("\n❌ No response received from AI. Please try again.")

        except KeyboardInterrupt:
            print("\n\n👋 Session interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Unexpected error: {e}")
            continue

    print("\n🟠 Binance Trading Assistant session ended. Trade safely! 🟠")

if __name__ == "__main__":
    main()
