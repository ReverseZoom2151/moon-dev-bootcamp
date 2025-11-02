#!/usr/bin/env python3
"""
Bitfinex Trading Assistant GPT

Interactive script to chat with an OpenAI model specialized for Bitfinex trading,
using conversation memory and real-time market data integration.
"""

import os
import requests
import sys
from openai import OpenAI, OpenAIError
from typing import Dict, Optional, List
from datetime import datetime

# Import Bitfinex configuration
try:
    from Day_26_Projects.bitfinex_config import (
        API_KEY, API_SECRET, PRIMARY_SYMBOL
    )
except ImportError:
    print("Warning: bitfinex_config not found, using default values")
    API_KEY = ""
    API_SECRET = ""
    PRIMARY_SYMBOL = "btcusd"

# --- Configuration ---
CONFIG = {
    # Memory and model settings
    "MEMORY_FILE": f"bitfinex_trading_memory_{PRIMARY_SYMBOL.lower()}.txt",
    "OPENAI_MODEL": "gpt-4o",
    "MEMORY_START_DELIMITER": "#### START BITFINEX TRADING MEMORY ####",
    "MEMORY_END_DELIMITER": "#### END BITFINEX TRADING MEMORY ####",
    "MAX_MEMORY_TOKENS": 6000,
    
    # Bitfinex API settings
    "BITFINEX_API_BASE": "https://api.bitfinex.com",
    "PRIMARY_SYMBOL": PRIMARY_SYMBOL,
    "PRICE_PRECISION": 8,
    
    # Trading context
    "EXCHANGE_NAME": "Bitfinex",
    "DEFAULT_SYMBOLS": ["btcusd", "ethusd", "ltcusd", "xrpusd", "adausd"],
}

# --- OpenAI Client Initialization ---
try:
    import dontshare as d
    if not hasattr(d, 'openai_key') or not d.openai_key:
        raise ImportError("Variable 'openai_key' not found or empty in dontshare.py")
    OPENAI_API_KEY = d.openai_key
    print("OpenAI API key loaded for Bitfinex Trading Assistant.")
except ImportError as e:
    print(f"Error loading OpenAI API key: {e}")
    print("Please ensure dontshare.py exists and contains 'openai_key'.")
    sys.exit(1)

client = OpenAI(api_key=OPENAI_API_KEY)

# --- Bitfinex Market Data Functions ---

def get_bitfinex_ticker(symbol: str) -> Optional[Dict]:
    """Get current ticker data for Bitfinex symbol."""
    try:
        # Bitfinex uses lowercase symbols with 't' prefix for trading pairs
        formatted_symbol = f"t{symbol.upper()}" if not symbol.startswith('t') else symbol.upper()
        url = f"{CONFIG['BITFINEX_API_BASE']}/v2/ticker/{formatted_symbol}"
        
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        if isinstance(data, list) and len(data) >= 10:
            # Bitfinex ticker format: [BID, BID_SIZE, ASK, ASK_SIZE, DAILY_CHANGE, 
            #                         DAILY_CHANGE_RELATIVE, LAST_PRICE, VOLUME, HIGH, LOW]
            return {
                'symbol': formatted_symbol,
                'bid': data[0],
                'ask': data[2], 
                'last_price': data[6],
                'volume': data[7],
                'high': data[8],
                'low': data[9],
                'daily_change': data[4],
                'daily_change_perc': data[5] * 100  # Convert to percentage
            }
        return None
    except Exception as e:
        print(f"Error fetching Bitfinex ticker for {symbol}: {e}")
        return None

def get_bitfinex_orderbook(symbol: str, precision: str = "P0", len_: int = 25) -> Optional[Dict]:
    """Get order book for Bitfinex symbol."""
    try:
        formatted_symbol = f"t{symbol.upper()}" if not symbol.startswith('t') else symbol.upper()
        url = f"{CONFIG['BITFINEX_API_BASE']}/v2/book/{formatted_symbol}/{precision}"
        
        params = {'len': len_}
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        if isinstance(data, list):
            bids = [order for order in data if order[2] > 0]  # Positive amount = bid
            asks = [order for order in data if order[2] < 0]  # Negative amount = ask
            
            return {
                'symbol': formatted_symbol,
                'bids': bids[:10],  # Top 10 bids
                'asks': asks[:10]   # Top 10 asks
            }
        return None
    except Exception as e:
        print(f"Error fetching Bitfinex order book for {symbol}: {e}")
        return None

def get_bitfinex_candles(symbol: str, timeframe: str = "1h", limit: int = 24) -> Optional[List]:
    """Get recent candle data for Bitfinex symbol."""
    try:
        formatted_symbol = f"t{symbol.upper()}" if not symbol.startswith('t') else symbol.upper()
        url = f"{CONFIG['BITFINEX_API_BASE']}/v2/candles/trade:{timeframe}:{formatted_symbol}/hist"
        
        params = {'limit': limit, 'sort': -1}  # Sort descending (newest first)
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        
        return response.json()
    except Exception as e:
        print(f"Error fetching Bitfinex candles for {symbol}: {e}")
        return None

def get_market_summary() -> str:
    """Get a summary of key market data for context."""
    try:
        summary_lines = [f"\n🔶 BITFINEX MARKET SNAPSHOT - {datetime.now().strftime('%H:%M:%S')} UTC 🔶"]
        
        for symbol in CONFIG["DEFAULT_SYMBOLS"]:
            ticker = get_bitfinex_ticker(symbol)
            if ticker:
                price = ticker['last_price']
                change_pct = ticker['daily_change_perc']
                volume = ticker['volume']
                
                change_indicator = "📈" if change_pct > 0 else "📉" if change_pct < 0 else "➡️"
                
                summary_lines.append(
                    f"{change_indicator} {symbol.upper()}: ${price:,.8g} ({change_pct:+.2f}%) | "
                    f"Vol: {volume:,.0f}"
                )
        
        summary_lines.append("="*70)
        return "\n".join(summary_lines)
        
    except Exception as e:
        return f"\n⚠️ Error fetching market summary: {e}\n"

def get_symbol_analysis(symbol: str) -> str:
    """Get detailed analysis for a specific symbol."""
    try:
        ticker = get_bitfinex_ticker(symbol)
        if not ticker:
            return f"❌ Could not fetch data for {symbol}"
        
        price = ticker['last_price']
        change = ticker['daily_change']
        change_pct = ticker['daily_change_perc']
        high = ticker['high']
        low = ticker['low']
        volume = ticker['volume']
        bid = ticker['bid']
        ask = ticker['ask']
        spread = ask - bid
        spread_pct = (spread / price) * 100 if price > 0 else 0
        
        analysis = [
            f"\n📊 {symbol.upper()} DETAILED ANALYSIS",
            f"{'='*40}",
            f"💰 Current Price: ${price:,.8g}",
            f"📈 Daily Change: ${change:+,.8g} ({change_pct:+.2f}%)",
            f"🔺 Daily High: ${high:,.8g}",
            f"🔻 Daily Low: ${low:,.8g}",
            f"📊 Daily Volume: {volume:,.0f} {symbol[:3].upper()}",
            f"💹 Bid/Ask: ${bid:,.8g} / ${ask:,.8g}",
            f"📏 Spread: ${spread:.8g} ({spread_pct:.3f}%)",
            f"📈 Price Range: {((high-low)/low*100):+.2f}%" if low > 0 else "",
        ]
        
        return "\n".join(analysis) + "\n"
        
    except Exception as e:
        return f"❌ Error analyzing {symbol}: {e}"

def get_funding_rates() -> str:
    """Get funding rates for major perpetual contracts."""
    try:
        funding_symbols = ["tBTCF0:USTF0", "tETHF0:USTF0", "tXRPF0:USTF0"]
        funding_data = []
        
        for fsymbol in funding_symbols:
            try:
                url = f"{CONFIG['BITFINEX_API_BASE']}/v2/ticker/{fsymbol}"
                response = requests.get(url, timeout=5)
                if response.status_code == 200:
                    data = response.json()
                    if isinstance(data, list) and len(data) >= 7:
                        funding_rate = data[5] * 100 * 3  # Convert to 8h rate %
                        base_symbol = fsymbol.split('F0')[0][1:]
                        funding_data.append(f"{base_symbol}: {funding_rate:+.4f}%")
            except:
                continue
        
        if funding_data:
            return f"\n💰 FUNDING RATES (8H): {' | '.join(funding_data)}\n"
        return ""
        
    except Exception as e:
        return f"\n⚠️ Error fetching funding rates: {e}\n"

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
            print(f"Note: Bitfinex memory file '{filepath}' not found. Starting new trading conversation.")
            return create_initial_trading_context()
    except IOError as e:
        print(f"Error reading file {filepath}: {e}")
        return create_initial_trading_context()

def create_initial_trading_context() -> str:
    """Creates initial trading context for new conversations."""
    context = [
        "BITFINEX TRADING ASSISTANT CONTEXT",
        "="*50,
        f"Exchange: Bitfinex",
        f"Primary Symbol: {CONFIG['PRIMARY_SYMBOL']}",
        f"Focus: Advanced trading analysis, derivatives, lending markets",
        f"Session Started: {datetime.now().isoformat()}",
        "",
        "Specialized Features:",
        "- Spot and derivatives trading analysis",
        "- Funding rates and lending market insights", 
        "- Advanced order types and margin trading",
        "- Professional trading tools and strategies",
        "",
        get_market_summary(),
        get_funding_rates(),
        "",
        "Ready to assist with professional Bitfinex trading!"
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
    """Gets the user's prompt from console input with Bitfinex-specific commands."""
    print("\n" + "="*60)
    print("🔶 BITFINEX TRADING ASSISTANT 🔶")
    print("="*60)
    print("Commands: 'price SYMBOL', 'analyze SYMBOL', 'market', 'funding', 'exit'")
    print("Or discuss advanced trading strategies and derivatives!")
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
    """Handle special Bitfinex trading commands."""
    if not prompt:
        return prompt
    
    lower_prompt = prompt.lower().strip()
    
    # Price command
    if lower_prompt.startswith('price '):
        symbol = lower_prompt.split('price ', 1)[1].strip().lower()
        ticker = get_bitfinex_ticker(symbol)
        if ticker:
            price = ticker['last_price']
            change_pct = ticker['daily_change_perc']
            return f"Current {symbol.upper()} price is ${price:,.8g} ({change_pct:+.2f}% today). What would you like to know about this price action or trading opportunities?"
        else:
            return f"Could not fetch price for {symbol}. Can you help me analyze this symbol anyway?"
    
    # Analyze command
    elif lower_prompt.startswith('analyze '):
        symbol = lower_prompt.split('analyze ', 1)[1].strip().lower()
        analysis = get_symbol_analysis(symbol)
        return f"{analysis}\n\nBased on this Bitfinex data, what's your analysis or trading strategy?"
    
    # Market command
    elif lower_prompt == 'market':
        summary = get_market_summary()
        return f"{summary}\n\nWhat do you think about current market conditions? Any opportunities in derivatives or spot markets?"
    
    # Funding command
    elif lower_prompt == 'funding':
        funding = get_funding_rates()
        market = get_market_summary()
        return f"{market}{funding}\n\nHow do these funding rates affect trading strategy? Any arbitrage opportunities you see?"
    
    return prompt

def build_trading_prompt(memory: str, prompt: str, config: Dict) -> str:
    """Builds the full prompt with Bitfinex trading-specific context."""
    # Trim memory if too long
    if len(memory.split()) > config.get("MAX_MEMORY_TOKENS", 6000):
        print("📝 Trimming conversation memory...")
        memory_lines = memory.splitlines()
        # Keep initial context + recent conversation
        header_lines = [line for line in memory_lines[:20] if "BITFINEX" in line or "CONTEXT" in line or "=" in line]
        recent_lines = memory_lines[-150:]  # Keep last 150 lines
        memory = "\n".join(header_lines + ["...\n"] + recent_lines)

    # Add current market context
    current_market_info = get_market_summary()
    current_funding_info = get_funding_rates()
    
    system_prompt = f"""You are a specialized Bitfinex trading assistant and professional market analyst. You have access to real-time Bitfinex market data and help users with:

🎯 CORE CAPABILITIES:
- Advanced spot and derivatives trading analysis
- Margin trading and lending market strategies
- Funding rate analysis and carry trade opportunities
- Professional order types and execution strategies
- Market microstructure and liquidity analysis
- Risk management for leveraged positions

📊 MARKET DATA ACCESS:
- Real-time prices, order book depth, and spread analysis
- Funding rates for perpetual contracts
- Historical price patterns and volatility analysis
- Cross-pair arbitrage and correlation opportunities
- Lending rates and margin funding costs

🔶 BITFINEX SPECIALTIES:
- Advanced order types (OCO, trailing stops, etc.)
- Margin trading up to 10:1 leverage
- Lending and funding markets
- Professional trading tools and APIs
- Derivatives and perpetual contracts

⚠️ IMPORTANT GUIDELINES:
- Always emphasize risk management, especially with leverage
- Provide detailed analysis of margin requirements and liquidation risks
- Never guarantee profits - focus on probability and risk-reward ratios
- Encourage proper position sizing and stop-loss strategies
- Discuss both opportunities and risks of leveraged positions

Current market context will be provided. Always consider funding rates, spreads, and liquidity when giving advice.

{current_market_info}
{current_funding_info}"""

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
    print(f"\n🤖 Consulting Bitfinex Trading AI ({config['OPENAI_MODEL']})...")
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
    memory_update = f"\n\n[{timestamp}] User: {prompt}\n[{timestamp}] Bitfinex AI: {response}"
    new_memory = current_memory + memory_update

    if write_file_content(config["MEMORY_FILE"], new_memory.strip()):
        print(f"📝 Trading conversation updated in {config['MEMORY_FILE']}")
    else:
        print(f"⚠️ Warning: Failed to update memory file {config['MEMORY_FILE']}")

    return new_memory

# --- Main Execution ---

def main() -> None:
    """Main function for Bitfinex Trading Assistant."""
    print("🚀" + "="*70 + "🚀")
    print("🔶          BITFINEX TRADING ASSISTANT POWERED BY GPT-4O          🔶")
    print("🚀" + "="*70 + "🚀")
    print(f"📊 Primary Symbol: {CONFIG['PRIMARY_SYMBOL'].upper()}")
    print(f"🤖 AI Model: {CONFIG['OPENAI_MODEL']}")
    print(f"💾 Memory: {CONFIG['MEMORY_FILE']}")
    print(f"⚡ Focus: Professional trading, derivatives, margin markets")
    print(f"⚠️  Remember: Leverage amplifies both gains and losses. Trade wisely!")
    print("\n🎯 Ready for professional Bitfinex trading analysis!")

    # Load initial memory with trading context
    current_memory = read_file_content(CONFIG["MEMORY_FILE"])

    while True:
        try:
            # Get user input with command handling
            user_prompt = get_prompt_from_console()
            
            # Check for exit commands
            if not user_prompt or user_prompt.lower() in ["exit", "quit", "bye"]:
                print("\n👋 Thanks for using Bitfinex Trading Assistant!")
                print("📊 Trade smart, manage risk, and stay profitable! 🚀")
                break

            # Build full prompt with trading context
            full_prompt = build_trading_prompt(current_memory, user_prompt, CONFIG)

            # Get AI response
            ai_response = call_openai_model(full_prompt, CONFIG)

            # Display response and update memory
            if ai_response:
                print("\n" + "🤖" + "="*60 + "🤖")
                print("BITFINEX TRADING AI RESPONSE:")
                print("="*70)
                print(ai_response)
                print("="*70 + "\n")
                
                # Update memory
                current_memory = update_trading_memory(user_prompt, ai_response, current_memory, CONFIG)
            else:
                print("\n❌ No response received from AI. Please try again.")

        except KeyboardInterrupt:
            print("\n\n👋 Session interrupted. Trade safely!")
            break
        except Exception as e:
            print(f"\n❌ Unexpected error: {e}")
            continue

    print("\n🔶 Bitfinex Trading Assistant session ended. Professional trading awaits! 🔶")

if __name__ == "__main__":
    main()
