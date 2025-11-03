#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gordon CLI - The Complete Trading Assistant
===========================================
Combines financial research with technical trading.
"""

import os
import sys
import argparse
import asyncio
import json
import pandas as pd
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
# Import prompt_toolkit components with platform detection
try:
    from prompt_toolkit import PromptSession
    from prompt_toolkit.history import InMemoryHistory
    from prompt_toolkit.output import create_output
    from prompt_toolkit.input import create_input
    PROMPT_TOOLKIT_AVAILABLE = True
except ImportError:
    PROMPT_TOOLKIT_AVAILABLE = False
    PromptSession = None
    InMemoryHistory = None
    create_output = None
    create_input = None

# Fix Windows console encoding for Unicode characters
if sys.platform == 'win32':
    import codecs
    # Try to set UTF-8 code page
    try:
        os.system('chcp 65001 > nul 2>&1')
    except:
        pass
    # Wrap stdout/stderr with UTF-8 encoder
    try:
        if hasattr(sys.stdout, 'buffer'):
            sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
        if hasattr(sys.stderr, 'buffer'):
            sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')
    except:
        pass

# Load environment variables
load_dotenv()

# Verify critical dependencies are available before importing gordon modules
try:
    import ccxt
    CCXT_AVAILABLE = True
except ImportError:
    CCXT_AVAILABLE = False
    print("⚠️ Warning: ccxt module not found. Please install with: pip3 install ccxt")
    print("   Current Python:", sys.executable)
    print("   Python path:", sys.path[:3])
    sys.exit(1)

# Add the project root (parent of gordon) to path
# This allows running as both: python cli.py and python -m gordon.entrypoints.cli
project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Also add gordon directory to path for imports
gordon_dir = str(Path(__file__).parent.parent)
if gordon_dir not in sys.path:
    sys.path.insert(0, gordon_dir)

# Now we can import gordon modules properly
from gordon.agent.agent import Agent
from gordon.agent.conversational_assistant import ConversationalAssistant

# Conversation utilities (Day 30)
from gordon.agent import (
    ConversationSearcher,
    ConversationAnalytics,
    MultiUserConversationManager,
    export_all_conversations
)


def cmd_search_conversations(query: str, memory_dir: str = './conversation_memory'):
    """Search conversations."""
    searcher = ConversationSearcher(Path(memory_dir))
    results = searcher.search(query)
    
    if not results:
        print(f"No results found for '{query}'")
        return
    
    print(f"\n🔍 Found {len(results)} result(s) for '{query}':\n")
    
    for result in results[:10]:  # Limit to 10 results
        print(f"📄 File: {result['file']}")
        print(f"   Matches: {result['match_count']}")
        for match in result['matches'][:3]:  # Show first 3 matches
            print(f"   Line {match['line_number']}: {match['matched_line'][:80]}...")
        print()


def cmd_export_conversations(format: str = 'json', memory_dir: str = './conversation_memory', output_dir: Optional[str] = None):
    """Export conversations."""
    memory_path = Path(memory_dir)
    output_path = Path(output_dir) if output_dir else Path(memory_dir) / 'exports'
    
    print(f"\n📤 Exporting conversations to {format.upper()} format...")
    exported = export_all_conversations(memory_path, output_path, format=format)
    
    print(f"✅ Exported {len(exported)} conversation(s) to {output_path}")
    for file in exported[:5]:  # Show first 5
        print(f"   - {file.name}")


def cmd_analytics(memory_dir: str = './conversation_memory'):
    """Show conversation analytics."""
    analytics = ConversationAnalytics(Path(memory_dir))
    report = analytics.generate_insights_report()
    print(report)


def cmd_list_users(memory_dir: str = './conversation_memory'):
    """List all users."""
    manager = MultiUserConversationManager(base_memory_dir=memory_dir)
    users = manager.list_users()
    
    if not users:
        print("\n👤 No users found")
        return
    
    print(f"\n👤 Found {len(users)} user(s):\n")
    for user_id in users:
        stats = manager.get_user_stats(user_id)
        print(f"  {user_id}:")
        print(f"    Conversations: {stats['total_conversations']}")
        print(f"    Size: {stats['total_size_mb']} MB")


def cmd_switch_user(user_id: str, memory_dir: str = './conversation_memory'):
    """Switch to a different user."""
    manager = MultiUserConversationManager(base_memory_dir=memory_dir)
    manager.switch_user(user_id)
    print(f"✅ Switched to user: {user_id}")


def print_banner():
    """Print Gordon's welcome banner."""
    try:
        banner = """
    ╔════════════════════════════════════════════════════╗
    ║                                                    ║
    ║   ██████╗  ██████╗ ██████╗ ██████╗  ██████╗ ███╗  ║║
    ║  ██╔════╝ ██╔═══██╗██╔══██╗██╔══██╗██╔═══██╗████╗ ║║
    ║  ██║  ███╗██║   ██║██████╔╝██║  ██║██║   ██║██╔██╗║║
    ║  ██║   ██║██║   ██║██╔══██╗██║  ██║██║   ██║██║╚██║║
    ║  ╚██████╔╝╚██████╔╝██║  ██║██████╔╝╚██████╔╝██║ ╚█║║
    ║   ╚═════╝  ╚═════╝ ╚═╝  ╚═╝╚═════╝  ╚═════╝ ╚═╝  ╚╝║
    ║                                                    ║
    ║        Financial Research + Trading Agent          ║
    ║           Fundamental + Technical Analysis         ║
    ║                 Powered by AI & Data               ║
    ╚════════════════════════════════════════════════════╝
    """
        print(banner)
    except UnicodeEncodeError:
        # Fallback banner without Unicode characters
        print("""
    ============================================================
            GORDON - FINANCIAL RESEARCH & TRADING AGENT
    ============================================================
            Financial Research + Trading Agent
           Fundamental + Technical Analysis
                 Powered by AI & Data
    ============================================================
        """)


def print_help():
    """Print help information."""
    print("""
    🤖 Gordon Commands & Examples:
    ═══════════════════════════════════════

    📊 FINANCIAL RESEARCH:
    • "Analyze Apple's financial health"
    • "What's Tesla's revenue growth?"
    • "Compare Microsoft and Google margins"
    • "Show me Amazon's latest 10-K filing"
    • "Get analyst estimates for NVDA"

    📈 TRADING STRATEGIES:
    • "Run RSI strategy on BTC/USDT"
    • "Execute SMA crossover on ETH/USDT"
    • "Run VWAP strategy on SOL/USDT"
    • "Execute mean reversion on MATIC/USDT"
    • "Run Bollinger Bands on BNB/USDT"

    💰 ORDER EXECUTION:
    • "Place market buy for 0.1 BTC"
    • "Place limit order to buy ETH at $1800"
    • "Execute TWAP order for 1 BTC over 2 hours"
    • "Execute VWAP order for 5 ETH"

    🔬 BACKTESTING & OPTIMIZATION:
    • "Backtest SMA strategy on BTC from 2024-01-01 to 2024-12-31"
    • "Optimize RSI parameters for ETH"
    • "Test mean reversion on SOL with 20-day lookback"
    • "Compare strategies on BTC"

    🛡️ RISK MANAGEMENT:
    • "Check risk for buying 1 BTC"
    • "Calculate position size for 2% risk"
    • "Show my portfolio risk metrics"
    • "Check if trade violates limits"

    📡 MARKET DATA:
    • "Get live price of BTC/USDT"
    • "Show orderbook for ETH/USDT"
    • "Get recent trades on SOL/USDT"
    • "Stream market data for BTC"

    💡 HYBRID ANALYSIS (Gordon's Specialty!):
    • "hybrid AAPL" - Full fundamental + technical analysis
    • "analyze TSLA with trading signals"
    • "Full analysis on MSFT"
    • "Should I buy NVDA?"
    
    🐋 WHALE TRACKING (Natural Language):
    • "Track whale positions for Bitcoin"
    • "Show me large positions on BTCUSDT"
    • "Find institutional traders"
    • "Whale tracker for ETH"
    • CLI: whale-track [SYMBOL] [MIN_VALUE]
    
    📏 POSITION SIZING (Natural Language):
    • "Calculate position size for $10k balance at $50k price"
    • "How much should I buy with 2x leverage?"
    • "Position size with $10k balance, entry $50k, stop $48k"
    • "Calculate size for $1000 at $50k price"
    • CLI: position-size [balance|usd|risk] [args...]
    
    🎯 LIQUIDATION HUNTER (Natural Language):
    • "Run liquidation hunter for BTC"
    • "Analyze liquidation risk for ETHUSDT"
    • "Show liquidation data"
    • "Get whale positions data"
    • CLI: liq-hunter [SYMBOL], moondev-data [TYPE]
    
    📊 ORDER BOOK ANALYSIS (Natural Language):
    • "Analyze order book for BTCUSDT"
    • "Show whale orders"
    • "Order book depth analysis"
    • CLI: orderbook-analyze SYMBOL

    💬 QUICK COMMANDS (Day 30):
    • price SYMBOL - Get current price (e.g., "price BTCUSDT")
    • analyze SYMBOL - Get detailed analysis (e.g., "analyze ETHUSDT")
    • market - Get market summary
    
    ⚙️ SYSTEM COMMANDS:
    • help - Show this help message
    • status - Show system status
    • config - Show/modify configuration
    • risk - Show risk limits
    • clear - Clear screen
    • clear-memory - Clear conversation memory
    
    💬 CONVERSATION COMMANDS (Day 30):
    • search-conversations QUERY - Search conversation history
    • export-conversations [json|csv|txt] - Export conversations
    • conversation-analytics - Show conversation analytics
    • list-users - List all conversation users
    • switch-user USER_ID - Switch to different user
    
    🤖 ML INDICATOR COMMANDS (Day 33):
    • ml-discover-indicators - Discover available indicators
    • ml-evaluate-indicators [SYMBOL] - Evaluate indicator set
    • ml-loop-indicators [SYMBOL] [GENERATIONS] - Run indicator looping
    • ml-top-indicators [N] - Show top N indicators
    
    🔍 TRADER INTELLIGENCE COMMANDS (Day 38):
    • trader-analyze SYMBOL [DAYS] - Analyze early buyers and traders
    • find-accounts SYMBOL [MAX] - Find accounts to follow
    • institutional-traders SYMBOL - Analyze institutional traders
    
    🐋 WHALE TRACKING COMMANDS (Day 44 + Day 46):
    • whale-track [SYMBOL] [MIN_VALUE] - Track whale positions
    • multi-whale-track [SYMBOL] [MIN_VALUE] - Track multiple whale addresses (Day 46)
    • liquidation-risk [SYMBOL] [THRESHOLD%] - Analyze liquidation risk (Day 46)
    • aggregate-positions [SYMBOL] - Show aggregated positions report (Day 46)
    • position-size [balance|usd|risk] [args...] - Calculate position size
    
    🎯 LIQUIDATION HUNTER COMMANDS (Day 45):
    • liq-hunter [SYMBOL] - Run liquidation hunter analysis
    • moondev-data [liquidations|funding|oi|positions|whales] [LIMIT] - Fetch Moon Dev API data
    • orderbook-analyze SYMBOL - Analyze order book depth and whale orders
    
    📊 MARKET DASHBOARD COMMANDS (Day 47):
    • market-dashboard [binance|bitfinex] - Run full market dashboard analysis
    • trending-tokens [binance|bitfinex] - Show trending tokens
    • new-listings [binance|bitfinex] - Show new token listings
    • volume-leaders [binance|bitfinex] - Show volume leaders
    • funding-rates - Show funding rates (Bitfinex only)
    
    ⚡ QUICK BUY/SELL COMMANDS (Day 51):
    • quick-buy SYMBOL [USD_AMOUNT] - Execute instant market buy
    • quick-sell SYMBOL - Execute instant market sell
    • quick-monitor - Start file monitoring for rapid execution
    • add-to-quick-file SYMBOL [BUY|SELL] - Add token to monitoring file
    
    • exit/quit - Exit Gordon

    💡 Advanced Features:
    - Gordon combines Warren Buffett's fundamentals with quant trading
    - Supports 10+ trading strategies
    - Real-time market data streaming
    - Advanced algo orders (TWAP, VWAP, Iceberg)
    - Multi-exchange support (Binance, Bitfinex, Hyperliquid)
    - Comprehensive backtesting with optimization

    🚨 Safety First:
    - Dry-run mode enabled by default
    - Risk limits enforced automatically
    - Position sizing based on Kelly Criterion
    - Emergency stop available
    """)


def run_interactive_mode(gordon: Agent, conversational: bool = False, exchange: str = 'binance', symbol: str = 'BTCUSDT', config: dict = None):
    """Run Gordon in interactive mode."""
    print_banner()
    
    # Initialize conversational assistant if enabled
    conversational_assistant = None
    if conversational:
        try:
            from gordon.agent.gordon_agent import GordonAgent
            gordon_agent = GordonAgent(config or {})
            conversational_assistant = ConversationalAssistant(
                gordon_agent=gordon_agent,
                exchange_name=exchange,
                symbol=symbol,
                config=config or {}
            )
            print("\n💬 Conversational mode enabled (Day 30)")
            print(f"📊 Exchange: {exchange.upper()}, Symbol: {symbol}")
            print("💾 Conversation memory: Active")
        except Exception as e:
            print(f"\n⚠️ Failed to initialize conversational mode: {e}")
            print("Falling back to standard mode...")
            conversational_assistant = None
    
    if not conversational_assistant:
        print("\n🚀 Gordon is ready! Type 'help' for commands or 'exit' to quit.\n")
    else:
        print("\n🚀 Gordon is ready! Type 'help' for commands or 'exit' to quit.\n")
        print("💡 Quick commands: 'price SYMBOL', 'analyze SYMBOL', 'market'\n")

    # Create a prompt session with history (works across PowerShell, WSL, cmd.exe, Linux, macOS)
    session = None
    use_prompt_toolkit = True
    
    # Initialize platform detection variables BEFORE try block (needed in exception handler)
    is_wsl = False
    is_windows = sys.platform == 'win32'
    is_windows_terminal = False
    
    if PROMPT_TOOLKIT_AVAILABLE:
        try:
            # Check if we're in a proper interactive terminal
            if not sys.stdin.isatty():
                raise RuntimeError("Not running in an interactive terminal")
            
            # Check for Windows Terminal (has better prompt_toolkit support)
            if is_windows:
                try:
                    wt_session = os.environ.get('WT_SESSION')
                    if wt_session:
                        is_windows_terminal = True
                except:
                    pass
            
            if sys.platform == 'linux':
                # Check if running in WSL
                try:
                    with open('/proc/version', 'r') as f:
                        if 'microsoft' in f.read().lower():
                            is_wsl = True
                except:
                    pass
            
            # Strategy 1: Try with explicit output/input handlers (best for cross-platform)
            # This works in Windows Terminal, cmd.exe, Linux, macOS, WSL
            try:
                output = create_output()
                input_handler = create_input()
                
                # Create session with explicit handlers
                session = PromptSession(
                    history=InMemoryHistory(),
                    output=output,
                    input=input_handler
                )
            except Exception as e1:
                # Strategy 2: Try without explicit handlers (works in cmd.exe, some PowerShell)
                try:
                    session = PromptSession(history=InMemoryHistory())
                except Exception as e2:
                    # Strategy 3: For WSL/Linux, try with VT100 output
                    if is_wsl or (sys.platform == 'linux' and not is_windows):
                        try:
                            from prompt_toolkit.output.vt100 import Vt100_Output
                            from prompt_toolkit.input.posix_pipe import PosixPipeInput
                            output = Vt100_Output(sys.stdout)
                            input_handler = PosixPipeInput()
                            session = PromptSession(
                                history=InMemoryHistory(),
                                output=output,
                                input=input_handler
                            )
                        except Exception as e3:
                            # Strategy 4: Try with TERM environment variable set (helps in WSL)
                            try:
                                # Set environment variable to help prompt_toolkit detect terminal
                                original_term = os.environ.get('TERM')
                                if 'TERM' not in os.environ or not os.environ.get('TERM'):
                                    os.environ['TERM'] = 'xterm-256color'
                                try:
                                    session = PromptSession(history=InMemoryHistory())
                                finally:
                                    # Restore original TERM if we changed it
                                    if original_term is None:
                                        os.environ.pop('TERM', None)
                                    else:
                                        os.environ['TERM'] = original_term
                            except Exception:
                                # Fall through to standard input
                                raise e3
                    else:
                        # For Windows PowerShell/cmd.exe, if both fail, fall back to standard input
                        # This is expected in PowerShell - it works fine with standard input()
                        raise e2
                    
        except Exception as e:
            # Fallback to standard input if prompt_toolkit fails
            use_prompt_toolkit = False
            error_msg = str(e) if e else ""
            error_type = type(e).__name__ if e else ""
            
            # Only show warning for unexpected errors
            if is_wsl:
                # WSL should usually work - log the error for debugging
                if error_msg:
                    print(f"⚠️ Note: Advanced prompt features unavailable in WSL ({error_msg}). Using standard input.\n")
                elif error_type:
                    print(f"⚠️ Note: Advanced prompt features unavailable in WSL ({error_type}). Using standard input.\n")
            elif is_windows and not is_windows_terminal:
                # Windows PowerShell/cmd.exe without Windows Terminal - this is expected
                # Silently fall back to standard input (no warning)
                pass
            elif "No Windows console" not in error_msg and "NoConsoleScreenBufferError" not in error_type:
                # Only show warning for truly unexpected errors
                if error_msg and "Not running in an interactive terminal" not in error_msg:
                    # Don't show warning for expected cases - silently fall back
                    pass
    else:
        # prompt_toolkit not installed - use standard input
        use_prompt_toolkit = False

    while True:
        try:
            # Get user input with proper error handling
            if use_prompt_toolkit and session:
                try:
                    query = session.prompt("\n🤖 Gordon> ").strip()
                except (EOFError, KeyboardInterrupt):
                    # Handle Ctrl+D (EOF) or Ctrl+C gracefully
                    print("\n\n👋 Goodbye! Happy trading! 📈\n")
                    break
            else:
                try:
                    query = input("\n🤖 Gordon> ").strip()
                except (EOFError, KeyboardInterrupt):
                    # Handle Ctrl+D (EOF) or Ctrl+C gracefully
                    print("\n\n👋 Goodbye! Happy trading! 📈\n")
                    break
            
            # Handle empty input
            if not query:
                continue

            # Try natural language parsing first (before exact command matching)
            nl_parsed = False
            try:
                from gordon.agent.nl_command_parser import NLCommandParser
                parsed = NLCommandParser.parse(query)
                if parsed:
                    command_prefix, params = parsed
                    # Convert to CLI command
                    converted_command = NLCommandParser.convert_to_command(command_prefix, params)
                    # Use converted command instead of original query
                    query = converted_command
                    nl_parsed = True
            except Exception:
                # If NL parsing fails, continue with original query
                pass

            if query.lower() == 'help':
                print_help()
                continue

            elif query.lower() == 'clear':
                os.system('cls' if os.name == 'nt' else 'clear')
                print_banner()
                continue
            
            elif query.lower() == 'clear-memory':
                if conversational_assistant:
                    conversational_assistant.clear_memory()
                    print("✅ Conversation memory cleared")
                else:
                    print("⚠️ Conversational mode not enabled")
                continue

            elif query.lower().startswith('search-conversations '):
                # Search conversation history
                search_query = query[21:].strip()
                if search_query:
                    cmd_search_conversations(search_query)
                else:
                    print("⚠️ Usage: search-conversations QUERY")
                continue

            elif query.lower().startswith('export-conversations'):
                # Export conversations
                parts = query.split()
                format_type = parts[1] if len(parts) > 1 else 'json'
                cmd_export_conversations(format=format_type)
                continue

            elif query.lower() == 'conversation-analytics':
                # Show conversation analytics
                cmd_analytics()
                continue

            elif query.lower() == 'list-users':
                # List conversation users
                cmd_list_users()
                continue

            elif query.lower().startswith('switch-user '):
                # Switch conversation user
                user_id = query[12:].strip()
                if user_id:
                    cmd_switch_user(user_id)
                else:
                    print("⚠️ Usage: switch-user USER_ID")
                continue

            elif query.lower() == 'ml-discover-indicators':
                # Discover indicators
                print("\n🔍 Discovering available indicators...")
                try:
                    from gordon.ml import MLIndicatorManager
                    ml_manager = MLIndicatorManager()
                    indicators = ml_manager.discover_indicators()
                    print("\n✅ Indicator Discovery Complete:")
                    if 'pandas_ta' in indicators:
                        print(f"  📊 pandas_ta: {len(indicators['pandas_ta'])} indicators")
                    if 'talib' in indicators:
                        print(f"  📊 talib: {len(indicators['talib'])} indicators")
                    print("\n📁 Indicator lists saved to ./ml_results/")
                except Exception as e:
                    print(f"❌ Error: {e}")
                    import traceback
                    traceback.print_exc()
                continue

            elif query.lower().startswith('ml-evaluate-indicators'):
                # ML Indicator Evaluation
                parts = query.split()
                symbol = parts[1] if len(parts) > 1 else 'BTCUSDT'
                print(f"\n🔬 Starting ML indicator evaluation for {symbol}...")
                try:
                    from gordon.ml import MLIndicatorManager
                    from gordon.backtesting.data.fetcher import DataFetcher
                    import yaml
                    config_path = Path(__file__).parent.parent / 'config.yaml'
                    config = {}
                    if config_path.exists():
                        with open(config_path, 'r') as f:
                            config = yaml.safe_load(f)
                    data_fetcher = DataFetcher()
                    df = data_fetcher.fetch_for_backtesting_lib(symbol, '1h', limit=2000)
                    if df.empty:
                        print(f"❌ Could not fetch data for {symbol}")
                        continue
                    ml_manager = MLIndicatorManager(config)
                    result = ml_manager.evaluate_indicator_set(
                        df,
                        pandas_ta_indicators=[{'kind': 'rsi'}, {'kind': 'macd'}, {'kind': 'ema', 'length': 20}],
                        talib_indicators=['ADX', 'CCI', 'ROC']
                    )
                    if 'results' in result:
                        print("\n📊 Evaluation Results:")
                        for model_result in result['results']:
                            if 'error' not in model_result:
                                print(f"  {model_result['model']}:")
                                print(f"    R2: {model_result.get('r2', 0):.4f}")
                                print(f"    MSE: {model_result.get('mse', 0):.4f}")
                    else:
                        print(f"❌ Evaluation failed: {result.get('error', 'Unknown error')}")
                except Exception as e:
                    print(f"❌ Error: {e}")
                    import traceback
                    traceback.print_exc()
                continue

            elif query.lower().startswith('ml-loop-indicators'):
                # ML Indicator Looping
                parts = query.split()
                symbol = parts[1] if len(parts) > 1 else 'BTCUSDT'
                generations = int(parts[2]) if len(parts) > 2 else 10
                print(f"\n🔄 Starting ML indicator looping for {symbol} ({generations} generations)...")
                try:
                    from gordon.ml import MLIndicatorManager
                    from gordon.backtesting.data.fetcher import DataFetcher
                    import yaml
                    config_path = Path(__file__).parent.parent / 'config.yaml'
                    config = {}
                    if config_path.exists():
                        with open(config_path, 'r') as f:
                            config = yaml.safe_load(f)
                    if 'ml' not in config:
                        config['ml'] = {}
                    if 'indicator_looping' not in config['ml']:
                        config['ml']['indicator_looping'] = {}
                    config['ml']['indicator_looping']['generations'] = generations
                    data_fetcher = DataFetcher()
                    df = data_fetcher.fetch_for_backtesting_lib(symbol, '1h', limit=2000)
                    if df.empty:
                        print(f"❌ Could not fetch data for {symbol}")
                        continue
                    ml_manager = MLIndicatorManager(config)
                    print("⏳ This may take a while...")
                    results = ml_manager.run_indicator_looping(df)
                    print(f"\n✅ Completed {len(results)} generations")
                    print("📁 Results saved to ./ml_results/")
                except Exception as e:
                    print(f"❌ Error: {e}")
                    import traceback
                    traceback.print_exc()
                continue

            elif query.lower().startswith('ml-top-indicators'):
                # Show top indicators
                parts = query.split()
                top_n = int(parts[1]) if len(parts) > 1 else 50
                print(f"\n🏆 Top {top_n} Indicators:")
                try:
                    from gordon.ml import MLIndicatorManager
                    import yaml
                    config_path = Path(__file__).parent.parent / 'config.yaml'
                    config = {}
                    if config_path.exists():
                        with open(config_path, 'r') as f:
                            config = yaml.safe_load(f)
                    ml_manager = MLIndicatorManager(config)
                    rankings = ml_manager.get_top_indicators(top_n=top_n)
                    if rankings:
                        for metric, df in rankings.items():
                            if not df.empty:
                                print(f"\n📊 Ranked by {metric}:")
                                print(df.head(10).to_string(index=False))
                    else:
                        print("⚠️  No ranking data found. Run 'ml-loop-indicators SYMBOL' first.")
                except Exception as e:
                    print(f"❌ Error: {e}")
                    import traceback
                    traceback.print_exc()
                continue

            elif query.lower().startswith('rrs-analyze'):
                # RRS Analysis
                parts = query.split()
                symbol = parts[1] if len(parts) > 1 else 'BTCUSDT'
                timeframe = parts[2] if len(parts) > 2 else '1h'
                benchmark = parts[3] if len(parts) > 3 else 'BTCUSDT'
                print(f"\n📊 Running RRS analysis: {symbol} vs {benchmark} ({timeframe})...")
                try:
                    from gordon.research.rrs import RRSManager
                    from gordon.backtesting.data.fetcher import DataFetcher
                    import yaml
                    config_path = Path(__file__).parent.parent / 'config.yaml'
                    config = {}
                    if config_path.exists():
                        with open(config_path, 'r') as f:
                            config = yaml.safe_load(f)
                    data_fetcher = DataFetcher()
                    rrs_manager = RRSManager(data_fetcher=data_fetcher, config=config.get('rrs', {}))
                    rrs_result = rrs_manager.analyze_symbol(symbol, timeframe=timeframe)
                    if rrs_result is not None and not rrs_result.empty:
                        latest = rrs_result.iloc[-1]
                        print(f"\n✅ RRS Analysis Complete:")
                        print(f"  Symbol: {symbol}")
                        print(f"  Current RRS: {latest['smoothed_rrs']:.4f}")
                        print(f"  RRS Momentum: {latest['rrs_momentum']:.4f}")
                        print(f"  RRS Trend: {latest['rrs_trend']:.4f}")
                        print(f"  Risk-Adjusted RRS: {latest['risk_adjusted_rrs']:.4f}")
                        print(f"  Outperformance Ratio: {latest['outperformance_ratio']:.2%}")
                        print(f"  Volume Ratio: {latest['volume_ratio']:.2f}")
                    else:
                        print(f"❌ Could not generate RRS analysis for {symbol}")
                except Exception as e:
                    print(f"❌ Error: {e}")
                    import traceback
                    traceback.print_exc()
                continue

            elif query.lower().startswith('rrs-rankings'):
                # RRS Rankings
                parts = query.split()
                symbols_str = parts[1] if len(parts) > 1 else 'BTCUSDT,ETHUSDT,BNBUSDT,SOLUSDT'
                timeframe = parts[2] if len(parts) > 2 else '1h'
                benchmark = parts[3] if len(parts) > 3 else 'BTCUSDT'
                symbols = [s.strip() for s in symbols_str.split(',')]
                print(f"\n🏆 Generating RRS Rankings for {len(symbols)} symbols...")
                try:
                    from gordon.research.rrs import RRSManager
                    from gordon.backtesting.data.fetcher import DataFetcher
                    import yaml
                    config_path = Path(__file__).parent.parent / 'config.yaml'
                    config = {}
                    if config_path.exists():
                        with open(config_path, 'r') as f:
                            config = yaml.safe_load(f)
                    data_fetcher = DataFetcher()
                    rrs_manager = RRSManager(data_fetcher=data_fetcher, config=config.get('rrs', {}))
                    rrs_results = rrs_manager.analyze_multiple_symbols(symbols, timeframe=timeframe, benchmark=benchmark)
                    if rrs_results:
                        signals_df = rrs_manager.generate_rankings_and_signals(rrs_results, timeframe)
                        if not signals_df.empty:
                            print(f"\n📊 Top 10 Rankings:")
                            print(signals_df.head(10)[['rank', 'symbol', 'current_rrs', 'primary_signal', 'signal_confidence', 'risk_level']].to_string(index=False))
                        else:
                            print("❌ Could not generate signals")
                    else:
                        print("❌ Could not generate rankings")
                except Exception as e:
                    print(f"❌ Error: {e}")
                    import traceback
                    traceback.print_exc()
                continue

            elif query.lower().startswith('rrs-signals'):
                # RRS Trading Signals
                parts = query.split()
                symbol_type = parts[1] if len(parts) > 1 else 'STRONG_BUY'
                symbols_str = parts[2] if len(parts) > 2 else 'BTCUSDT,ETHUSDT,BNBUSDT,SOLUSDT'
                timeframe = parts[3] if len(parts) > 3 else '1h'
                symbols = [s.strip() for s in symbols_str.split(',')]
                print(f"\n🎯 Generating {symbol_type} signals from RRS analysis...")
                try:
                    from gordon.research.rrs import RRSManager
                    from gordon.backtesting.data.fetcher import DataFetcher
                    import yaml
                    config_path = Path(__file__).parent.parent / 'config.yaml'
                    config = {}
                    if config_path.exists():
                        with open(config_path, 'r') as f:
                            config = yaml.safe_load(f)
                    data_fetcher = DataFetcher()
                    rrs_manager = RRSManager(data_fetcher=data_fetcher, config=config.get('rrs', {}))
                    rrs_results = rrs_manager.analyze_multiple_symbols(symbols, timeframe=timeframe)
                    if rrs_results:
                        signals_df = rrs_manager.generate_rankings_and_signals(rrs_results, timeframe)
                        if not signals_df.empty:
                            top_signals = rrs_manager.signal_generator.get_top_signals(signals_df, symbol_type, top_n=10)
                            if not top_signals.empty:
                                print(f"\n✅ Top {symbol_type} Signals:")
                                print(top_signals[['symbol', 'current_rrs', 'signal_confidence', 'risk_level', 'volume_confirmation']].to_string(index=False))
                            else:
                                print(f"⚠️ No {symbol_type} signals found")
                        else:
                            print("❌ Could not generate signals")
                    else:
                        print("❌ Could not generate rankings")
                except Exception as e:
                    print(f"❌ Error: {e}")
                    import traceback
                    traceback.print_exc()
                continue

            elif query.lower().startswith('trader-analyze'):
                # Trader Intelligence Analysis
                parts = query.split()
                symbol = parts[1] if len(parts) > 1 else 'BTCUSDT'
                lookback_days = int(parts[2]) if len(parts) > 2 else 30
                print(f"\n🔍 Analyzing trader intelligence for {symbol} (last {lookback_days} days)...")
                try:
                    from gordon.research.trader_intelligence import TraderIntelligenceManager
                    from gordon.exchanges.factory import ExchangeFactory
                    import yaml
                    config_path = Path(__file__).parent.parent / 'config.yaml'
                    config = {}
                    if config_path.exists():
                        with open(config_path, 'r') as f:
                            config = yaml.safe_load(f)
                    
                    # Get exchange adapter
                    exchange_config = config.get('exchanges', {}).get('binance', {})
                    exchange_adapter = ExchangeFactory.create_exchange(
                        'binance',
                        exchange_config,
                        event_bus=None
                    )
                    
                    manager = TraderIntelligenceManager(
                        exchange_adapter=exchange_adapter,
                        config=config.get('trader_intelligence', {})
                    )
                    
                    results = manager.analyze_symbol(symbol, lookback_days=lookback_days)
                    
                    if not results['trader_profiles'].empty:
                        print(f"\n📊 Trader Intelligence Summary:")
                        print(f"  Total Trades: {len(results['trades'])}")
                        print(f"  Unique Traders: {len(results['trader_profiles'])}")
                        print(f"  Institutional/Whale Trades: {results['classified_trades']['is_professional'].sum()}")
                        print(f"\n🏆 Top 10 Traders by Volume:")
                        print(results['top_traders'].head(10)[['rank', 'trader', 'total_volume', 'trade_count', 'avg_trade_size']].to_string(index=False))
                        
                        if not results['early_buyers'].empty:
                            print(f"\n👥 Top 10 Early Buyers:")
                            print(results['early_buyers'].head(10)[['trader', 'timestamp', 'usd_value']].to_string(index=False))
                    else:
                        print("❌ No trader data found")
                except Exception as e:
                    print(f"❌ Error: {e}")
                    import traceback
                    traceback.print_exc()
                continue

            elif query.lower().startswith('find-accounts'):
                # Find Accounts to Follow
                parts = query.split()
                symbol = parts[1] if len(parts) > 1 else 'BTCUSDT'
                max_accounts = int(parts[2]) if len(parts) > 2 else 20
                print(f"\n🔍 Finding accounts to follow for {symbol}...")
                try:
                    from gordon.research.trader_intelligence import TraderIntelligenceManager
                    from gordon.exchanges.factory import ExchangeFactory
                    import yaml
                    config_path = Path(__file__).parent.parent / 'config.yaml'
                    config = {}
                    if config_path.exists():
                        with open(config_path, 'r') as f:
                            config = yaml.safe_load(f)
                    
                    exchange_config = config.get('exchanges', {}).get('binance', {})
                    exchange_adapter = ExchangeFactory.create_exchange(
                        'binance',
                        exchange_config,
                        event_bus=None
                    )
                    
                    manager = TraderIntelligenceManager(
                        exchange_adapter=exchange_adapter,
                        config=config.get('trader_intelligence', {})
                    )
                    
                    accounts = manager.get_accounts_to_follow(symbol, max_accounts=max_accounts)
                    
                    if accounts:
                        print(f"\n✅ Found {len(accounts)} accounts to follow:")
                        for i, account in enumerate(accounts[:10], 1):
                            print(f"  {i}. {account}")
                        print(f"\n💡 Use these accounts in social_trading.target_accounts config")
                    else:
                        print("⚠️ No accounts found matching criteria")
                except Exception as e:
                    print(f"❌ Error: {e}")
                    import traceback
                    traceback.print_exc()
                continue

            elif query.lower().startswith('institutional-traders'):
                # Institutional Traders Analysis
                parts = query.split()
                symbol = parts[1] if len(parts) > 1 else 'BTCUSDT'
                print(f"\n🏛️ Analyzing institutional traders for {symbol}...")
                try:
                    from gordon.research.trader_intelligence import TraderIntelligenceManager
                    from gordon.exchanges.factory import ExchangeFactory
                    import yaml
                    config_path = Path(__file__).parent.parent / 'config.yaml'
                    config = {}
                    if config_path.exists():
                        with open(config_path, 'r') as f:
                            config = yaml.safe_load(f)
                    
                    exchange_config = config.get('exchanges', {}).get('binance', {})
                    exchange_adapter = ExchangeFactory.create_exchange(
                        'binance',
                        exchange_config,
                        event_bus=None
                    )
                    
                    manager = TraderIntelligenceManager(
                        exchange_adapter=exchange_adapter,
                        config=config.get('trader_intelligence', {})
                    )
                    
                    results = manager.analyze_symbol(symbol, lookback_days=30)
                    
                    if not results['classified_trades'].empty:
                        institutional = results['classified_trades'][
                            results['classified_trades']['is_professional']
                        ]
                        
                        if not institutional.empty:
                            print(f"\n🏛️ Institutional/Whale Trades: {len(institutional)}")
                            print(f"💰 Total Institutional Volume: ${institutional['usd_value'].sum():,.2f}")
                            print(f"\n📊 Top Institutional Traders:")
                            trader_profiles = results['trader_profiles']
                            institutional_profiles = trader_profiles[
                                trader_profiles['trader_classification'].isin(['Institutional', 'Whale'])
                            ]
                            print(institutional_profiles.head(10)[['rank', 'trader', 'trader_classification', 'total_volume', 'trade_count']].to_string(index=False))
                        else:
                            print("⚠️ No institutional trades found")
                    else:
                        print("❌ No trade data found")
                except Exception as e:
                    print(f"❌ Error: {e}")
                    import traceback
                    traceback.print_exc()
                continue

            elif query.lower().startswith('whale-track'):
                # Whale Position Tracking
                parts = query.split()
                symbol = parts[1] if len(parts) > 1 else None
                min_value = float(parts[2]) if len(parts) > 2 else None
                print(f"\n🐋 Tracking whale positions{f' for {symbol}' if symbol else ''}...")
                try:
                    from gordon.core.utilities import WhaleTrackingManager
                    from gordon.exchanges.factory import ExchangeFactory
                    import yaml
                    import asyncio
                    config_path = Path(__file__).parent.parent / 'config.yaml'
                    config = {}
                    if config_path.exists():
                        with open(config_path, 'r') as f:
                            config = yaml.safe_load(f)
                    
                    exchange_config = config.get('exchanges', {}).get('binance', {})
                    exchange_adapter = ExchangeFactory.create_exchange(
                        'binance',
                        exchange_config,
                        event_bus=None
                    )
                    
                    manager = WhaleTrackingManager(
                        exchange_adapter=exchange_adapter,
                        config=config.get('whale_tracking', {})
                    )
                    
                    async def track():
                        results = await manager.track_whales(symbol=symbol, min_value_usd=min_value)
                        report = manager.get_whale_summary_report(results)
                        print(f"\n{report}")
                        
                        if not results['top_positions'].empty:
                            print(f"\n📊 Top Positions:")
                            print(results['top_positions'][['symbol', 'position_value_usd', 'pnl_percent', 'whale_tier']].head(10).to_string(index=False))
                    
                    asyncio.run(track())
                except Exception as e:
                    print(f"❌ Error: {e}")
                    import traceback
                    traceback.print_exc()
                continue

            elif query.lower().startswith('multi-whale-track') or query.lower().startswith('multi-address-track'):
                # Day 46: Multi-address whale tracking
                parts = query.split()
                symbol = parts[1] if len(parts) > 1 else None
                min_value = float(parts[2]) if len(parts) > 2 else None
                print(f"\n🐋🐋 Multi-Address Whale Tracking{f' for {symbol}' if symbol else ''}...")
                try:
                    from gordon.core.utilities import WhaleTrackingManager
                    from gordon.exchanges.factory import ExchangeFactory
                    import yaml
                    import asyncio
                    config_path = Path(__file__).parent.parent / 'config.yaml'
                    config = {}
                    if config_path.exists():
                        with open(config_path, 'r') as f:
                            config = yaml.safe_load(f)
                    
                    exchange_config = config.get('exchanges', {}).get('binance', {})
                    exchange_adapter = ExchangeFactory.create_exchange(
                        'binance',
                        exchange_config,
                        event_bus=None
                    )
                    
                    whale_config = config.get('whale_tracking', {})
                    manager = WhaleTrackingManager(
                        exchange_adapter=exchange_adapter,
                        config=whale_config
                    )
                    
                    async def track_multi():
                        results = await manager.track_multi_address_whales(
                            symbol=symbol,
                            min_value_usd=min_value,
                            export_csv=True
                        )
                        
                        if results['positions'].empty:
                            print("❌ No positions found for tracked addresses")
                            return
                        
                        print(f"\n✅ Found {len(results['positions'])} positions from {len(results['positions']['address'].unique())} addresses")
                        
                        # Show aggregated report
                        agg_report = manager.get_aggregated_positions_report(results['positions'])
                        print(f"\n{agg_report}")
                        
                        # Show liquidation risk if available
                        if not results['liquidation_risk'].empty:
                            liq_report = await manager.get_liquidation_risk_report(
                                results['positions'],
                                threshold_pct=whale_config.get('multi_address_tracking', {}).get('liquidation_risk_threshold', 3.0)
                            )
                            print(f"\n{liq_report}")
                        
                        # Show saved files
                        saved_files = results.get('saved_files', {})
                        if saved_files:
                            print(f"\n💾 Saved CSV files:")
                            for key, path in saved_files.items():
                                if path:
                                    print(f"   - {key}: {path}")
                    
                    asyncio.run(track_multi())
                except Exception as e:
                    print(f"❌ Error: {e}")
                    import traceback
                    traceback.print_exc()
                continue

            elif query.lower().startswith('liquidation-risk') or query.lower().startswith('liq-risk'):
                # Day 46: Liquidation risk analysis
                parts = query.split()
                symbol = parts[1] if len(parts) > 1 else None
                threshold = float(parts[2]) if len(parts) > 2 else 3.0
                print(f"\n💥 Liquidation Risk Analysis (threshold: {threshold}%)...")
                try:
                    from gordon.core.utilities import WhaleTrackingManager
                    from gordon.exchanges.factory import ExchangeFactory
                    import yaml
                    import asyncio
                    config_path = Path(__file__).parent.parent / 'config.yaml'
                    config = {}
                    if config_path.exists():
                        with open(config_path, 'r') as f:
                            config = yaml.safe_load(f)
                    
                    exchange_config = config.get('exchanges', {}).get('binance', {})
                    exchange_adapter = ExchangeFactory.create_exchange(
                        'binance',
                        exchange_config,
                        event_bus=None
                    )
                    
                    manager = WhaleTrackingManager(
                        exchange_adapter=exchange_adapter,
                        config=config.get('whale_tracking', {})
                    )
                    
                    async def analyze_risk():
                        # First get positions (from multi-address tracking or single tracking)
                        results = await manager.track_multi_address_whales(
                            symbol=symbol,
                            export_csv=False
                        )
                        
                        positions_df = results.get('positions', pd.DataFrame())
                        if positions_df.empty:
                            # Fallback to single tracking
                            single_results = await manager.track_whales(symbol=symbol)
                            positions_df = single_results.get('whale_positions', pd.DataFrame())
                        
                        if positions_df.empty:
                            print("❌ No positions found for liquidation risk analysis")
                            return
                        
                        report = await manager.get_liquidation_risk_report(
                            positions_df,
                            threshold_pct=threshold,
                            top_n=20
                        )
                        print(f"\n{report}")
                    
                    asyncio.run(analyze_risk())
                except Exception as e:
                    print(f"❌ Error: {e}")
                    import traceback
                    traceback.print_exc()
                continue

            elif query.lower().startswith('aggregate-positions') or query.lower().startswith('agg-positions'):
                # Day 46: Aggregated positions report
                parts = query.split()
                symbol = parts[1] if len(parts) > 1 else None
                print(f"\n📊 Aggregated Positions Report{f' for {symbol}' if symbol else ''}...")
                try:
                    from gordon.core.utilities import WhaleTrackingManager
                    from gordon.exchanges.factory import ExchangeFactory
                    import yaml
                    import asyncio
                    config_path = Path(__file__).parent.parent / 'config.yaml'
                    config = {}
                    if config_path.exists():
                        with open(config_path, 'r') as f:
                            config = yaml.safe_load(f)
                    
                    exchange_config = config.get('exchanges', {}).get('binance', {})
                    exchange_adapter = ExchangeFactory.create_exchange(
                        'binance',
                        exchange_config,
                        event_bus=None
                    )
                    
                    manager = WhaleTrackingManager(
                        exchange_adapter=exchange_adapter,
                        config=config.get('whale_tracking', {})
                    )
                    
                    async def aggregate():
                        results = await manager.track_multi_address_whales(
                            symbol=symbol,
                            export_csv=False
                        )
                        
                        positions_df = results.get('positions', pd.DataFrame())
                        if positions_df.empty:
                            single_results = await manager.track_whales(symbol=symbol)
                            positions_df = single_results.get('whale_positions', pd.DataFrame())
                        
                        if positions_df.empty:
                            print("❌ No positions found for aggregation")
                            return
                        
                        report = manager.get_aggregated_positions_report(positions_df)
                        print(f"\n{report}")
                    
                    asyncio.run(aggregate())
                except Exception as e:
                    print(f"❌ Error: {e}")
                    import traceback
                    traceback.print_exc()
                continue

            elif query.lower().startswith('market-dashboard') or query.lower().startswith('market-tracker'):
                # Day 47: Market Dashboard
                parts = query.split()
                exchange = parts[1].lower() if len(parts) > 1 else 'binance'
                print(f"\n📊 Market Dashboard for {exchange.upper()}...")
                try:
                    from gordon.market import MarketDashboard
                    from gordon.exchanges.factory import ExchangeFactory
                    import yaml
                    import asyncio
                    config_path = Path(__file__).parent.parent / 'config.yaml'
                    config = {}
                    if config_path.exists():
                        with open(config_path, 'r') as f:
                            config = yaml.safe_load(f)
                    
                    exchange_config = config.get('exchanges', {}).get(exchange, {})
                    exchange_adapter = ExchangeFactory.create_exchange(
                        exchange,
                        exchange_config,
                        event_bus=None
                    )
                    
                    # Initialize exchange adapter
                    async def init_and_run():
                        await exchange_adapter.initialize()
                        
                        # Get market dashboard config
                        market_config = config.get('market_dashboard', {}).get(exchange, {})
                        dashboard = MarketDashboard(
                            exchange_adapter=exchange_adapter,
                            exchange_name=exchange,
                            config=market_config
                        )
                        
                        results = await dashboard.run_full_analysis(export_csv=True)
                        dashboard.display_results(results)
                        
                        summary = dashboard.get_summary_report(results)
                        print(f"\n{summary}")
                    
                    asyncio.run(init_and_run())
                except Exception as e:
                    print(f"❌ Error: {e}")
                    import traceback
                    traceback.print_exc()
                continue

            elif query.lower().startswith('trending-tokens') or query.lower().startswith('trending'):
                # Day 47: Trending Tokens
                parts = query.split()
                exchange = parts[1].lower() if len(parts) > 1 else 'binance'
                print(f"\n🚀 Fetching trending tokens from {exchange.upper()}...")
                try:
                    from gordon.market import MarketDashboard
                    from gordon.exchanges.factory import ExchangeFactory
                    import yaml
                    import asyncio
                    config_path = Path(__file__).parent.parent / 'config.yaml'
                    config = {}
                    if config_path.exists():
                        with open(config_path, 'r') as f:
                            config = yaml.safe_load(f)
                    
                    exchange_config = config.get('exchanges', {}).get(exchange, {})
                    exchange_adapter = ExchangeFactory.create_exchange(
                        exchange,
                        exchange_config,
                        event_bus=None
                    )
                    
                    async def init_and_get_trending():
                        await exchange_adapter.initialize()
                        
                        market_config = config.get('market_dashboard', {}).get(exchange, {})
                        dashboard = MarketDashboard(
                            exchange_adapter=exchange_adapter,
                            exchange_name=exchange,
                            config=market_config
                        )
                        
                        trending_df = await dashboard.market_client.fetch_trending_tokens()
                        if not trending_df.empty:
                            dashboard.display.display_trending_tokens(trending_df, exchange.upper())
                        else:
                            print("❌ No trending tokens found")
                    
                    asyncio.run(init_and_get_trending())
                except Exception as e:
                    print(f"❌ Error: {e}")
                    import traceback
                    traceback.print_exc()
                continue

            elif query.lower().startswith('new-listings') or query.lower().startswith('new-tokens'):
                # Day 47: New Listings
                parts = query.split()
                exchange = parts[1].lower() if len(parts) > 1 else 'binance'
                print(f"\n🌟 Fetching new listings from {exchange.upper()}...")
                try:
                    from gordon.market import MarketDashboard
                    from gordon.exchanges.factory import ExchangeFactory
                    import yaml
                    import asyncio
                    config_path = Path(__file__).parent.parent / 'config.yaml'
                    config = {}
                    if config_path.exists():
                        with open(config_path, 'r') as f:
                            config = yaml.safe_load(f)
                    
                    exchange_config = config.get('exchanges', {}).get(exchange, {})
                    exchange_adapter = ExchangeFactory.create_exchange(
                        exchange,
                        exchange_config,
                        event_bus=None
                    )
                    
                    async def init_and_get_listings():
                        await exchange_adapter.initialize()
                        
                        market_config = config.get('market_dashboard', {}).get(exchange, {})
                        dashboard = MarketDashboard(
                            exchange_adapter=exchange_adapter,
                            exchange_name=exchange,
                            config=market_config
                        )
                        
                        listings_df = await dashboard.market_client.fetch_new_listings()
                        if not listings_df.empty:
                            dashboard.display.display_new_listings(listings_df, exchange.upper())
                        else:
                            print("❌ No new listings found")
                    
                    asyncio.run(init_and_get_listings())
                except Exception as e:
                    print(f"❌ Error: {e}")
                    import traceback
                    traceback.print_exc()
                continue

            elif query.lower().startswith('volume-leaders') or query.lower().startswith('volume'):
                # Day 47: Volume Leaders
                parts = query.split()
                exchange = parts[1].lower() if len(parts) > 1 else 'binance'
                print(f"\n📊 Fetching volume leaders from {exchange.upper()}...")
                try:
                    from gordon.market import MarketDashboard
                    from gordon.exchanges.factory import ExchangeFactory
                    import yaml
                    import asyncio
                    config_path = Path(__file__).parent.parent / 'config.yaml'
                    config = {}
                    if config_path.exists():
                        with open(config_path, 'r') as f:
                            config = yaml.safe_load(f)
                    
                    exchange_config = config.get('exchanges', {}).get(exchange, {})
                    exchange_adapter = ExchangeFactory.create_exchange(
                        exchange,
                        exchange_config,
                        event_bus=None
                    )
                    
                    async def init_and_get_volume():
                        await exchange_adapter.initialize()
                        
                        market_config = config.get('market_dashboard', {}).get(exchange, {})
                        dashboard = MarketDashboard(
                            exchange_adapter=exchange_adapter,
                            exchange_name=exchange,
                            config=market_config
                        )
                        
                        volume_df = await dashboard.market_client.fetch_high_volume_tokens()
                        if not volume_df.empty:
                            dashboard.display.display_volume_leaders(volume_df, exchange.upper())
                        else:
                            print("❌ No volume data found")
                    
                    asyncio.run(init_and_get_volume())
                except Exception as e:
                    print(f"❌ Error: {e}")
                    import traceback
                    traceback.print_exc()
                continue

            elif query.lower().startswith('funding-rates') or query.lower().startswith('funding'):
                # Day 47: Funding Rates (Bitfinex only)
                print(f"\n💰 Fetching funding rates from Bitfinex...")
                try:
                    from gordon.market import MarketDashboard, FundingRateAnalyzer
                    from gordon.exchanges.factory import ExchangeFactory
                    import yaml
                    import asyncio
                    config_path = Path(__file__).parent.parent / 'config.yaml'
                    config = {}
                    if config_path.exists():
                        with open(config_path, 'r') as f:
                            config = yaml.safe_load(f)
                    
                    exchange_config = config.get('exchanges', {}).get('bitfinex', {})
                    exchange_adapter = ExchangeFactory.create_exchange(
                        'bitfinex',
                        exchange_config,
                        event_bus=None
                    )
                    
                    async def init_and_get_funding():
                        await exchange_adapter.initialize()
                        
                        market_config = config.get('market_dashboard', {}).get('bitfinex', {})
                        analyzer = FundingRateAnalyzer(
                            exchange_adapter=exchange_adapter,
                            config=market_config
                        )
                        
                        funding_df = await analyzer.fetch_funding_rates()
                        if not funding_df.empty:
                            analyzer.display_funding_rates(funding_df)
                            
                            # Show arbitrage opportunities
                            arbitrage_df = analyzer.analyze_arbitrage_opportunities(funding_df)
                            if not arbitrage_df.empty:
                                print(f"\n🎯 Found {len(arbitrage_df)} arbitrage opportunities!")
                                print(arbitrage_df[['symbol', 'base_symbol', 'funding_rate', 'opportunity_score']].head(10).to_string(index=False))
                        else:
                            print("❌ No funding rate data found")
                    
                    asyncio.run(init_and_get_funding())
                except Exception as e:
                    print(f"❌ Error: {e}")
                    import traceback
                    traceback.print_exc()
                continue

            elif query.lower().startswith('pnl-close') or query.lower().startswith('pnl-close-position'):
                # Day 48: PnL-based position closing
                parts = query.split()
                symbol = parts[1].upper() if len(parts) > 1 else None
                if not symbol:
                    print("Usage: pnl-close SYMBOL")
                    continue
                print(f"\n💰 PnL-based Position Closing for {symbol}...")
                try:
                    from gordon.core.utilities import PnLPositionManager
                    from gordon.exchanges.factory import ExchangeFactory
                    import yaml
                    import asyncio
                    config_path = Path(__file__).parent.parent / 'config.yaml'
                    config = {}
                    if config_path.exists():
                        with open(config_path, 'r') as f:
                            config = yaml.safe_load(f)
                    
                    exchange_config = config.get('exchanges', {}).get('binance', {})
                    exchange_adapter = ExchangeFactory.create_exchange(
                        'binance',
                        exchange_config,
                        event_bus=None
                    )
                    
                    async def close_pnl():
                        await exchange_adapter.initialize()
                        
                        pnl_config = config.get('position_management', {}).get('pnl_manager', {})
                        pnl_manager = PnLPositionManager(
                            exchange_adapter=exchange_adapter,
                            config=pnl_config
                        )
                        
                        # Get current price
                        ticker = await exchange_adapter.get_ticker(symbol)
                        if not ticker:
                            print(f"❌ Could not get price for {symbol}")
                            return
                        
                        current_price = float(ticker.get('last', 0))
                        
                        # Check if position should be closed
                        close_info = pnl_manager.check_and_close_position(symbol, current_price)
                        
                        if close_info and close_info.get('should_close'):
                            print(f"✅ {close_info['reason']}")
                            print(f"   PnL: {close_info['pnl_pct']:.2f}%")
                            print(f"   Closing position...")
                            
                            success = await pnl_manager.close_position(symbol)
                            if success:
                                print(f"✅ Position closed successfully!")
                            else:
                                print(f"❌ Failed to close position")
                        else:
                            if close_info:
                                print(f"📊 Current PnL: {close_info['pnl_pct']:.2f}%")
                                print(f"   Take Profit: {close_info.get('take_profit_pct', 0):.2f}%")
                                print(f"   Stop Loss: -{close_info.get('stop_loss_pct', 0):.2f}%")
                                print(f"   Status: Position not at threshold")
                            else:
                                print(f"❌ No tracked position for {symbol}")
                    
                    asyncio.run(close_pnl())
                except Exception as e:
                    print(f"❌ Error: {e}")
                    import traceback
                    traceback.print_exc()
                continue

            elif query.lower().startswith('chunk-close') or query.lower().startswith('chunk-close-position'):
                # Day 48: Chunk position closing
                parts = query.split()
                symbol = parts[1].upper() if len(parts) > 1 else None
                max_size = float(parts[2]) if len(parts) > 2 else None
                if not symbol:
                    print("Usage: chunk-close SYMBOL [MAX_SIZE_USD]")
                    continue
                print(f"\n📦 Chunk Position Closing for {symbol}...")
                try:
                    from gordon.core.utilities import ChunkPositionCloser
                    from gordon.exchanges.factory import ExchangeFactory
                    import yaml
                    import asyncio
                    config_path = Path(__file__).parent.parent / 'config.yaml'
                    config = {}
                    if config_path.exists():
                        with open(config_path, 'r') as f:
                            config = yaml.safe_load(f)
                    
                    exchange_config = config.get('exchanges', {}).get('binance', {})
                    exchange_adapter = ExchangeFactory.create_exchange(
                        'binance',
                        exchange_config,
                        event_bus=None
                    )
                    
                    async def chunk_close():
                        await exchange_adapter.initialize()
                        
                        chunk_config = config.get('position_management', {}).get('chunk_closer', {})
                        closer = ChunkPositionCloser(
                            exchange_adapter=exchange_adapter,
                            config=chunk_config
                        )
                        
                        # Get position
                        position = await exchange_adapter.get_position(symbol)
                        if not position or abs(float(position.get('quantity', 0))) < 0.0001:
                            print(f"❌ No position to close for {symbol}")
                            return
                        
                        quantity = float(position['quantity'])
                        
                        result = await closer.close_position_in_chunks(
                            symbol=symbol,
                            total_quantity=quantity,
                            max_order_size_usd=max_size
                        )
                        
                        if result.get('success'):
                            print(f"✅ Closed {result['successful_chunks']}/{result['total_chunks']} chunks")
                            print(f"   Total closed: ${result['total_closed_usd']:.2f}")
                        else:
                            print(f"❌ Chunk closing failed")
                            if result.get('error'):
                                print(f"   Error: {result['error']}")
                    
                    asyncio.run(chunk_close())
                except Exception as e:
                    print(f"❌ Error: {e}")
                    import traceback
                    traceback.print_exc()
                continue

            elif query.lower().startswith('easy-entry') or query.lower().startswith('easy-entry-strategy'):
                # Day 48: Easy Entry Strategy
                parts = query.split()
                symbol = parts[1].upper() if len(parts) > 1 else None
                entry_type = parts[2].lower() if len(parts) > 2 else 'market'
                target_size = float(parts[3]) if len(parts) > 3 else None
                
                if not symbol:
                    print("Usage: easy-entry SYMBOL [market|sd-zone|sma-trend] [TARGET_SIZE_USD]")
                    continue
                
                print(f"\n🎯 Easy Entry Strategy for {symbol} ({entry_type})...")
                try:
                    from gordon.core.strategies import EasyEntryStrategy
                    from gordon.exchanges.factory import ExchangeFactory
                    import yaml
                    import asyncio
                    config_path = Path(__file__).parent.parent / 'config.yaml'
                    config = {}
                    if config_path.exists():
                        with open(config_path, 'r') as f:
                            config = yaml.safe_load(f)
                    
                    exchange_config = config.get('exchanges', {}).get('binance', {})
                    exchange_adapter = ExchangeFactory.create_exchange(
                        'binance',
                        exchange_config,
                        event_bus=None
                    )
                    
                    async def run_entry():
                        await exchange_adapter.initialize()
                        
                        entry_config = config.get('position_management', {}).get('easy_entry', {})
                        strategy = EasyEntryStrategy(
                            exchange_adapter=exchange_adapter,
                            config=entry_config
                        )
                        
                        target_position_usd = target_size or entry_config.get('default_total_position_size_usd', 1000.0)
                        
                        if entry_type == 'market':
                            result = await strategy.market_buy_loop(
                                symbol=symbol,
                                target_position_usd=target_position_usd
                            )
                            if result.get('success'):
                                print(f"✅ Market buy loop completed")
                                print(f"   Final position: ${result['final_position_usd']:.2f}")
                                print(f"   Successful attempts: {result['successful_attempts']}")
                            else:
                                print(f"❌ Market buy loop failed")
                        
                        elif entry_type == 'sd-zone' or entry_type == 'supply-demand':
                            result = await strategy.supply_demand_zone_entry(
                                symbol=symbol,
                                target_position_usd=target_position_usd
                            )
                            if result.get('success'):
                                print(f"✅ Supply/demand zone entry executed")
                                print(f"   Buy amount: ${result.get('buy_amount_usd', 0):.2f}")
                            else:
                                print(f"❌ {result.get('message', result.get('error', 'Entry failed'))}")
                                if result.get('support_distance'):
                                    print(f"   Support distance: {result['support_distance']:.2f}%")
                        
                        elif entry_type == 'sma-trend' or entry_type == 'trend':
                            result = await strategy.sma_trend_entry(
                                symbol=symbol,
                                target_position_usd=target_position_usd
                            )
                            if result.get('success'):
                                print(f"✅ SMA trend entry executed")
                                print(f"   Buy amount: ${result.get('buy_amount_usd', 0):.2f}")
                            else:
                                print(f"❌ {result.get('message', result.get('error', 'Entry failed'))}")
                        else:
                            print(f"❌ Unknown entry type: {entry_type}")
                            print("   Valid types: market, sd-zone, sma-trend")
                    
                    asyncio.run(run_entry())
                except Exception as e:
                    print(f"❌ Error: {e}")
                    import traceback
                    traceback.print_exc()
                continue

            elif query.lower().startswith('track-position') or query.lower().startswith('track-pnl'):
                # Day 48: Track position for PnL management
                parts = query.split()
                symbol = parts[1].upper() if len(parts) > 1 else None
                entry_price = float(parts[2]) if len(parts) > 2 else None
                quantity = float(parts[3]) if len(parts) > 3 else None
                tp_pct = float(parts[4]) if len(parts) > 4 else None
                sl_pct = float(parts[5]) if len(parts) > 5 else None
                
                if not symbol or entry_price is None or quantity is None:
                    print("Usage: track-position SYMBOL ENTRY_PRICE QUANTITY [TAKE_PROFIT%] [STOP_LOSS%]")
                    continue
                
                print(f"\n📊 Tracking position: {symbol} @ ${entry_price:.2f} (Qty: {quantity})...")
                try:
                    from gordon.core.utilities import PnLPositionManager
                    from gordon.exchanges.factory import ExchangeFactory
                    import yaml
                    config_path = Path(__file__).parent.parent / 'config.yaml'
                    config = {}
                    if config_path.exists():
                        with open(config_path, 'r') as f:
                            config = yaml.safe_load(f)
                    
                    pnl_config = config.get('position_management', {}).get('pnl_manager', {})
                    pnl_manager = PnLPositionManager(
                        exchange_adapter=None,  # Not needed for tracking
                        config=pnl_config
                    )
                    
                    pnl_manager.track_position(
                        symbol=symbol,
                        entry_price=entry_price,
                        quantity=quantity,
                        take_profit_pct=tp_pct,
                        stop_loss_pct=sl_pct
                    )
                    
                    print(f"✅ Position tracking started")
                    print(f"   Take Profit: {tp_pct or pnl_config.get('default_take_profit_pct', 10.0):.2f}%")
                    print(f"   Stop Loss: {sl_pct or pnl_config.get('default_stop_loss_pct', 5.0):.2f}%")
                except Exception as e:
                    print(f"❌ Error: {e}")
                    import traceback
                    traceback.print_exc()
                continue

            elif query.lower().startswith('check-pnl') or query.lower().startswith('pnl-status'):
                # Day 48: Check PnL status
                parts = query.split()
                symbol = parts[1].upper() if len(parts) > 1 else None
                if not symbol:
                    print("Usage: check-pnl SYMBOL")
                    continue
                print(f"\n💰 Checking PnL for {symbol}...")
                try:
                    from gordon.core.utilities import PnLPositionManager
                    from gordon.exchanges.factory import ExchangeFactory
                    import yaml
                    import asyncio
                    config_path = Path(__file__).parent.parent / 'config.yaml'
                    config = {}
                    if config_path.exists():
                        with open(config_path, 'r') as f:
                            config = yaml.safe_load(f)
                    
                    exchange_config = config.get('exchanges', {}).get('binance', {})
                    exchange_adapter = ExchangeFactory.create_exchange(
                        'binance',
                        exchange_config,
                        event_bus=None
                    )
                    
                    async def check_pnl():
                        await exchange_adapter.initialize()
                        
                        pnl_config = config.get('position_management', {}).get('pnl_manager', {})
                        pnl_manager = PnLPositionManager(
                            exchange_adapter=exchange_adapter,
                            config=pnl_config
                        )
                        
                        # Get current price
                        ticker = await exchange_adapter.get_ticker(symbol)
                        if not ticker:
                            print(f"❌ Could not get price for {symbol}")
                            return
                        
                        current_price = float(ticker.get('last', 0))
                        
                        # Get PnL
                        pnl_info = pnl_manager.get_position_pnl(symbol, current_price)
                        
                        if pnl_info:
                            print(f"📊 Position PnL for {symbol}:")
                            print(f"   Entry Price: ${pnl_info['entry_price']:.2f}")
                            print(f"   Current Price: ${pnl_info['current_price']:.2f}")
                            print(f"   Quantity: {pnl_info['quantity']:.6f}")
                            print(f"   PnL: {pnl_info['pnl_pct']:+.2f}% (${pnl_info['pnl_usd']:+.2f})")
                            print(f"   Position Value: ${pnl_info['position_value']:.2f}")
                            
                            # Check thresholds
                            close_info = pnl_manager.check_and_close_position(symbol, current_price)
                            if close_info and close_info.get('should_close'):
                                print(f"   ⚠️  {close_info['reason']}")
                            else:
                                position = pnl_manager.positions.get(symbol.upper(), {})
                                tp = position.get('take_profit_pct', pnl_config.get('default_take_profit_pct', 10.0))
                                sl = position.get('stop_loss_pct', pnl_config.get('default_stop_loss_pct', 5.0))
                                print(f"   Take Profit: +{tp:.2f}%")
                                print(f"   Stop Loss: -{sl:.2f}%")
                        else:
                            print(f"❌ No tracked position for {symbol}")
                    
                    asyncio.run(check_pnl())
                except Exception as e:
                    print(f"❌ Error: {e}")
                    import traceback
                    traceback.print_exc()
                continue

            elif query.lower().startswith('ma-reversal') or query.lower().startswith('2xma-reversal'):
                # Day 49: 2x MA Reversal Strategy
                parts = query.split()
                symbol = parts[1].upper() if len(parts) > 1 else None
                action = parts[2].lower() if len(parts) > 2 else 'execute'
                
                symbol = symbol or 'BTCUSDT'
                print(f"\n📊 2x MA Reversal Strategy for {symbol}...")
                try:
                    from gordon.core.strategies import MAReversalStrategy
                    from gordon.exchanges.factory import ExchangeFactory
                    import yaml
                    import asyncio
                    config_path = Path(__file__).parent.parent / 'config.yaml'
                    config = {}
                    if config_path.exists():
                        with open(config_path, 'r') as f:
                            config = yaml.safe_load(f)
                    
                    exchange_config = config.get('exchanges', {}).get('binance', {})
                    exchange_adapter = ExchangeFactory.create_exchange(
                        'binance',
                        exchange_config,
                        event_bus=None
                    )
                    
                    async def run_ma_reversal():
                        await exchange_adapter.initialize()
                        
                        ma_config = config.get('ma_reversal_strategy', {})
                        strategy = MAReversalStrategy(
                            exchange_adapter=exchange_adapter,
                            config=ma_config
                        )
                        strategy.symbol = symbol
                        
                        if action == 'execute':
                            result = await strategy.execute(symbol)
                            if result:
                                print(f"✅ Strategy executed: {result.get('action')}")
                                if result.get('signal'):
                                    signal = result['signal']
                                    print(f"   Signal: {signal.get('signal')}")
                                    print(f"   Price: ${signal.get('price', 0):.2f}")
                                    print(f"   Fast MA: ${signal.get('fast_ma', 0):.2f}")
                                    print(f"   Slow MA: ${signal.get('slow_ma', 0):.2f}")
                        elif action == 'check':
                            signal_info = await strategy.check_signals(symbol)
                            if signal_info:
                                print(f"📊 Signal: {signal_info['signal']}")
                                print(f"   Price: ${signal_info['price']:.2f}")
                                print(f"   Fast MA: ${signal_info['fast_ma']:.2f}")
                                print(f"   Slow MA: ${signal_info['slow_ma']:.2f}")
                            else:
                                print("❌ No signal at this time")
                    
                    asyncio.run(run_ma_reversal())
                except Exception as e:
                    print(f"❌ Error: {e}")
                    import traceback
                    traceback.print_exc()
                continue

            elif query.lower().startswith('enhanced-sd') or query.lower().startswith('enhanced-supply-demand'):
                # Day 50: Enhanced Supply/Demand Zone Strategy
                parts = query.split()
                symbol = parts[1].upper() if len(parts) > 1 else None
                
                if not symbol:
                    print("Usage: enhanced-sd SYMBOL")
                    continue
                
                print(f"\n🎯 Enhanced Supply/Demand Zone Strategy for {symbol}...")
                try:
                    from gordon.core.strategies import EnhancedSupplyDemandStrategy
                    from gordon.exchanges.factory import ExchangeFactory
                    import yaml
                    import asyncio
                    config_path = Path(__file__).parent.parent / 'config.yaml'
                    config = {}
                    if config_path.exists():
                        with open(config_path, 'r') as f:
                            config = yaml.safe_load(f)
                    
                    exchange_config = config.get('exchanges', {}).get('binance', {})
                    exchange_adapter = ExchangeFactory.create_exchange(
                        'binance',
                        exchange_config,
                        event_bus=None
                    )
                    
                    async def run_enhanced_sd():
                        await exchange_adapter.initialize()
                        
                        sd_config = config.get('enhanced_sd_strategy', {})
                        strategy = EnhancedSupplyDemandStrategy(
                            exchange_adapter=exchange_adapter,
                            config=sd_config
                        )
                        
                        result = await strategy.execute(symbol)
                        
                        if result:
                            action = result.get('action')
                            print(f"✅ Action: {action}")
                            
                            if action == 'BOUGHT':
                                print(f"   Bought: ${result.get('amount_usd', 0):.2f}")
                            elif action == 'SOLD':
                                print(f"   Sold: {result.get('sell_percentage', 0)*100:.0f}%")
                                print(f"   Verified: {'✅' if result.get('verified') else '❌'}")
                            elif action == 'NO_ZONE':
                                print(f"   Price: ${result.get('price', 0):.2f}")
                                print(f"   Not in any zone")
                            else:
                                print(f"   Message: {result.get('message', result.get('reason', 'N/A'))}")
                    
                    asyncio.run(run_enhanced_sd())
                except Exception as e:
                    print(f"❌ Error: {e}")
                    import traceback
                    traceback.print_exc()
                continue

            elif query.lower().startswith('quick-buy') or query.lower().startswith('qbuy'):
                # Day 51: Quick Buy
                parts = query.split()
                symbol = parts[1].upper() if len(parts) > 1 else None
                usd_amount = float(parts[2]) if len(parts) > 2 else None
                
                if not symbol:
                    print("Usage: quick-buy SYMBOL [USD_AMOUNT]")
                    print("Example: quick-buy BTCUSDT 10")
                    continue
                
                print(f"\n🟢 Quick Buy: {symbol}...")
                try:
                    from gordon.core.strategies import QuickBuySellStrategy
                    from gordon.exchanges.factory import ExchangeFactory
                    import yaml
                    import asyncio
                    config_path = Path(__file__).parent.parent / 'config.yaml'
                    config = {}
                    if config_path.exists():
                        with open(config_path, 'r') as f:
                            config = yaml.safe_load(f)
                    
                    qbs_config = config.get('quick_buysell', {})
                    exchange_name = qbs_config.get('exchange', 'binance')
                    exchange_config = config.get('exchanges', {}).get(exchange_name, {})
                    exchange_adapter = ExchangeFactory.create_exchange(
                        exchange_name,
                        exchange_config,
                        event_bus=None
                    )
                    
                    async def quick_buy():
                        await exchange_adapter.initialize()
                        
                        strategy = QuickBuySellStrategy(
                            exchange_adapter=exchange_adapter,
                            config=qbs_config
                        )
                        
                        amount = usd_amount or qbs_config.get('usd_size', 10.0)
                        await strategy.quick_buy_manual(symbol, amount)
                        print(f"✅ Quick buy executed for {symbol}")
                    
                    asyncio.run(quick_buy())
                except Exception as e:
                    print(f"❌ Error: {e}")
                    import traceback
                    traceback.print_exc()
                continue

            elif query.lower().startswith('quick-sell') or query.lower().startswith('qsell'):
                # Day 51: Quick Sell
                parts = query.split()
                symbol = parts[1].upper() if len(parts) > 1 else None
                
                if not symbol:
                    print("Usage: quick-sell SYMBOL")
                    print("Example: quick-sell BTCUSDT")
                    continue
                
                print(f"\n🔴 Quick Sell: {symbol}...")
                try:
                    from gordon.core.strategies import QuickBuySellStrategy
                    from gordon.exchanges.factory import ExchangeFactory
                    import yaml
                    import asyncio
                    config_path = Path(__file__).parent.parent / 'config.yaml'
                    config = {}
                    if config_path.exists():
                        with open(config_path, 'r') as f:
                            config = yaml.safe_load(f)
                    
                    qbs_config = config.get('quick_buysell', {})
                    exchange_name = qbs_config.get('exchange', 'binance')
                    exchange_config = config.get('exchanges', {}).get(exchange_name, {})
                    exchange_adapter = ExchangeFactory.create_exchange(
                        exchange_name,
                        exchange_config,
                        event_bus=None
                    )
                    
                    async def quick_sell():
                        await exchange_adapter.initialize()
                        
                        strategy = QuickBuySellStrategy(
                            exchange_adapter=exchange_adapter,
                            config=qbs_config
                        )
                        
                        await strategy.quick_sell_manual(symbol)
                        print(f"✅ Quick sell executed for {symbol}")
                    
                    asyncio.run(quick_sell())
                except Exception as e:
                    print(f"❌ Error: {e}")
                    import traceback
                    traceback.print_exc()
                continue

            elif query.lower().startswith('quick-monitor') or query.lower().startswith('qbs-monitor') or query.lower().startswith('start-quick-monitor'):
                # Day 51: Start Quick Buy/Sell Monitor
                print(f"\n📁 Starting Quick Buy/Sell file monitor...")
                try:
                    from gordon.core.strategies import QuickBuySellStrategy
                    from gordon.exchanges.factory import ExchangeFactory
                    import yaml
                    import asyncio
                    config_path = Path(__file__).parent.parent / 'config.yaml'
                    config = {}
                    if config_path.exists():
                        with open(config_path, 'r') as f:
                            config = yaml.safe_load(f)
                    
                    qbs_config = config.get('quick_buysell', {})
                    exchange_name = qbs_config.get('exchange', 'binance')
                    exchange_config = config.get('exchanges', {}).get(exchange_name, {})
                    exchange_adapter = ExchangeFactory.create_exchange(
                        exchange_name,
                        exchange_config,
                        event_bus=None
                    )
                    
                    async def start_monitor():
                        await exchange_adapter.initialize()
                        
                        strategy = QuickBuySellStrategy(
                            exchange_adapter=exchange_adapter,
                            config=qbs_config
                        )
                        
                        await strategy.initialize(qbs_config)
                        await strategy.start_monitoring()
                        
                        print(f"✅ Monitoring started for file: {qbs_config.get('token_file_path', './token_addresses.txt')}")
                        print(f"   To buy: Add '{exchange_name.upper()}_SYMBOL' to file")
                        print(f"   To sell: Add '{exchange_name.upper()}_SYMBOL x' to file")
                        print(f"   Press Ctrl+C to stop monitoring")
                        
                        # Keep running
                        try:
                            while True:
                                await asyncio.sleep(1)
                        except KeyboardInterrupt:
                            await strategy.stop_monitoring()
                            print("\n✅ Monitoring stopped")
                    
                    asyncio.run(start_monitor())
                except Exception as e:
                    print(f"❌ Error: {e}")
                    import traceback
                    traceback.print_exc()
                continue

            elif query.lower().startswith('add-to-quick-file') or query.lower().startswith('qbs-add'):
                # Day 51: Add token to monitoring file
                parts = query.split()
                symbol = parts[1].upper() if len(parts) > 1 else None
                command = parts[2].upper() if len(parts) > 2 else 'BUY'
                
                if not symbol:
                    print("Usage: add-to-quick-file SYMBOL [BUY|SELL]")
                    print("Example: add-to-quick-file BTCUSDT BUY")
                    continue
                
                print(f"\n📝 Adding {symbol} ({command}) to file...")
                try:
                    from gordon.core.utilities import add_token_to_file
                    import yaml
                    config_path = Path(__file__).parent.parent / 'config.yaml'
                    config = {}
                    if config_path.exists():
                        with open(config_path, 'r') as f:
                            config = yaml.safe_load(f)
                    
                    qbs_config = config.get('quick_buysell', {})
                    file_path = qbs_config.get('token_file_path', './token_addresses.txt')
                    
                    add_token_to_file(file_path, symbol, command)
                    print(f"✅ Added {symbol} ({command}) to {file_path}")
                    
                except Exception as e:
                    print(f"❌ Error: {e}")
                    import traceback
                    traceback.print_exc()
                continue

            elif query.lower().startswith('position-size'):
                # Position Sizing Calculator
                parts = query.split()
                method = parts[1] if len(parts) > 1 else 'balance'
                print(f"\n📏 Calculating position size ({method})...")
                try:
                    from gordon.core.utilities import PositionSizingHelper
                    from gordon.exchanges.factory import ExchangeFactory
                    import yaml
                    config_path = Path(__file__).parent.parent / 'config.yaml'
                    config = {}
                    if config_path.exists():
                        with open(config_path, 'r') as f:
                            config = yaml.safe_load(f)
                    
                    sizing = PositionSizingHelper(config.get('position_sizing', {}))
                    
                    if method == 'balance':
                        balance = float(parts[2]) if len(parts) > 2 else 10000
                        price = float(parts[3]) if len(parts) > 3 else 50000
                        leverage = int(parts[4]) if len(parts) > 4 else 1
                        leverage, size = sizing.calculate_size_from_balance_percent(balance, price, leverage)
                        print(f"\n✅ Position Size: {size} (Leverage: {leverage}x)")
                    elif method == 'usd':
                        amount = float(parts[2]) if len(parts) > 2 else 1000
                        price = float(parts[3]) if len(parts) > 3 else 50000
                        leverage = int(parts[4]) if len(parts) > 4 else 1
                        leverage, size = sizing.calculate_size_from_usd_amount(amount, price, leverage)
                        print(f"\n✅ Position Size: {size} (Leverage: {leverage}x)")
                    elif method == 'risk':
                        balance = float(parts[2]) if len(parts) > 2 else 10000
                        entry = float(parts[3]) if len(parts) > 3 else 50000
                        stop = float(parts[4]) if len(parts) > 4 else 49000
                        leverage = int(parts[5]) if len(parts) > 5 else 1
                        leverage, size = sizing.calculate_size_from_risk_percent(balance, entry, stop, leverage=leverage)
                        print(f"\n✅ Position Size: {size} (Leverage: {leverage}x)")
                    else:
                        print("Usage: position-size [balance|usd|risk] [args...]")
                except Exception as e:
                    print(f"❌ Error: {e}")
                    import traceback
                    traceback.print_exc()
                continue

            elif query.lower() == 'status':
                print("""
                ✅ System Status:
                • Research Agent: Online
                • Trading System: Online
                • Risk Manager: Active
                • Exchanges: Connected
                • Strategies: 10 loaded
                """)
                continue

            elif query.lower() == 'config':
                # Display configuration
                from gordon.agent.config_manager import get_config
                config = get_config()
                config.display()
                continue

            elif query.lower() == 'risk':
                # Display risk limits
                from gordon.agent.config_manager import get_config
                config = get_config()
                print("\n🛡️ Risk Management Settings:")
                print("=" * 50)
                print(f"• Max Position Size: {config.get('trading.risk.max_position_size', 0.1):.1%}")
                print(f"• Max Drawdown: {config.get('trading.risk.max_drawdown', 0.2):.1%}")
                print(f"• Daily Loss Limit: {config.get('trading.risk.daily_loss_limit', 0.05):.1%}")
                print(f"• Risk per Trade: {config.get('trading.risk.risk_per_trade', 0.02):.1%}")
                print(f"• Correlation Limit: {config.get('trading.risk.correlation_limit', 0.7):.1f}")
                print(f"• Dry Run Mode: {'✅ ON' if config.is_dry_run() else '❌ OFF'}")
                continue

            # Special hybrid analysis command
            elif query.lower().startswith('hybrid '):
                symbol = query[7:].strip().upper()
                print(f"\n🔬 Initiating hybrid analysis for {symbol}...")
                from gordon.agent.hybrid_analyzer import analyze
                result = analyze(symbol)

                # Display the analysis
                print("\n" + "=" * 60)
                print(f"📊 GORDON'S HYBRID ANALYSIS: {symbol}")
                print("=" * 60)

                # Trading recommendation
                if 'trading_signals' in result:
                    signals = result['trading_signals']
                    print(f"\n🎯 RECOMMENDATION: {signals.get('recommendation', 'HOLD')}")
                    print(f"📊 Confidence: {signals.get('confidence', 'LOW')}")
                    print(f"📈 Combined Score: {signals.get('combined_score', 50):.1f}/100")

                    if signals.get('strategies'):
                        print(f"🔧 Suggested Strategies: {', '.join(signals['strategies'])}")

                # Action items
                if 'action_items' in result:
                    print("\n📝 Action Items:")
                    for action in result['action_items']:
                        print(f"  {action}")

                print("\n" + "─" * 50)
                continue

            # Process regular queries - route everything through agent for LLM interpretation
            elif query:
                print("\n🔄 Processing...\n")
                
                # Use conversational assistant if enabled (better context)
                if conversational_assistant:
                    try:
                        response = asyncio.run(conversational_assistant.chat(query))
                        print("\n" + "🤖" + "="*60 + "🤖")
                        print("GORDON RESPONSE:")
                        print("="*70)
                        print(response)
                        print("="*70 + "\n")
                    except Exception as e:
                        print(f"\n❌ Error in conversational mode: {e}")
                        # Fallback to agent mode
                        try:
                            response = asyncio.run(gordon.run(query))
                            print("\n🤖 Gordon Response:")
                            print(response)
                        except Exception as e2:
                            print(f"\n❌ Error in agent mode: {e2}")
                            import traceback
                            traceback.print_exc()
                else:
                    # Use the Agent's run method (async, with LLM interpretation)
                    try:
                        response = asyncio.run(gordon.run(query))
                        print("\n🤖 Gordon Response:")
                        print(response)
                    except Exception as e:
                        print(f"\n❌ Error: {e}")
                        import traceback
                        traceback.print_exc()
                
                print("\n" + "─" * 50)

        except (KeyboardInterrupt, EOFError):
            print("\n\n👋 Goodbye! Happy trading! 📈\n")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}\n")


def run_single_query(gordon: Agent, query: str):
    """Run a single query and exit."""
    print(f"\n🔄 Processing: {query}\n")
    # Use the Agent's run method (async, with LLM interpretation)
    try:
        response = asyncio.run(gordon.run(query))
        print("\n🤖 Gordon Response:")
        print(response)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    print()


def show_dashboard():
    """Display a simple dashboard of all available functionality."""
    print("\n" + "=" * 60)
    print("GORDON - TRADING & RESEARCH AGENT DASHBOARD")
    print("=" * 60)

    print("\n📊 SUPPORTED EXCHANGES:")
    print("  • Binance")
    print("  • Bitfinex")
    print("  • HyperLiquid")
    print("  • Interactive Brokers (Day 43)")
    print("  • Polymarket (Day 56)")

    print("\n📈 AVAILABLE STRATEGIES (40+):")
    strategies_by_category = {
        "Technical Indicators": [
            "SMA (Day 6)", "RSI (Day 7)", "VWAP (Day 8)",
            "VWMA (Day 9)", "Bollinger Bands (Day 10)",
            "StochasticRSI (Day 16)"
        ],
        "Trading Patterns": [
            "Engulfing (Day 12)", "Gap Trading (Day 17)",
            "Breakout (Day 18)", "Mean Reversion (Day 20)"
        ],
        "Advanced Strategies": [
            "Liquidation Hunter (Day 21-22)",
            "Supply/Demand Zones (Day 11, 50)",
            "Market Making (Day 45)",
            "RRS Analysis (Day 37)",
            "Social Signal Trading (Day 36)"
        ],
        "Risk Management": [
            "Position Sizing (Day 5)",
            "Stop Loss/Take Profit (Day 4)",
            "Portfolio Risk (Day 56)"
        ],
        "ML & Research": [
            "ML Indicator Evaluation (Day 33)",
            "Twitter Sentiment (Day 28)",
            "Hybrid Analysis (Fundamental + Technical)"
        ]
    }

    for category, strategies in strategies_by_category.items():
        print(f"\n  {category}:")
        for strategy in strategies:
            print(f"    • {strategy}")

    print("\n🛠️ UTILITIES:")
    print("  • Master Utils (consolidated utilities)")
    print("  • Event Bus (async event handling)")
    print("  • Position Manager")
    print("  • Risk Manager")
    print("  • Strategy Manager")
    print("  • Backtesting Engine")
    print("  • ML Indicator Evaluator")
    print("  • Conversation Memory (Day 30)")

    print("\n📁 PROJECT STRUCTURE:")
    print("""
    gordon/
    ├── agent/              # AI agent (research + trading)
    ├── core/               # Core trading components
    │   ├── strategies/     # Trading strategies
    │   ├── risk/          # Risk management
    │   └── orchestrator.py # Exchange orchestrator
    ├── exchanges/          # Exchange adapters
    ├── backtesting/        # Backtesting engine
    ├── ml/                 # ML indicator evaluation
    ├── research/           # Research components
    └── entrypoints/        # Entry points
    """)

    print("\n🚀 QUICK START:")
    print("  # Agent mode (research + trading):")
    print("  python -m gordon.entrypoints.cli")
    print("  python -m gordon.entrypoints.cli \"Analyze Apple's financials\"")
    print("")
    print("  # Orchestrator mode (pure trading):")
    print("  python -m gordon.entrypoints.cli --mode orchestrator --exchanges binance --strategies 6:sma")
    print("  python -m gordon.entrypoints.cli --mode orchestrator --exchanges all --trading-mode paper")

    print("\n📊 MODES:")
    print("  • Agent Mode: Financial research + trading via AI")
    print("  • Orchestrator Mode: Pure trading with event-driven architecture")
    print("  • Conversational Mode: Interactive chat with memory (Day 30)")

    print("\n" + "=" * 60)


def run_example_strategies():
    """Run example trading strategies."""
    print("\n" + "=" * 60)
    print("EXAMPLE TRADING STRATEGIES")
    print("=" * 60)

    examples = [
        {
            "name": "Simple Moving Average (Day 6)",
            "command": "python -m gordon.entrypoints.cli --mode orchestrator --exchanges hyperliquid --strategies 6:sma --trading-mode paper",
            "description": "Trades based on SMA crossovers"
        },
        {
            "name": "RSI Strategy (Day 7)",
            "command": "python -m gordon.entrypoints.cli --mode orchestrator --exchanges binance --strategies 7:rsi --trading-mode paper",
            "description": "Trades based on RSI oversold/overbought conditions"
        },
        {
            "name": "Bollinger Bands (Day 10)",
            "command": "python -m gordon.entrypoints.cli --mode orchestrator --exchanges bitfinex --strategies 10:bollinger_bot --trading-mode paper",
            "description": "Trades using Bollinger Bands breakouts"
        },
        {
            "name": "Mean Reversion (Day 20)",
            "command": "python -m gordon.entrypoints.cli --mode orchestrator --exchanges hyperliquid --strategies 20:mr_bot --trading-mode paper",
            "description": "Mean reversion trading strategy"
        },
        {
            "name": "Liquidation Hunter (Day 21)",
            "command": "python -m gordon.entrypoints.cli --mode orchestrator --exchanges all --strategies 21:liq_bt_btc --trading-mode backtest",
            "description": "Hunts liquidation levels for entries"
        },
        {
            "name": "Supply Demand Zones (Day 50)",
            "command": "python -m gordon.entrypoints.cli --mode orchestrator --exchanges hyperliquid --strategies 50:sdz --trading-mode paper",
            "description": "Trades based on supply and demand zones"
        },
        {
            "name": "Multi-Strategy Portfolio",
            "command": "python -m gordon.entrypoints.cli --mode orchestrator --exchanges all --strategies 6:sma 7:rsi 10:bollinger_bot --trading-mode paper",
            "description": "Runs multiple strategies simultaneously"
        }
    ]

    print("\nAvailable example strategies:\n")
    for i, example in enumerate(examples, 1):
        print(f"{i}. {example['name']}")
        print(f"   Description: {example['description']}")
        print(f"   Command: {example['command']}")
        print()

    choice = input("Select an example to run (1-7) or 'q' to quit: ")

    if choice.lower() == 'q':
        return

    try:
        idx = int(choice) - 1
        if 0 <= idx < len(examples):
            print(f"\nRunning: {examples[idx]['name']}")
            print(f"Command: {examples[idx]['command']}")
            os.system(examples[idx]['command'])
        else:
            print("Invalid choice")
    except ValueError:
        print("Invalid input")


async def run_orchestrator_mode(args):
    """Run in Exchange Orchestrator mode."""
    try:
        from gordon.core.orchestrator import ExchangeOrchestrator
        
        # Load orchestrator config if specified
        config_path = args.config
        if not config_path:
            config_path = Path(__file__).parent.parent / 'config' / 'orchestrator_config.json'
        
        orchestrator = ExchangeOrchestrator(config_path=str(config_path) if config_path else None)
        
        # Initialize exchanges if specified
        if args.exchanges:
            exchanges_to_init = args.exchanges if "all" not in args.exchanges else ["binance", "bitfinex", "hyperliquid"]
            for exchange in exchanges_to_init:
                print(f"Initializing {exchange}...")
                success = await orchestrator.initialize_exchange(exchange)
                if success:
                    print(f"✓ {exchange} initialized successfully")
                else:
                    print(f"✗ Failed to initialize {exchange}")
        
        # Load strategies if specified
        if args.strategies:
            for strategy in args.strategies:
                if ":" in strategy:
                    day, strat_name = strategy.split(":")
                    orchestrator.load_strategy_from_day(int(day), strat_name)
                    print(f"✓ Loaded strategy: Day {day} - {strat_name}")
        
        # Start orchestrator
        await orchestrator.start()
        print("\n✅ Exchange Orchestrator started successfully")
        print("Press Ctrl+C to stop\n")
        
        # Keep running
        try:
            while True:
                await asyncio.sleep(60)
                status = orchestrator.get_status()
                print(f"\n📊 Status: {json.dumps(status, indent=2)}")
        except KeyboardInterrupt:
            print("\n\nStopping orchestrator...")
            await orchestrator.stop()
            print("✅ Orchestrator stopped")
        
    except Exception as e:
        print(f"❌ Failed to initialize Exchange Orchestrator: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def main():
    """Main entry point for Gordon CLI."""
    # Handle special flags before argparse
    if len(sys.argv) > 1:
        if sys.argv[1] == "--examples":
            run_example_strategies()
            return
        elif sys.argv[1] == "--dashboard":
            show_dashboard()
            return
    
    parser = argparse.ArgumentParser(
        description='Gordon - Financial Research & Trading Agent',
        epilog="""
Examples:
  # Agent mode (research + trading):
  python -m gordon.entrypoints.cli
  python -m gordon.entrypoints.cli "Analyze Apple's financials"
  python -m gordon.entrypoints.cli --conversational --symbol BTCUSDT

  # Orchestrator mode (pure trading):
  python -m gordon.entrypoints.cli --mode orchestrator --exchanges binance --strategies 6:sma
  python -m gordon.entrypoints.cli --mode orchestrator --exchanges all --trading-mode paper
  
  # Special commands:
  python -m gordon.entrypoints.cli --dashboard
  python -m gordon.entrypoints.cli --examples
        """
    )
    parser.add_argument(
        'query',
        nargs='*',
        help='Query to process (interactive mode if empty)'
    )
    parser.add_argument(
        '--config',
        help='Path to configuration file',
        default=None
    )
    parser.add_argument(
        '--exchange',
        help='Exchange to use (binance, bitfinex, hyperliquid)',
        default='binance'
    )
    parser.add_argument(
        '--position-size',
        type=float,
        help='Base position size for trades',
        default=0.01
    )
    parser.add_argument(
        '--conversational',
        action='store_true',
        help='Enable conversational mode with memory (Day 30)',
        default=False
    )
    parser.add_argument(
        '--symbol',
        help='Primary trading symbol for conversational mode',
        default='BTCUSDT'
    )
    parser.add_argument(
        '--mode',
        choices=['agent', 'orchestrator'],
        default='agent',
        help='Operation mode: agent (research+trading) or orchestrator (pure trading)'
    )
    parser.add_argument(
        '--exchanges',
        nargs='+',
        help='Exchanges to initialize (for orchestrator mode)',
        choices=['binance', 'bitfinex', 'hyperliquid', 'all']
    )
    parser.add_argument(
        '--strategies',
        nargs='+',
        help='Strategies to load in format day:name (for orchestrator mode)',
        default=[]
    )
    parser.add_argument(
        '--trading-mode',
        choices=['live', 'paper', 'backtest'],
        default='paper',
        help='Trading mode (for orchestrator mode)'
    )
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level'
    )
    parser.add_argument(
        '--examples',
        action='store_true',
        help='Show interactive example strategies menu (orchestrator mode)'
    )
    parser.add_argument(
        '--dashboard',
        action='store_true',
        help='Show functionality dashboard'
    )

    args = parser.parse_args()

    # Setup logging level
    if args.log_level:
        import logging
        logging.basicConfig(
            level=getattr(logging, args.log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

    # Handle dashboard/examples flags
    if args.dashboard:
        show_dashboard()
        return
    
    if args.examples:
        if args.mode != 'orchestrator':
            print("⚠️  Examples menu is only available in orchestrator mode")
            print("   Use: python -m gordon.entrypoints.cli --mode orchestrator --examples")
            return
        run_example_strategies()
        return

    # Load configuration
    config = {
        'exchanges': {},
        'base_position_size': args.position_size,
        'max_drawdown': 0.1,
        'risk_per_trade': 0.02
    }

    # Add exchange configuration if API keys are available
    if os.getenv('BINANCE_API_KEY'):
        config['exchanges']['binance'] = {
            'api_key': os.getenv('BINANCE_API_KEY'),
            'api_secret': os.getenv('BINANCE_API_SECRET')
        }

    if os.getenv('OPENAI_API_KEY'):
        config['openai_api_key'] = os.getenv('OPENAI_API_KEY')

    if os.getenv('FINANCIAL_DATASETS_API_KEY'):
        config['financial_datasets_api_key'] = os.getenv('FINANCIAL_DATASETS_API_KEY')

    # Initialize based on mode
    if args.mode == 'orchestrator':
        # Exchange Orchestrator mode (pure trading)
        asyncio.run(run_orchestrator_mode(args))
        return
    
    # Agent mode (research + trading)
    try:
        gordon = Agent()
    except Exception as e:
        print(f"❌ Failed to initialize Gordon: {e}")
        print("\n💡 Make sure you have set the required environment variables:")
        print("   - OPENAI_API_KEY")
        print("   - FINANCIAL_DATASETS_API_KEY (optional)")
        print("   - BINANCE_API_KEY (optional, for trading)")
        print("   - BINANCE_API_SECRET (optional, for trading)")
        sys.exit(1)

    # Load conversation config if available
    try:
        import yaml
        config_path = Path(__file__).parent.parent.parent / 'gordon' / 'config.yaml'
        if config_path.exists():
            with open(config_path, 'r') as f:
                full_config = yaml.safe_load(f)
                if 'conversation' in full_config:
                    config['conversation'] = full_config['conversation']
    except Exception:
        pass  # Use defaults if config can't be loaded
    
    # Run in appropriate mode (only for agent mode)
    if args.mode == 'agent':
        if args.query:
            # Single query mode
            query = ' '.join(args.query)
            run_single_query(gordon, query)
        else:
            # Interactive mode
            run_interactive_mode(
                gordon,
                conversational=args.conversational,
                exchange=args.exchange,
                symbol=args.symbol,
                config=config
            )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nGoodbye!\n")
    except Exception as e:
        print(f"\nFatal error: {e}\n")
        import traceback
        traceback.print_exc()
        sys.exit(1)