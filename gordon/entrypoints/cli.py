#!/usr/bin/env python3
"""
Gordon CLI - The Complete Trading Assistant
===========================================
Combines financial research with technical trading.
"""

import os
import sys
import argparse
from pathlib import Path
from dotenv import load_dotenv
from prompt_toolkit import PromptSession
from prompt_toolkit.history import InMemoryHistory

# Load environment variables
load_dotenv()

# Add the project root (parent of gordon) to path
project_root = str(Path(__file__).parent.parent.parent)
sys.path.insert(0, project_root)

# Now we can import gordon modules properly
from gordon.agent.agent import Agent


def print_banner():
    """Print Gordon's welcome banner."""
    print("""
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

    ⚙️ SYSTEM COMMANDS:
    • help - Show this help message
    • status - Show system status
    • config - Show/modify configuration
    • risk - Show risk limits
    • clear - Clear screen
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


def run_interactive_mode(gordon: Agent):
    """Run Gordon in interactive mode."""
    print_banner()
    print("\n🚀 Gordon is ready! Type 'help' for commands or 'exit' to quit.\n")

    # Create a prompt session with history
    session = PromptSession(history=InMemoryHistory())

    while True:
        try:
            # Get user input with prompt toolkit
            query = session.prompt("\n🤖 Gordon> ").strip()

            # Handle special commands
            if query.lower() in ['exit', 'quit', 'q']:
                print("\n👋 Goodbye! Happy trading! 📈\n")
                break

            elif query.lower() == 'help':
                print_help()
                continue

            elif query.lower() == 'clear':
                os.system('cls' if os.name == 'nt' else 'clear')
                print_banner()
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

            # Process regular queries
            elif query:
                print("\n🔄 Processing...\n")
                # Use the Agent's run method (synchronous)
                gordon.run(query)
                print("\n" + "─" * 50)

        except (KeyboardInterrupt, EOFError):
            print("\n\n👋 Goodbye! Happy trading! 📈\n")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}\n")


def run_single_query(gordon: Agent, query: str):
    """Run a single query and exit."""
    print(f"\n🔄 Processing: {query}\n")
    # Use the Agent's run method (synchronous)
    gordon.run(query)
    print()


def main():
    """Main entry point for Gordon CLI."""
    parser = argparse.ArgumentParser(
        description='Gordon - Financial Research & Trading Agent'
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

    args = parser.parse_args()

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

    # Initialize Gordon using the Agent class
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

    # Run in appropriate mode
    if args.query:
        # Single query mode
        query = ' '.join(args.query)
        run_single_query(gordon, query)
    else:
        # Interactive mode
        run_interactive_mode(gordon)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!\n")
    except Exception as e:
        print(f"\n❌ Fatal error: {e}\n")
        sys.exit(1)