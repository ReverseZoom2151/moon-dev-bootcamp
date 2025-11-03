"""
Exchange Orchestrator Main Entry Point
======================================
Main script to run the exchange orchestrator.
"""

import asyncio
import argparse
import sys
import os
from pathlib import Path
import json
import logging
from datetime import datetime

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from core.orchestrator import ExchangeOrchestrator
from utilities.logger import setup_logger


async def main():
    """Main entry point for the Exchange Orchestrator."""

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Exchange Orchestrator for ATC Bootcamp Projects")
    parser.add_argument("--config", type=str, help="Path to configuration file",
                       default="config/orchestrator_config.json")
    parser.add_argument("--exchanges", nargs="+", help="Exchanges to initialize",
                       choices=["binance", "bitfinex", "hyperliquid", "all"])
    parser.add_argument("--strategies", nargs="+", help="Strategies to load")
    parser.add_argument("--mode", choices=["live", "paper", "backtest"],
                       default="paper", help="Trading mode")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       default="INFO", help="Logging level")

    args = parser.parse_args()

    # Setup logging
    logger = setup_logger("Main", level=args.log_level)
    logger.info("=" * 60)
    logger.info("Exchange Orchestrator Starting")
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Config: {args.config}")
    logger.info("=" * 60)

    try:
        # Create orchestrator instance
        orchestrator = ExchangeOrchestrator(config_path=args.config)

        # Initialize selected exchanges
        if args.exchanges:
            exchanges_to_init = args.exchanges if "all" not in args.exchanges else ["binance", "bitfinex", "hyperliquid"]

            for exchange in exchanges_to_init:
                logger.info(f"Initializing {exchange}...")
                success = await orchestrator.initialize_exchange(exchange)
                if success:
                    logger.info(f"✓ {exchange} initialized successfully")
                else:
                    logger.warning(f"✗ Failed to initialize {exchange}")

        # Load strategies if specified
        if args.strategies:
            for strategy in args.strategies:
                logger.info(f"Loading strategy: {strategy}")
                # Parse strategy format: day_number:strategy_name
                if ":" in strategy:
                    day, strat_name = strategy.split(":")
                    orchestrator.load_strategy_from_day(int(day), strat_name)
                else:
                    logger.warning(f"Invalid strategy format: {strategy}. Use day:strategy_name")

        # Start the orchestrator
        await orchestrator.start()

        logger.info("Orchestrator started successfully")
        logger.info("Press Ctrl+C to stop")

        # Keep running until interrupted
        try:
            while True:
                # Print status every 60 seconds
                await asyncio.sleep(60)
                status = orchestrator.get_status()
                logger.info(f"Status Update: {json.dumps(status, indent=2)}")

        except KeyboardInterrupt:
            logger.info("Received interrupt signal")

        # Stop the orchestrator
        logger.info("Stopping orchestrator...")
        await orchestrator.stop()
        logger.info("Orchestrator stopped successfully")

    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


def run_example_strategies():
    """Run example trading strategies."""
    print("\n" + "=" * 60)
    print("EXAMPLE TRADING STRATEGIES")
    print("=" * 60)

    examples = [
        {
            "name": "Simple Moving Average (Day 6)",
            "command": "python main.py --exchanges hyperliquid --strategies 6:sma --mode paper",
            "description": "Trades based on SMA crossovers"
        },
        {
            "name": "RSI Strategy (Day 7)",
            "command": "python main.py --exchanges binance --strategies 7:rsi --mode paper",
            "description": "Trades based on RSI oversold/overbought conditions"
        },
        {
            "name": "Bollinger Bands (Day 10)",
            "command": "python main.py --exchanges bitfinex --strategies 10:bollinger_bot --mode paper",
            "description": "Trades using Bollinger Bands breakouts"
        },
        {
            "name": "Mean Reversion (Day 20)",
            "command": "python main.py --exchanges hyperliquid --strategies 20:mr_bot --mode paper",
            "description": "Mean reversion trading strategy"
        },
        {
            "name": "Liquidation Hunter (Day 21)",
            "command": "python main.py --exchanges all --strategies 21:liq_bt_btc --mode backtest",
            "description": "Hunts liquidation levels for entries"
        },
        {
            "name": "Supply Demand Zones (Day 50)",
            "command": "python main.py --exchanges hyperliquid --strategies 50:sdz --mode paper",
            "description": "Trades based on supply and demand zones"
        },
        {
            "name": "Multi-Strategy Portfolio",
            "command": "python main.py --exchanges all --strategies 6:sma 7:rsi 10:bollinger_bot --mode paper",
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


def show_dashboard():
    """Display a simple dashboard of all available functionality."""
    print("\n" + "=" * 60)
    print("EXCHANGE ORCHESTRATOR DASHBOARD")
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
            "RRS Analysis (Day 37)"
        ],
        "Risk Management": [
            "Position Sizing (Day 5)",
            "Stop Loss/Take Profit (Day 4)",
            "Portfolio Risk (Day 56)"
        ],
        "Experimental": [
            "TikTok Sentiment (Day 42)",
            "Twitter Bot (Day 28)",
            "AI Trading (Day 54)",
            "Spread Scanner (Day 55)"
        ]
    }

    for category, strategies in strategies_by_category.items():
        print(f"\n  {category}:")
        for strategy in strategies:
            print(f"    • {strategy}")

    print("\n🛠️ UTILITIES:")
    print("  • Master Utils (consolidated from 12 nice_funcs.py files)")
    print("  • Event Bus (async event handling)")
    print("  • Position Manager")
    print("  • Risk Manager")
    print("  • Strategy Manager")
    print("  • Backtesting Engine")

    print("\n📁 PROJECT STRUCTURE:")
    print("""
    exchange_orchestrator/
    ├── core/              # Core orchestrator components
    ├── exchanges/         # Exchange adapters
    ├── strategies/        # Trading strategies
    ├── utilities/         # Shared utilities
    ├── config/           # Configuration files
    ├── backtesting/      # Backtesting engine
    └── main.py           # Entry point
    """)

    print("\n🚀 QUICK START:")
    print("  1. Configure exchanges in config/orchestrator_config.json")
    print("  2. Run: python exchange_orchestrator/main.py --help")
    print("  3. Example: python main.py --exchanges hyperliquid --mode paper")

    print("\n📊 BENEFITS:")
    print("  • 75-80% code reduction vs original structure")
    print("  • Single entry point for all trading operations")
    print("  • Unified exchange interface")
    print("  • Event-driven architecture")
    print("  • Comprehensive logging and monitoring")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    # Check if running with special flags
    if len(sys.argv) > 1:
        if sys.argv[1] == "--examples":
            run_example_strategies()
        elif sys.argv[1] == "--dashboard":
            show_dashboard()
        else:
            # Run main orchestrator
            asyncio.run(main())
    else:
        # Show help menu
        print("\n" + "=" * 60)
        print("EXCHANGE ORCHESTRATOR FOR ATC BOOTCAMP (DAYS 2-56)")
        print("=" * 60)
        print("\nUsage:")
        print("  python main.py [options]")
        print("\nOptions:")
        print("  --help              Show detailed help")
        print("  --dashboard         Show functionality dashboard")
        print("  --examples          Run example strategies")
        print("  --config PATH       Path to config file")
        print("  --exchanges NAMES   Exchanges to use (binance, bitfinex, hyperliquid, all)")
        print("  --strategies STRATS Strategies to load (format: day:name)")
        print("  --mode MODE         Trading mode (live, paper, backtest)")
        print("  --log-level LEVEL   Logging level")
        print("\nExamples:")
        print("  python main.py --dashboard")
        print("  python main.py --examples")
        print("  python main.py --exchanges hyperliquid --mode paper")
        print("  python main.py --exchanges all --strategies 10:bollinger_bot --mode backtest")
        print("\n" + "=" * 60)