#!/usr/bin/env python3
"""
Run Gordon - Simple launcher script
This handles the import issues when running Gordon directly.
"""

import os
import sys
import asyncio
from pathlib import Path

# Add parent directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Now we can import everything
from agent.agent import Agent
from dotenv import load_dotenv
from prompt_toolkit import PromptSession
from prompt_toolkit.history import InMemoryHistory

# Load environment variables
load_dotenv()


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

    📊 FINANCIAL RESEARCH (Powered by AI):
    • "Analyze Apple's financial health"
    • "What's Tesla's revenue growth?"
    • "Compare Microsoft and Google margins"
    • "Show me Amazon's latest 10-K filing"

    📈 TECHNICAL TRADING (Coming Soon):
    • "Run RSI strategy on BTC"
    • "Execute SMA crossover on ETH"
    • "Backtest mean reversion on SOL"

    💡 HYBRID ANALYSIS (Gordon's Specialty):
    • "Analyze AAPL fundamentals"
    • "Research TSLA financials"
    • "What are the best tech stocks to buy?"

    ⚙️ SYSTEM COMMANDS:
    • help - Show this help message
    • clear - Clear screen
    • exit/quit - Exit Gordon

    💡 Tips:
    - Gordon specializes in financial research and analysis
    - Specify companies clearly (AAPL, TSLA, MSFT)
    - Use natural language - Gordon understands context
    """)


def main():
    """Main function to run Gordon."""
    print_banner()

    # Check for API keys
    if not os.getenv('OPENAI_API_KEY'):
        print("\n❌ ERROR: OPENAI_API_KEY not found!")
        print("\n📝 To fix this:")
        print("1. Copy .env.example to .env")
        print("2. Add your OpenAI API key to .env")
        print("3. Run this script again\n")
        return

    print("\n🚀 Gordon is ready! Type 'help' for commands or 'exit' to quit.\n")

    # Initialize the agent (using original Dexter agent for now)
    agent = Agent()

    # Create a prompt session with history
    session = PromptSession(history=InMemoryHistory())

    while True:
        try:
            # Get user input
            query = session.prompt("🤖 Gordon> ").strip()

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

            # Process regular queries
            elif query:
                print("\n🔄 Processing...\n")
                # Run the agent (this will use Dexter's research capabilities)
                agent.run(query)
                print("\n" + "─" * 50)

        except (KeyboardInterrupt, EOFError):
            print("\n\n👋 Goodbye! Happy trading! 📈\n")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!\n")
    except Exception as e:
        print(f"\n❌ Fatal error: {e}\n")
        import traceback
        traceback.print_exc()