def print_intro():
    """Display the welcome screen with ASCII art for Gordon."""
    # ANSI color codes
    LIGHT_BLUE = "\033[94m"
    GREEN = "\033[92m"
    RESET = "\033[0m"
    BOLD = "\033[1m"

    # Clear screen effect with some spacing
    print("\n" * 2)

    # Welcome box with gradient effect
    box_width = 60
    welcome_text = "Welcome to Gordon"
    subtitle = "Financial Research + Technical Trading"
    padding = (box_width - len(welcome_text) - 2) // 2
    subtitle_padding = (box_width - len(subtitle) - 2) // 2

    print(f"{GREEN}{'═' * box_width}{RESET}")
    print(f"{GREEN}║{' ' * padding}{BOLD}{welcome_text}{RESET}{GREEN}{' ' * (box_width - len(welcome_text) - padding - 2)}║{RESET}")
    print(f"{GREEN}║{' ' * subtitle_padding}{subtitle}{' ' * (box_width - len(subtitle) - subtitle_padding - 2)}║{RESET}")
    print(f"{GREEN}{'═' * box_width}{RESET}")
    print()

    # ASCII art for GORDON in block letters
    gordon_art = f"""{BOLD}{LIGHT_BLUE}
 ██████╗  ██████╗ ██████╗ ██████╗  ██████╗ ███╗   ██╗
██╔════╝ ██╔═══██╗██╔══██╗██╔══██╗██╔═══██╗████╗  ██║
██║  ███╗██║   ██║██████╔╝██║  ██║██║   ██║██╔██╗ ██║
██║   ██║██║   ██║██╔══██╗██║  ██║██║   ██║██║╚██╗██║
╚██████╔╝╚██████╔╝██║  ██║██████╔╝╚██████╔╝██║ ╚████║
 ╚═════╝  ╚═════╝ ╚═╝  ╚═╝╚═════╝  ╚═════╝ ╚═╝  ╚═══╝
{RESET}"""

    print(gordon_art)
    print()
    print(f"{GREEN}🤖{RESET} Your AI agent for fundamental analysis AND technical trading")
    print(f"{LIGHT_BLUE}📊{RESET} Combines Warren Buffett's wisdom with quantitative strategies")
    print(f"{GREEN}📈{RESET} 10+ trading strategies | Risk management | Live execution")
    print()
    print("Ask me anything about financials, trading, or both!")
    print("Type 'help' for examples or 'exit' to quit.")
    print()

