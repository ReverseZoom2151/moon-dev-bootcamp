"""Utility functions for the Twitter Solana token buyer bot."""

def is_contract_address(text: str) -> str | None:
    """Check if text contains a string that looks like a Solana contract address.

    Args:
        text: The input string (e.g., tweet text).

    Returns:
        The potential Solana address string if found, otherwise None.
    """
    # Solana addresses are Base58 encoded and typically 32-44 characters long.
    base58_chars = '123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz'
    words = text.split()
    for word in words:
        if 32 <= len(word) <= 44 and all(c in base58_chars for c in word):
            # Basic check passed, could add more validation if needed (e.g., regex)
            return word
    return None 