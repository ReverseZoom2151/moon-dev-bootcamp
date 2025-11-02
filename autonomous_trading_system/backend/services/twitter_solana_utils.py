"""Utility functions for the Twitter Solana token buyer bot service."""

import re
import logging
from typing import Optional

logger = logging.getLogger(__name__)

def is_contract_address(text: str) -> Optional[str]:
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

def validate_solana_address(address: str) -> bool:
    """Validate if a string is a properly formatted Solana address.
    
    Args:
        address: The address string to validate
        
    Returns:
        True if the address appears to be valid, False otherwise
    """
    if not address:
        return False
        
    # Basic length check
    if not (32 <= len(address) <= 44):
        return False
        
    # Base58 character set check
    base58_chars = '123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz'
    if not all(c in base58_chars for c in address):
        return False
        
    # Additional checks could be added here (checksum validation, etc.)
    return True

def sanitize_tweet_text(text: str) -> str:
    """Clean and sanitize tweet text for processing.
    
    Args:
        text: Raw tweet text
        
    Returns:
        Sanitized text string
    """
    if not text:
        return ""
        
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text.strip())
    
    # Remove control characters
    text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)
    
    return text

def extract_token_mentions(text: str) -> list[str]:
    """Extract potential token mentions from tweet text.
    
    Args:
        text: Tweet text to analyze
        
    Returns:
        List of potential token symbols or mentions
    """
    # Common patterns for token mentions
    patterns = [
        r'\$([A-Z]{2,10})',  # $TOKEN format
        r'#([A-Z]{2,10})',   # #TOKEN format
        r'\b([A-Z]{2,10})\s*token\b',  # TOKEN token format
        r'\b([A-Z]{2,10})\s*coin\b',   # TOKEN coin format
    ]
    
    mentions = []
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        mentions.extend(matches)
    
    # Remove duplicates and filter common words
    common_words = {'THE', 'AND', 'FOR', 'YOU', 'ARE', 'NOT', 'CAN', 'GET', 'NEW', 'NOW', 'ALL'}
    mentions = list(set([m.upper() for m in mentions if m.upper() not in common_words]))
    
    return mentions 