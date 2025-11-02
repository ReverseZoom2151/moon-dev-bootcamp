"""
Symbol Normalizer Mixin
========================
Provides symbol format normalization across different exchanges.

Common patterns identified:
- Binance: BTC/USDT format
- Bitfinex: tBTCUST format (with 't' prefix)
- HyperLiquid: BTC-USD format

This mixin provides methods to normalize symbols between standard format
and exchange-specific formats.
"""

from typing import Dict, Optional, Tuple


class SymbolNormalizerMixin:
    """
    Mixin class that provides symbol normalization functionality.

    This mixin adds methods to convert between standard symbol format
    (BASE/QUOTE) and exchange-specific formats.

    Usage:
    ------
        class MyExchange(SymbolNormalizerMixin, BaseExchange):
            def __init__(self, credentials, event_bus):
                super().__init__(credentials, event_bus)
                self.init_symbol_normalizer(
                    default_quote='USDT',
                    separator='/',
                    prefix='',
                    suffix=''
                )

            async def place_order(self, symbol, side, amount):
                normalized = self.normalize_symbol(symbol)
                # Use normalized symbol...

    Attributes:
    -----------
        _default_quote: Default quote currency (e.g., 'USDT')
        _symbol_separator: Separator between base and quote (e.g., '/', '-')
        _symbol_prefix: Prefix for symbols (e.g., 't' for Bitfinex)
        _symbol_suffix: Suffix for symbols
        _symbol_mapping: Cache of normalized -> exchange format mappings
    """

    def init_symbol_normalizer(
        self,
        default_quote: str = 'USDT',
        separator: str = '/',
        prefix: str = '',
        suffix: str = ''
    ):
        """
        Initialize the symbol normalizer.

        Args:
            default_quote: Default quote currency to append if not specified
            separator: Character separating base and quote currencies
            prefix: Prefix to add to symbols (exchange-specific)
            suffix: Suffix to add to symbols (exchange-specific)
        """
        self._default_quote = default_quote
        self._symbol_separator = separator
        self._symbol_prefix = prefix
        self._symbol_suffix = suffix
        self._symbol_mapping = {}  # Cache for symbol mappings

    def normalize_symbol(self, symbol: str, quote: Optional[str] = None) -> str:
        """
        Normalize a symbol to the exchange's format.

        Converts from standard format (BASE/QUOTE or BASE) to exchange-specific format.

        Args:
            symbol: Symbol to normalize (e.g., 'BTC', 'BTC/USDT', 'BTC-USD')
            quote: Optional quote currency override

        Returns:
            Normalized symbol in exchange format

        Examples:
        ---------
            # Binance (separator='/')
            normalize_symbol('BTC') -> 'BTC/USDT'
            normalize_symbol('BTC/USDT') -> 'BTC/USDT'

            # Bitfinex (prefix='t', separator='')
            normalize_symbol('BTC') -> 'tBTCUST'
            normalize_symbol('BTC/USDT') -> 'tBTCUST'

            # HyperLiquid (separator='-', quote='USD')
            normalize_symbol('BTC') -> 'BTC-USD'
            normalize_symbol('BTC/USDT') -> 'BTC-USD'
        """
        # Check cache first
        cache_key = f"{symbol}:{quote}"
        if cache_key in self._symbol_mapping:
            return self._symbol_mapping[cache_key]

        # Parse the symbol
        base, parsed_quote = self.parse_symbol(symbol)

        # Use provided quote or parsed quote or default
        final_quote = quote or parsed_quote or self._default_quote

        # Build normalized symbol
        if self._symbol_separator:
            normalized = f"{base}{self._symbol_separator}{final_quote}"
        else:
            normalized = f"{base}{final_quote}"

        # Add prefix/suffix
        normalized = f"{self._symbol_prefix}{normalized}{self._symbol_suffix}"

        # Cache and return
        self._symbol_mapping[cache_key] = normalized
        return normalized

    def parse_symbol(self, symbol: str) -> Tuple[str, Optional[str]]:
        """
        Parse a symbol into base and quote currencies.

        Args:
            symbol: Symbol to parse (e.g., 'BTC/USDT', 'BTC-USD', 'tBTCUST')

        Returns:
            Tuple of (base_currency, quote_currency or None)

        Examples:
        ---------
            parse_symbol('BTC/USDT') -> ('BTC', 'USDT')
            parse_symbol('BTC-USD') -> ('BTC', 'USD')
            parse_symbol('BTC') -> ('BTC', None)
            parse_symbol('tBTCUST') -> ('BTC', 'UST')
        """
        # Remove prefix/suffix if present
        clean_symbol = symbol
        if self._symbol_prefix and clean_symbol.startswith(self._symbol_prefix):
            clean_symbol = clean_symbol[len(self._symbol_prefix):]
        if self._symbol_suffix and clean_symbol.endswith(self._symbol_suffix):
            clean_symbol = clean_symbol[:-len(self._symbol_suffix)]

        # Try common separators
        for sep in ['/', '-', '_']:
            if sep in clean_symbol:
                parts = clean_symbol.split(sep)
                return parts[0], parts[1] if len(parts) > 1 else None

        # No separator found, try to identify quote currency
        # Common quote currencies
        common_quotes = ['USDT', 'USDC', 'USD', 'UST', 'BTC', 'ETH', 'EUR', 'GBP']

        for quote in common_quotes:
            if clean_symbol.endswith(quote):
                base = clean_symbol[:-len(quote)]
                if base:  # Make sure there's something left as base
                    return base, quote

        # If no quote found, return symbol as base with no quote
        return clean_symbol, None

    def standardize_symbol(self, symbol: str) -> str:
        """
        Convert exchange-specific symbol to standard format (BASE/QUOTE).

        Args:
            symbol: Exchange-specific symbol

        Returns:
            Standardized symbol in BASE/QUOTE format

        Examples:
        ---------
            # Binance
            standardize_symbol('BTC/USDT') -> 'BTC/USDT'

            # Bitfinex
            standardize_symbol('tBTCUST') -> 'BTC/UST'

            # HyperLiquid
            standardize_symbol('BTC-USD') -> 'BTC/USD'
        """
        base, quote = self.parse_symbol(symbol)

        if quote:
            return f"{base}/{quote}"
        else:
            return f"{base}/{self._default_quote}"

    def add_symbol_mapping(self, standard: str, exchange_specific: str):
        """
        Manually add a symbol mapping to the cache.

        Useful for non-standard symbols or special cases.

        Args:
            standard: Standard symbol format (e.g., 'BTC/USDT')
            exchange_specific: Exchange-specific format (e.g., 'tBTCUST')
        """
        self._symbol_mapping[standard] = exchange_specific

    def get_symbol_mapping(self) -> Dict[str, str]:
        """
        Get the current symbol mapping cache.

        Returns:
            Dictionary of cached symbol mappings
        """
        return self._symbol_mapping.copy()

    def clear_symbol_cache(self):
        """
        Clear the symbol mapping cache.

        Useful if symbol formats change or need to be refreshed.
        """
        self._symbol_mapping.clear()

    def bulk_normalize_symbols(self, symbols: list, quote: Optional[str] = None) -> list:
        """
        Normalize multiple symbols at once.

        Args:
            symbols: List of symbols to normalize
            quote: Optional quote currency override

        Returns:
            List of normalized symbols
        """
        return [self.normalize_symbol(symbol, quote) for symbol in symbols]

    def extract_base_currency(self, symbol: str) -> str:
        """
        Extract just the base currency from a symbol.

        Args:
            symbol: Symbol to parse

        Returns:
            Base currency

        Examples:
        ---------
            extract_base_currency('BTC/USDT') -> 'BTC'
            extract_base_currency('tBTCUST') -> 'BTC'
            extract_base_currency('BTC-USD') -> 'BTC'
        """
        base, _ = self.parse_symbol(symbol)
        return base

    def extract_quote_currency(self, symbol: str) -> str:
        """
        Extract just the quote currency from a symbol.

        Args:
            symbol: Symbol to parse

        Returns:
            Quote currency (or default if not found)

        Examples:
        ---------
            extract_quote_currency('BTC/USDT') -> 'USDT'
            extract_quote_currency('tBTCUST') -> 'UST'
            extract_quote_currency('BTC') -> 'USDT' (default)
        """
        _, quote = self.parse_symbol(symbol)
        return quote or self._default_quote
