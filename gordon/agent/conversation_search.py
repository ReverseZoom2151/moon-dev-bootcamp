"""
Conversation Search Manager
============================
Day 30 Enhancement: Search through conversation history.

Provides full-text search capabilities for conversation memories.
"""

import re
import logging
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)


class ConversationSearcher:
    """Search through conversation memories."""
    
    def __init__(self, memory_dir: Path):
        """
        Initialize conversation searcher.
        
        Args:
            memory_dir: Directory containing memory files
        """
        self.memory_dir = memory_dir
    
    def search(
        self,
        query: str,
        memory_files: Optional[List[Path]] = None,
        case_sensitive: bool = False,
        use_regex: bool = False
    ) -> List[Dict[str, any]]:
        """
        Search conversations for query.
        
        Args:
            query: Search query
            memory_files: Specific files to search (None = all)
            case_sensitive: Whether search is case-sensitive
            use_regex: Whether query is a regex pattern
            
        Returns:
            List of search results with context
        """
        if memory_files is None:
            memory_files = list(self.memory_dir.glob('*.txt'))
        
        results = []
        
        for memory_file in memory_files:
            try:
                file_results = self._search_file(
                    memory_file,
                    query,
                    case_sensitive=case_sensitive,
                    use_regex=use_regex
                )
                results.extend(file_results)
            except Exception as e:
                logger.error(f"Error searching {memory_file}: {e}")
        
        # Sort by relevance (number of matches)
        results.sort(key=lambda x: x['match_count'], reverse=True)
        
        return results
    
    def _search_file(
        self,
        memory_file: Path,
        query: str,
        case_sensitive: bool = False,
        use_regex: bool = False
    ) -> List[Dict[str, any]]:
        """Search a single memory file."""
        if not memory_file.exists():
            return []
        
        content = memory_file.read_text(encoding='utf-8')
        lines = content.splitlines()
        
        # Prepare search pattern
        if use_regex:
            pattern = re.compile(query if case_sensitive else query, re.IGNORECASE)
        else:
            pattern = re.compile(
                re.escape(query),
                re.IGNORECASE if not case_sensitive else 0
            )
        
        matches = []
        current_context = []
        match_count = 0
        
        for i, line in enumerate(lines):
            # Check for matches
            if pattern.search(line):
                match_count += len(pattern.findall(line))
                
                # Get context (3 lines before and after)
                context_start = max(0, i - 3)
                context_end = min(len(lines), i + 4)
                context = lines[context_start:context_end]
                
                matches.append({
                    'file': memory_file.name,
                    'line_number': i + 1,
                    'matched_line': line,
                    'context': '\n'.join(context),
                    'timestamp': self._extract_timestamp(line)
                })
        
        if matches:
            return [{
                'file': memory_file.name,
                'match_count': match_count,
                'matches': matches
            }]
        
        return []
    
    def search_by_symbol(
        self,
        symbol: str,
        memory_files: Optional[List[Path]] = None
    ) -> List[Dict[str, any]]:
        """
        Search for conversations mentioning a specific symbol.
        
        Args:
            symbol: Trading symbol (e.g., 'BTCUSDT')
            memory_files: Specific files to search
            
        Returns:
            List of conversations mentioning the symbol
        """
        return self.search(
            symbol,
            memory_files=memory_files,
            case_sensitive=False
        )
    
    def search_by_date_range(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        memory_files: Optional[List[Path]] = None
    ) -> List[Dict[str, any]]:
        """
        Search conversations within a date range.
        
        Args:
            start_date: Start date (None = no lower bound)
            end_date: End date (None = no upper bound)
            memory_files: Specific files to search
            
        Returns:
            List of conversations in date range
        """
        if memory_files is None:
            memory_files = list(self.memory_dir.glob('*.txt'))
        
        results = []
        
        for memory_file in memory_files:
            try:
                content = memory_file.read_text(encoding='utf-8')
                lines = content.splitlines()
                
                file_matches = []
                for line in lines:
                    timestamp = self._extract_timestamp(line)
                    if timestamp:
                        try:
                            dt = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
                            
                            if start_date and dt < start_date:
                                continue
                            if end_date and dt > end_date:
                                continue
                            
                            file_matches.append({
                                'line': line,
                                'timestamp': timestamp
                            })
                        except ValueError:
                            continue
                
                if file_matches:
                    results.append({
                        'file': memory_file.name,
                        'matches': file_matches,
                        'match_count': len(file_matches)
                    })
                    
            except Exception as e:
                logger.error(f"Error searching {memory_file} by date: {e}")
        
        return results
    
    def _extract_timestamp(self, line: str) -> Optional[str]:
        """Extract timestamp from line."""
        match = re.search(r'\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]', line)
        if match:
            return match.group(1)
        return None
    
    def get_statistics(self, memory_files: Optional[List[Path]] = None) -> Dict[str, any]:
        """
        Get statistics about conversations.
        
        Args:
            memory_files: Specific files to analyze
            
        Returns:
            Dictionary with statistics
        """
        if memory_files is None:
            memory_files = list(self.memory_dir.glob('*.txt'))
        
        total_files = len(memory_files)
        total_lines = 0
        total_size = 0
        symbol_counts = {}
        
        for memory_file in memory_files:
            try:
                content = memory_file.read_text(encoding='utf-8')
                total_lines += len(content.splitlines())
                total_size += len(content.encode('utf-8'))
                
                # Count symbol mentions
                for symbol in ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'SOLUSDT']:
                    count = content.upper().count(symbol)
                    if count > 0:
                        symbol_counts[symbol] = symbol_counts.get(symbol, 0) + count
                        
            except Exception as e:
                logger.error(f"Error analyzing {memory_file}: {e}")
        
        return {
            'total_files': total_files,
            'total_lines': total_lines,
            'total_size_bytes': total_size,
            'total_size_mb': round(total_size / (1024 * 1024), 2),
            'symbol_mentions': symbol_counts,
            'average_lines_per_file': round(total_lines / total_files, 2) if total_files > 0 else 0
        }

