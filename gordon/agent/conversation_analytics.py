"""
Conversation Analytics Manager
===============================
Day 30 Enhancement: Analytics and insights for conversation history.

Provides statistics, trends, and insights from conversation data.
"""

import re
import logging
from typing import Dict, List, Optional
from pathlib import Path
from datetime import datetime, timedelta
from collections import Counter, defaultdict

logger = logging.getLogger(__name__)


class ConversationAnalytics:
    """Analyze conversation history for insights."""
    
    def __init__(self, memory_dir: Path):
        """
        Initialize conversation analytics.
        
        Args:
            memory_dir: Directory containing memory files
        """
        self.memory_dir = memory_dir
    
    def get_conversation_stats(self, memory_file: Optional[Path] = None) -> Dict:
        """
        Get statistics for a conversation or all conversations.
        
        Args:
            memory_file: Specific file to analyze (None = all files)
            
        Returns:
            Dictionary with statistics
        """
        if memory_file:
            files = [memory_file]
        else:
            files = list(self.memory_dir.glob('*.txt'))
        
        total_messages = 0
        total_user_messages = 0
        total_ai_messages = 0
        total_symbols_mentioned = Counter()
        total_words = 0
        timestamps = []
        
        for file_path in files:
            try:
                content = file_path.read_text(encoding='utf-8')
                lines = content.splitlines()
                
                for line in lines:
                    # Count messages
                    if 'User:' in line:
                        total_user_messages += 1
                        total_messages += 1
                    elif 'Gordon:' in line or 'AI:' in line:
                        total_ai_messages += 1
                        total_messages += 1
                    
                    # Count words
                    words = line.split()
                    total_words += len(words)
                    
                    # Extract symbols
                    symbols = re.findall(r'\b[A-Z]{2,10}USDT\b', line.upper())
                    total_symbols_mentioned.update(symbols)
                    
                    # Extract timestamps
                    timestamp_match = re.search(r'\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]', line)
                    if timestamp_match:
                        try:
                            dt = datetime.strptime(timestamp_match.group(1), "%Y-%m-%d %H:%M:%S")
                            timestamps.append(dt)
                        except ValueError:
                            pass
                            
            except Exception as e:
                logger.error(f"Error analyzing {file_path}: {e}")
        
        # Calculate time span
        time_span = None
        if timestamps:
            timestamps.sort()
            time_span = {
                'start': timestamps[0].isoformat(),
                'end': timestamps[-1].isoformat(),
                'duration_days': (timestamps[-1] - timestamps[0]).days
            }
        
        return {
            'total_messages': total_messages,
            'user_messages': total_user_messages,
            'ai_messages': total_ai_messages,
            'total_words': total_words,
            'average_words_per_message': round(total_words / total_messages, 2) if total_messages > 0 else 0,
            'most_mentioned_symbols': dict(total_symbols_mentioned.most_common(10)),
            'time_span': time_span,
            'files_analyzed': len(files)
        }
    
    def get_activity_timeline(self, days: int = 30) -> Dict[str, int]:
        """
        Get activity timeline for the last N days.
        
        Args:
            days: Number of days to analyze
            
        Returns:
            Dictionary mapping dates to message counts
        """
        files = list(self.memory_dir.glob('*.txt'))
        activity = defaultdict(int)
        
        cutoff_date = datetime.now() - timedelta(days=days)
        
        for file_path in files:
            try:
                content = file_path.read_text(encoding='utf-8')
                lines = content.splitlines()
                
                for line in lines:
                    timestamp_match = re.search(r'\[(\d{4}-\d{2}-\d{2})', line)
                    if timestamp_match:
                        try:
                            date_str = timestamp_match.group(1)
                            dt = datetime.strptime(date_str, "%Y-%m-%d")
                            
                            if dt >= cutoff_date:
                                activity[date_str] += 1
                        except ValueError:
                            pass
                            
            except Exception as e:
                logger.error(f"Error analyzing {file_path}: {e}")
        
        return dict(sorted(activity.items()))
    
    def get_topic_analysis(self, top_n: int = 10) -> Dict[str, List[str]]:
        """
        Analyze conversation topics.
        
        Args:
            top_n: Number of top topics to return
            
        Returns:
            Dictionary with topic analysis
        """
        files = list(self.memory_dir.glob('*.txt'))
        
        # Keywords for different topics
        topic_keywords = {
            'price_analysis': ['price', 'cost', 'value', 'worth', 'valuation'],
            'technical_analysis': ['rsi', 'sma', 'ema', 'macd', 'bollinger', 'indicator', 'trend'],
            'risk_management': ['risk', 'stop loss', 'take profit', 'position size', 'drawdown'],
            'strategy': ['strategy', 'approach', 'method', 'plan', 'tactic'],
            'market_conditions': ['bull', 'bear', 'market', 'trend', 'volatility'],
            'execution': ['buy', 'sell', 'order', 'trade', 'execute', 'position']
        }
        
        topic_counts = defaultdict(int)
        topic_examples = defaultdict(list)
        
        for file_path in files:
            try:
                content = file_path.read_text(encoding='utf-8').lower()
                
                for topic, keywords in topic_keywords.items():
                    for keyword in keywords:
                        count = content.count(keyword)
                        if count > 0:
                            topic_counts[topic] += count
                            
                            # Extract example line
                            lines = content.splitlines()
                            for line in lines:
                                if keyword in line.lower() and len(topic_examples[topic]) < 3:
                                    topic_examples[topic].append(line.strip()[:100])
                                    break
                                    
            except Exception as e:
                logger.error(f"Error analyzing {file_path}: {e}")
        
        # Get top topics
        top_topics = sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)[:top_n]
        
        return {
            'topics': dict(top_topics),
            'examples': dict(topic_examples)
        }
    
    def get_user_engagement_stats(self) -> Dict:
        """
        Analyze user engagement patterns.
        
        Returns:
            Dictionary with engagement statistics
        """
        files = list(self.memory_dir.glob('*.txt'))
        
        session_lengths = []
        messages_per_session = []
        time_between_messages = []
        last_timestamp = None
        
        for file_path in files:
            try:
                content = file_path.read_text(encoding='utf-8')
                lines = content.splitlines()
                
                session_messages = 0
                session_start = None
                
                for line in lines:
                    timestamp_match = re.search(r'\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]', line)
                    if timestamp_match:
                        try:
                            dt = datetime.strptime(timestamp_match.group(1), "%Y-%m-%d %H:%M:%S")
                            
                            if session_start is None:
                                session_start = dt
                            
                            if last_timestamp:
                                time_diff = (dt - last_timestamp).total_seconds() / 60  # minutes
                                if time_diff < 60:  # Within same session
                                    time_between_messages.append(time_diff)
                            
                            last_timestamp = dt
                            
                            if 'User:' in line:
                                session_messages += 1
                                
                        except ValueError:
                            pass
                
                if session_start and last_timestamp:
                    session_length = (last_timestamp - session_start).total_seconds() / 3600  # hours
                    if session_length > 0:
                        session_lengths.append(session_length)
                        messages_per_session.append(session_messages)
                        
            except Exception as e:
                logger.error(f"Error analyzing {file_path}: {e}")
        
        return {
            'average_session_length_hours': round(sum(session_lengths) / len(session_lengths), 2) if session_lengths else 0,
            'average_messages_per_session': round(sum(messages_per_session) / len(messages_per_session), 2) if messages_per_session else 0,
            'average_time_between_messages_minutes': round(sum(time_between_messages) / len(time_between_messages), 2) if time_between_messages else 0,
            'total_sessions': len(session_lengths)
        }
    
    def generate_insights_report(self) -> str:
        """
        Generate a comprehensive insights report.
        
        Returns:
            Formatted report string
        """
        stats = self.get_conversation_stats()
        timeline = self.get_activity_timeline(days=30)
        topics = self.get_topic_analysis()
        engagement = self.get_user_engagement_stats()
        
        report_lines = [
            "=" * 70,
            "CONVERSATION ANALYTICS REPORT",
            "=" * 70,
            f"\nGenerated: {datetime.now().isoformat()}",
            "",
            "📊 OVERALL STATISTICS",
            "-" * 70,
            f"Total Messages: {stats['total_messages']}",
            f"  - User Messages: {stats['user_messages']}",
            f"  - AI Messages: {stats['ai_messages']}",
            f"Total Words: {stats['total_words']:,}",
            f"Average Words per Message: {stats['average_words_per_message']}",
            f"Files Analyzed: {stats['files_analyzed']}",
            "",
            "📈 ACTIVITY TIMELINE (Last 30 Days)",
            "-" * 70,
        ]
        
        if timeline:
            for date, count in list(timeline.items())[-10:]:  # Last 10 days
                report_lines.append(f"{date}: {count} messages")
        else:
            report_lines.append("No activity in the last 30 days")
        
        report_lines.extend([
            "",
            "🏷️ TOP TOPICS",
            "-" * 70,
        ])
        
        for topic, count in list(topics['topics'].items())[:5]:
            report_lines.append(f"{topic}: {count} mentions")
        
        report_lines.extend([
            "",
            "👤 USER ENGAGEMENT",
            "-" * 70,
            f"Average Session Length: {engagement['average_session_length_hours']:.2f} hours",
            f"Average Messages per Session: {engagement['average_messages_per_session']:.2f}",
            f"Average Time Between Messages: {engagement['average_time_between_messages_minutes']:.2f} minutes",
            f"Total Sessions: {engagement['total_sessions']}",
            "",
            "=" * 70,
        ])
        
        return "\n".join(report_lines)

