"""
Conversation Export/Import Manager
===================================
Day 30 Enhancement: Export and import conversation history.

Allows users to save, share, and restore conversation memories.
"""

import json
import csv
import logging
from typing import Dict, List, Optional
from pathlib import Path
from datetime import datetime

from .conversation_memory import ConversationMemory, ExchangeConversationMemory

logger = logging.getLogger(__name__)


class ConversationExporter:
    """Export conversation memories to various formats."""
    
    @staticmethod
    def export_to_json(memory_path: Path, output_path: Optional[Path] = None) -> Path:
        """
        Export conversation memory to JSON.
        
        Args:
            memory_path: Path to memory file
            output_path: Output path (defaults to memory_path with .json extension)
            
        Returns:
            Path to exported file
        """
        if not memory_path.exists():
            raise FileNotFoundError(f"Memory file not found: {memory_path}")
        
        if output_path is None:
            output_path = memory_path.with_suffix('.json')
        
        # Read memory content
        content = memory_path.read_text(encoding='utf-8')
        
        # Parse into structured format
        export_data = {
            'file': str(memory_path.name),
            'exported_at': datetime.now().isoformat(),
            'content': content,
            'line_count': len(content.splitlines()),
            'size_bytes': len(content.encode('utf-8'))
        }
        
        # Write JSON
        output_path.write_text(json.dumps(export_data, indent=2), encoding='utf-8')
        logger.info(f"Exported conversation to {output_path}")
        
        return output_path
    
    @staticmethod
    def export_to_csv(memory_path: Path, output_path: Optional[Path] = None) -> Path:
        """
        Export conversation memory to CSV format.
        
        Args:
            memory_path: Path to memory file
            output_path: Output path (defaults to memory_path with .csv extension)
            
        Returns:
            Path to exported file
        """
        if not memory_path.exists():
            raise FileNotFoundError(f"Memory file not found: {memory_path}")
        
        if output_path is None:
            output_path = memory_path.with_suffix('.csv')
        
        # Read and parse memory
        content = memory_path.read_text(encoding='utf-8')
        lines = content.splitlines()
        
        # Parse conversations
        conversations = []
        current_user = None
        current_ai = None
        current_timestamp = None
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#') or 'START' in line or 'END' in line:
                continue
            
            if line.startswith('[') and ']' in line:
                # Extract timestamp
                timestamp_end = line.index(']')
                current_timestamp = line[1:timestamp_end]
                line = line[timestamp_end + 1:].strip()
            
            if 'User:' in line:
                if current_user and current_ai:
                    conversations.append({
                        'timestamp': current_timestamp or '',
                        'user': current_user,
                        'ai': current_ai
                    })
                current_user = line.replace('User:', '').strip()
                current_ai = None
            elif 'Gordon:' in line or 'AI:' in line:
                current_ai = line.replace('Gordon:', '').replace('AI:', '').strip()
        
        # Add last conversation
        if current_user and current_ai:
            conversations.append({
                'timestamp': current_timestamp or '',
                'user': current_user,
                'ai': current_ai
            })
        
        # Write CSV
        if conversations:
            with open(output_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=['timestamp', 'user', 'ai'])
                writer.writeheader()
                writer.writerows(conversations)
        else:
            # Empty CSV with headers
            with open(output_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=['timestamp', 'user', 'ai'])
                writer.writeheader()
        
        logger.info(f"Exported conversation to {output_path}")
        return output_path
    
    @staticmethod
    def export_to_txt(memory_path: Path, output_path: Optional[Path] = None) -> Path:
        """
        Export conversation memory to plain text.
        
        Args:
            memory_path: Path to memory file
            output_path: Output path (defaults to memory_path with .export.txt extension)
            
        Returns:
            Path to exported file
        """
        if not memory_path.exists():
            raise FileNotFoundError(f"Memory file not found: {memory_path}")
        
        if output_path is None:
            output_path = memory_path.parent / f"{memory_path.stem}.export.txt"
        
        # Read and format
        content = memory_path.read_text(encoding='utf-8')
        
        # Add export header
        export_header = f"""
{'='*70}
CONVERSATION EXPORT
Exported: {datetime.now().isoformat()}
Source: {memory_path.name}
{'='*70}

"""
        
        output_path.write_text(export_header + content, encoding='utf-8')
        logger.info(f"Exported conversation to {output_path}")
        
        return output_path


class ConversationImporter:
    """Import conversation memories from various formats."""
    
    @staticmethod
    def import_from_json(json_path: Path, memory: ConversationMemory) -> bool:
        """
        Import conversation from JSON export.
        
        Args:
            json_path: Path to JSON export file
            memory: Memory instance to import into
            
        Returns:
            True if successful
        """
        try:
            data = json.loads(json_path.read_text(encoding='utf-8'))
            
            # Write content to memory file
            memory.memory_path.write_text(data['content'], encoding='utf-8')
            logger.info(f"Imported conversation from {json_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error importing from JSON: {e}")
            return False
    
    @staticmethod
    def import_from_txt(txt_path: Path, memory: ConversationMemory) -> bool:
        """
        Import conversation from text file.
        
        Args:
            txt_path: Path to text file
            memory: Memory instance to import into
            
        Returns:
            True if successful
        """
        try:
            content = txt_path.read_text(encoding='utf-8')
            
            # Remove export header if present
            if 'CONVERSATION EXPORT' in content:
                lines = content.splitlines()
                start_idx = 0
                for i, line in enumerate(lines):
                    if '='*70 in line and i > 0:
                        start_idx = i + 1
                        break
                content = '\n'.join(lines[start_idx:])
            
            memory.memory_path.write_text(content.strip(), encoding='utf-8')
            logger.info(f"Imported conversation from {txt_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error importing from text: {e}")
            return False


def export_all_conversations(memory_dir: Path, output_dir: Path, format: str = 'json'):
    """
    Export all conversations in a directory.
    
    Args:
        memory_dir: Directory containing memory files
        output_dir: Output directory
        format: Export format ('json', 'csv', or 'txt')
        
    Returns:
        List of exported file paths
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    exported_files = []
    
    for memory_file in memory_dir.glob('*.txt'):
        try:
            if format == 'json':
                output_path = output_dir / f"{memory_file.stem}.json"
                ConversationExporter.export_to_json(memory_file, output_path)
            elif format == 'csv':
                output_path = output_dir / f"{memory_file.stem}.csv"
                ConversationExporter.export_to_csv(memory_file, output_path)
            elif format == 'txt':
                output_path = output_dir / f"{memory_file.stem}.txt"
                ConversationExporter.export_to_txt(memory_file, output_path)
            else:
                logger.error(f"Unknown format: {format}")
                continue
            
            exported_files.append(output_path)
            
        except Exception as e:
            logger.error(f"Error exporting {memory_file}: {e}")
    
    return exported_files

