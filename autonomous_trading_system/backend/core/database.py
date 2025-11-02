"""
Database module for the Autonomous Trading System
Handles database connections and operations
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)

class Database:
    """Simple database interface for the trading system"""
    
    def __init__(self):
        self.connected = False
    
    async def connect(self):
        """Connect to database"""
        # For now, we'll use in-memory storage
        # In production, this would connect to PostgreSQL, MongoDB, etc.
        self.connected = True
        logger.info("✅ Database connected (in-memory mode)")
    
    async def disconnect(self):
        """Disconnect from database"""
        self.connected = False
        logger.info("✅ Database disconnected")


# Global database instance
db: Optional[Database] = None


async def init_db():
    """Initialize database connection"""
    global db
    db = Database()
    await db.connect()


async def close_db():
    """Close database connection"""
    global db
    if db:
        await db.disconnect()


def get_db() -> Database:
    """Get database instance"""
    if not db:
        raise RuntimeError("Database not initialized")
    return db 