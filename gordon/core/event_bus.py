"""
Event Bus System
================
Central event management for the orchestrator.
Implements publish-subscribe pattern for decoupled communication.
"""

import asyncio
import logging
from typing import Dict, List, Callable, Any, Optional
from datetime import datetime
from collections import defaultdict
import json


class EventBus:
    """
    Asynchronous event bus for managing events across the orchestrator.

    Features:
    - Async event handling
    - Priority-based event processing
    - Event filtering and routing
    - Event history tracking
    """

    def __init__(self, max_history: int = 1000):
        """
        Initialize the Event Bus.

        Args:
            max_history: Maximum number of events to keep in history
        """
        self.logger = logging.getLogger(__name__)
        self.subscribers: Dict[str, List[Callable]] = defaultdict(list)
        self.event_history: List[Dict] = []
        self.max_history = max_history
        self.event_queue = asyncio.Queue()
        self.processing = False
        self.filters: Dict[str, Callable] = {}
        self.event_stats = defaultdict(int)

    def subscribe(self, event_type: str, handler: Callable,
                 priority: int = 0, filter_func: Optional[Callable] = None):
        """
        Subscribe to an event type.

        Args:
            event_type: Type of event to subscribe to
            handler: Async function to handle the event
            priority: Handler priority (higher = earlier execution)
            filter_func: Optional filter function
        """
        subscription = {
            "handler": handler,
            "priority": priority,
            "filter": filter_func
        }

        self.subscribers[event_type].append(subscription)

        # Sort by priority
        self.subscribers[event_type].sort(
            key=lambda x: x["priority"],
            reverse=True
        )

        self.logger.debug(f"Subscribed handler to event: {event_type}")

    def unsubscribe(self, event_type: str, handler: Callable):
        """
        Unsubscribe from an event type.

        Args:
            event_type: Type of event to unsubscribe from
            handler: Handler function to remove
        """
        if event_type in self.subscribers:
            self.subscribers[event_type] = [
                sub for sub in self.subscribers[event_type]
                if sub["handler"] != handler
            ]
            self.logger.debug(f"Unsubscribed handler from event: {event_type}")

    async def emit(self, event_type: str, data: Any = None,
                  metadata: Optional[Dict] = None):
        """
        Emit an event asynchronously.

        Args:
            event_type: Type of event to emit
            data: Event data
            metadata: Optional metadata
        """
        event = {
            "type": event_type,
            "data": data,
            "metadata": metadata or {},
            "timestamp": datetime.now().isoformat(),
            "id": f"{event_type}_{datetime.now().timestamp()}"
        }

        # Add to queue
        await self.event_queue.put(event)

        # Start processing if not already running
        if not self.processing:
            asyncio.create_task(self._process_queue())

        # Track statistics
        self.event_stats[event_type] += 1

        # Add to history
        self._add_to_history(event)

        self.logger.debug(f"Event emitted: {event_type}")

    async def emit_sync(self, event_type: str, data: Any = None,
                       metadata: Optional[Dict] = None):
        """
        Emit an event and wait for all handlers to complete.

        Args:
            event_type: Type of event to emit
            data: Event data
            metadata: Optional metadata
        """
        event = {
            "type": event_type,
            "data": data,
            "metadata": metadata or {},
            "timestamp": datetime.now().isoformat(),
            "id": f"{event_type}_{datetime.now().timestamp()}"
        }

        # Process immediately
        await self._process_event(event)

        # Track statistics
        self.event_stats[event_type] += 1

        # Add to history
        self._add_to_history(event)

    async def _process_queue(self):
        """Process events from the queue."""
        self.processing = True

        try:
            while not self.event_queue.empty():
                event = await self.event_queue.get()
                await self._process_event(event)
        finally:
            self.processing = False

    async def _process_event(self, event: Dict):
        """
        Process a single event.

        Args:
            event: Event dictionary
        """
        event_type = event["type"]

        if event_type not in self.subscribers:
            return

        # Get all subscribers for this event
        subscribers = self.subscribers[event_type]

        # Process each subscriber
        tasks = []
        for subscription in subscribers:
            handler = subscription["handler"]
            filter_func = subscription["filter"]

            # Apply filter if present
            if filter_func and not filter_func(event):
                continue

            # Create task for async handler
            if asyncio.iscoroutinefunction(handler):
                tasks.append(self._call_handler(handler, event))
            else:
                # Wrap sync handler
                tasks.append(asyncio.create_task(
                    asyncio.to_thread(handler, event)
                ))

        # Execute all handlers
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Log any exceptions
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    self.logger.error(
                        f"Error in event handler for {event_type}: {result}"
                    )

    async def _call_handler(self, handler: Callable, event: Dict):
        """
        Call an event handler with error handling.

        Args:
            handler: Handler function
            event: Event data
        """
        try:
            await handler(event)
        except Exception as e:
            self.logger.error(f"Error in event handler: {e}")
            raise

    def _add_to_history(self, event: Dict):
        """
        Add event to history with size limit.

        Args:
            event: Event to add
        """
        self.event_history.append(event)

        # Trim history if needed
        if len(self.event_history) > self.max_history:
            self.event_history = self.event_history[-self.max_history:]

    def get_history(self, event_type: Optional[str] = None,
                   limit: int = 100) -> List[Dict]:
        """
        Get event history.

        Args:
            event_type: Optional filter by event type
            limit: Maximum number of events to return

        Returns:
            List of events
        """
        history = self.event_history

        if event_type:
            history = [e for e in history if e["type"] == event_type]

        return history[-limit:]

    def get_statistics(self) -> Dict[str, int]:
        """
        Get event statistics.

        Returns:
            Dictionary of event counts by type
        """
        return dict(self.event_stats)

    def clear_history(self):
        """Clear event history."""
        self.event_history.clear()
        self.logger.info("Event history cleared")

    def add_global_filter(self, name: str, filter_func: Callable):
        """
        Add a global filter that applies to all events.

        Args:
            name: Filter name
            filter_func: Filter function
        """
        self.filters[name] = filter_func
        self.logger.info(f"Added global filter: {name}")

    def remove_global_filter(self, name: str):
        """
        Remove a global filter.

        Args:
            name: Filter name to remove
        """
        if name in self.filters:
            del self.filters[name]
            self.logger.info(f"Removed global filter: {name}")

    async def wait_for_event(self, event_type: str,
                            timeout: Optional[float] = None) -> Optional[Dict]:
        """
        Wait for a specific event to occur.

        Args:
            event_type: Event type to wait for
            timeout: Optional timeout in seconds

        Returns:
            Event data if received, None if timeout
        """
        future = asyncio.Future()

        def handler(event):
            if not future.done():
                future.set_result(event)

        self.subscribe(event_type, handler, priority=100)

        try:
            result = await asyncio.wait_for(future, timeout=timeout)
            return result
        except asyncio.TimeoutError:
            return None
        finally:
            self.unsubscribe(event_type, handler)

    def export_history(self, filepath: str):
        """
        Export event history to JSON file.

        Args:
            filepath: Path to save the history
        """
        try:
            with open(filepath, 'w') as f:
                json.dump(self.event_history, f, indent=2)
            self.logger.info(f"Event history exported to {filepath}")
        except Exception as e:
            self.logger.error(f"Failed to export history: {e}")


class EventTypes:
    """Standard event types used across the orchestrator."""

    # System events
    ORCHESTRATOR_STARTED = "orchestrator_started"
    ORCHESTRATOR_STOPPED = "orchestrator_stopped"

    # Trading events
    TRADE_EXECUTED = "trade_executed"
    ORDER_PLACED = "order_placed"
    ORDER_FILLED = "order_filled"
    ORDER_CANCELLED = "order_cancelled"
    ORDER_FAILED = "order_failed"

    # Position events
    POSITION_OPENED = "position_opened"
    POSITION_CLOSED = "position_closed"
    POSITION_UPDATED = "position_updated"

    # Strategy events
    STRATEGY_SIGNAL = "strategy_signal"
    STRATEGY_STARTED = "strategy_started"
    STRATEGY_STOPPED = "strategy_stopped"
    STRATEGY_ERROR = "strategy_error"

    # Risk events
    RISK_ALERT = "risk_alert"
    RISK_LIMIT_REACHED = "risk_limit_reached"
    DRAWDOWN_ALERT = "drawdown_alert"

    # Market data events
    PRICE_UPDATE = "price_update"
    ORDERBOOK_UPDATE = "orderbook_update"
    TRADE_UPDATE = "trade_update"

    # Exchange events
    EXCHANGE_CONNECTED = "exchange_connected"
    EXCHANGE_DISCONNECTED = "exchange_disconnected"
    EXCHANGE_ERROR = "exchange_error"

    # Performance events
    METRICS_UPDATED = "metrics_updated"
    PNL_UPDATE = "pnl_update"