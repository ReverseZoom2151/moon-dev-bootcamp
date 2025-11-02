"""
Example Usage of Algorithmic Orders
====================================
This file demonstrates how to use the refactored algorithmic order modules.
"""

import asyncio
from exchange_orchestrator.core.algo_orders import (
    AlgoType,
    ManualLoopOrder,
    ScheduledOrder,
    TWAPOrder,
    VWAPOrder,
    IcebergOrder,
    GridOrder,
    DCAOrder
)


# Example 1: Using ManualLoopOrder directly
async def example_manual_loop(orchestrator, event_bus):
    """
    Example of using ManualLoopOrder for testing order placement/cancellation.
    """
    print("=== Manual Loop Example ===")

    order = ManualLoopOrder(
        exchange_orchestrator=orchestrator,
        event_bus=event_bus,
        exchange="binance",
        symbol="BTC/USDT",
        size=0.001,
        params={
            'wait_time': 10,  # Wait 10 seconds before canceling
            'bid_offset': 0.01,  # 1% below market
            'dry_run': True  # Simulate without real orders
        }
    )

    # Start the order (will run until stopped)
    task = asyncio.create_task(order.start())

    # Let it run for 30 seconds
    await asyncio.sleep(30)

    # Stop the order
    order.is_running = False
    await task

    # Print statistics
    stats = order.get_stats()
    print(f"Stats: {stats}")


# Example 2: Using ScheduledOrder
async def example_scheduled(orchestrator, event_bus):
    """
    Example of using ScheduledOrder for periodic trading.
    """
    print("=== Scheduled Order Example ===")

    order = ScheduledOrder(
        exchange_orchestrator=orchestrator,
        event_bus=event_bus,
        exchange="binance",
        symbol="BTC/USDT",
        size=0.001,
        params={
            'interval': 28,  # Execute every 28 seconds
            'dry_run': True
        }
    )

    task = asyncio.create_task(order.start())
    await asyncio.sleep(120)  # Run for 2 minutes

    order.is_running = False
    await task


# Example 3: Using TWAPOrder
async def example_twap(orchestrator, event_bus):
    """
    Example of using TWAPOrder to split a large order over time.
    """
    print("=== TWAP Example ===")

    order = TWAPOrder(
        exchange_orchestrator=orchestrator,
        event_bus=event_bus,
        exchange="binance",
        symbol="BTC/USDT",
        size=1.0,  # Total 1 BTC to buy
        params={
            'duration': 60,  # Over 60 minutes
            'slices': 20,  # In 20 equal slices
            'side': 'buy'
        }
    )

    await order.start()  # Will complete after all slices executed


# Example 4: Using VWAPOrder
async def example_vwap(orchestrator, event_bus):
    """
    Example of using VWAPOrder with volume-weighted distribution.
    """
    print("=== VWAP Example ===")

    order = VWAPOrder(
        exchange_orchestrator=orchestrator,
        event_bus=event_bus,
        exchange="binance",
        symbol="BTC/USDT",
        size=1.0,
        params={
            'duration': 60,
            'slices': 20,
            'side': 'buy'
        }
    )

    await order.start()


# Example 5: Using IcebergOrder
async def example_iceberg(orchestrator, event_bus):
    """
    Example of using IcebergOrder to hide large order size.
    """
    print("=== Iceberg Order Example ===")

    order = IcebergOrder(
        exchange_orchestrator=orchestrator,
        event_bus=event_bus,
        exchange="binance",
        symbol="BTC/USDT",
        size=10.0,  # Total 10 BTC
        params={
            'visible_size': 1.0,  # Show only 1 BTC at a time
            'side': 'buy',
            'price': 50000  # Limit price
        }
    )

    await order.start()


# Example 6: Using GridOrder
async def example_grid(orchestrator, event_bus):
    """
    Example of using GridOrder for range-bound trading.
    """
    print("=== Grid Trading Example ===")

    order = GridOrder(
        exchange_orchestrator=orchestrator,
        event_bus=event_bus,
        exchange="binance",
        symbol="BTC/USDT",
        size=0.1,  # 0.1 BTC per grid level
        params={
            'levels': 10,  # 10 grid levels (5 buy, 5 sell)
            'spacing': 0.005,  # 0.5% spacing between levels
            'check_interval': 60  # Check every 60 seconds
        }
    )

    task = asyncio.create_task(order.start())

    # Run for 1 hour
    await asyncio.sleep(3600)

    # Stop grid
    order.is_running = False
    await task


# Example 7: Using DCAOrder
async def example_dca(orchestrator, event_bus):
    """
    Example of using DCAOrder for dollar cost averaging.
    """
    print("=== DCA Example ===")

    order = DCAOrder(
        exchange_orchestrator=orchestrator,
        event_bus=event_bus,
        exchange="binance",
        symbol="BTC/USDT",
        size=0.01,  # Buy 0.01 BTC each time
        params={
            'interval_hours': 24,  # Every 24 hours
            'total_buys': 30  # 30 total buys (1 month)
        }
    )

    await order.start()  # Will complete after all buys


# Example 8: Using AlgorithmicTrader facade (backward compatibility)
async def example_algorithmic_trader(orchestrator, event_bus):
    """
    Example of using the AlgorithmicTrader facade for backward compatibility.
    """
    print("=== AlgorithmicTrader Facade Example ===")

    from exchange_orchestrator.core.algo_orders import AlgorithmicTrader

    trader = AlgorithmicTrader(
        exchange_orchestrator=orchestrator,
        event_bus=event_bus,
        config={
            'dry_run': True,
            'dynamic_pricing': True,
            'max_orders': 100
        }
    )

    # Start algorithm using the original interface
    await trader.start_algorithm(
        algo_type=AlgoType.MANUAL_LOOP,
        exchange="binance",
        symbol="BTC/USDT",
        size=0.001,
        params={'wait_time': 10}
    )

    # Let it run for 30 seconds
    await asyncio.sleep(30)

    # Stop the algorithm
    algo_id = "binance_BTC/USDT_manual_loop"
    await trader.stop_algorithm(algo_id)

    # Get statistics
    stats = trader.get_statistics()
    print(f"Trader stats: {stats}")


# Example 9: Multiple algorithms running simultaneously
async def example_multiple_algorithms(orchestrator, event_bus):
    """
    Example of running multiple algorithms simultaneously.
    """
    print("=== Multiple Algorithms Example ===")

    # Start TWAP on BTC/USDT
    twap_order = TWAPOrder(
        orchestrator, event_bus, "binance", "BTC/USDT", 1.0,
        params={'duration': 60, 'slices': 20}
    )
    twap_task = asyncio.create_task(twap_order.start())

    # Start Grid on ETH/USDT
    grid_order = GridOrder(
        orchestrator, event_bus, "binance", "ETH/USDT", 0.5,
        params={'levels': 10, 'spacing': 0.005}
    )
    grid_task = asyncio.create_task(grid_order.start())

    # Start DCA on LINK/USDT
    dca_order = DCAOrder(
        orchestrator, event_bus, "binance", "LINK/USDT", 10,
        params={'interval_hours': 1, 'total_buys': 24}
    )
    dca_task = asyncio.create_task(dca_order.start())

    # Wait for all to complete or timeout after 1 hour
    await asyncio.wait(
        [twap_task, grid_task, dca_task],
        timeout=3600,
        return_when=asyncio.FIRST_COMPLETED
    )

    # Stop any still running
    grid_order.is_running = False
    dca_order.is_running = False


# Example 10: Custom event handling
async def example_custom_events(orchestrator, event_bus):
    """
    Example of handling custom events from algorithmic orders.
    """
    print("=== Custom Event Handling Example ===")

    # Define event handlers
    async def on_algorithm_started(event):
        print(f"Algorithm started: {event}")

    async def on_algorithm_stopped(event):
        print(f"Algorithm stopped: {event}")
        print(f"Final stats: {event.get('stats')}")

    # Subscribe to events
    event_bus.subscribe("algorithm_started", on_algorithm_started)
    event_bus.subscribe("algorithm_stopped", on_algorithm_stopped)

    # Run an algorithm
    order = ManualLoopOrder(
        orchestrator, event_bus, "binance", "BTC/USDT", 0.001,
        params={'wait_time': 5, 'dry_run': True}
    )

    await order.start()


# Main function to run all examples
async def main():
    """
    Main function to demonstrate usage.

    Note: You'll need to provide actual orchestrator and event_bus instances.
    """
    # These would be your actual instances
    orchestrator = None  # YourOrchestratorInstance()
    event_bus = None  # YourEventBusInstance()

    if orchestrator and event_bus:
        # Run examples (uncomment the ones you want to try)

        # await example_manual_loop(orchestrator, event_bus)
        # await example_scheduled(orchestrator, event_bus)
        # await example_twap(orchestrator, event_bus)
        # await example_vwap(orchestrator, event_bus)
        # await example_iceberg(orchestrator, event_bus)
        # await example_grid(orchestrator, event_bus)
        # await example_dca(orchestrator, event_bus)
        # await example_algorithmic_trader(orchestrator, event_bus)
        # await example_multiple_algorithms(orchestrator, event_bus)
        # await example_custom_events(orchestrator, event_bus)

        pass
    else:
        print("Please provide orchestrator and event_bus instances to run examples")


if __name__ == "__main__":
    # Run the examples
    asyncio.run(main())
