def open_order_deluxe(symbol_info, size, account):
    print(f'opening order for {symbol_info["Symbol"]} size {size}')
    exchange = Exchange(account, constants.MAINNET_API_URL)
    symbol = symbol_info['Symbol']
    entry_price = symbol_info['Entry Price']
    stop_loss = symbol_info['Stop Loss']
    take_profit = symbol_info['Take Profit']
    _, rounding = get_sz_px_decimals(symbol)
    if symbol == 'BTC':
        take_profit = int(take_profit)
        stop_loss = int(stop_loss)
    else:
        take_profit = round(take_profit, rounding)
        stop_loss = round(stop_loss, rounding)
    print(f'symbol: {symbol}, entry price: {entry_price}, stop loss: {stop_loss}, take profit: {take_profit}')
    is_buy = True
    cancel_symbol_orders(symbol, account)
    print(f'entry_price: {entry_price} type{type(entry_price)}, stop_loss: {stop_loss} type{type(stop_loss)}, take_profit: {take_profit} type{type(take_profit)}')
    order_result = exchange.order(
        symbol,
        is_buy,
        size,
        entry_price,
        {"limit": {"tif": "Gtc"}}
    )
    print(f'Limit order result for {symbol}: {order_result}')

    stop_order_type = {"trigger": {"triggerPx": stop_loss, "isMarket": True, "tpsl": "stop_loss"}}
    stop_result = exchange.order(
        symbol,
        not is_buy,
        size,
        stop_loss,
        stop_order_type,
        reduce_only=True
    )
    print(f'Stop loss order result for {symbol}: {stop_result}')

    take_profit_order_type = {"trigger": {"triggerPx": take_profit, "isMarket": True, "tpsl": "take_profit"}}
    take_profit_result = exchange.order(
        symbol,
        not is_buy,
        size,
        take_profit,
        take_profit_order_type,
        reduce_only=True
    )
    print(f"Take profit order result for {symbol}: {take_profit_result}")

    