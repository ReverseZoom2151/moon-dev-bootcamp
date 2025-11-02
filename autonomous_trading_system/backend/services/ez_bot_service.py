"""
EZ Bot Service
---
This service implements the "Easy Bot" trading logic from the Day 48 `ez.py` script.
It's designed to help users average into positions by buying in demand zones or via
market orders, and to close positions when they enter supply zones.

This service is intended to be controlled via API endpoints and does not run
a continuous loop on its own. It relies on the SolanaJupiterService for all
on-chain and data-fetching operations.
"""
import logging
import time
from autonomous_trading_system.backend.core.config import get_settings
from autonomous_trading_system.backend.services.solana_jupiter_service import SolanaJupiterService

logger = logging.getLogger(__name__)

class EZBotService:
    """A service to execute the trading strategies from the EZ Bot script."""

    def __init__(self, jupiter_service: SolanaJupiterService, settings=None):
        """
        Initializes the EZBotService.
        
        Args:
            jupiter_service: An instance of SolanaJupiterService.
            settings: Optional application settings. If None, they are loaded automatically.
        """
        if settings is None:
            self.settings = get_settings()
        else:
            self.settings = settings
        self.jupiter = jupiter_service
        logger.info("EZBotService initialized.")

    def _get_current_position_usd(self, contract_address: str):
        """Fetches current position size in units and its USD value."""
        pos_units = self.jupiter.get_token_balance(contract_address)
        price = self.jupiter.get_token_price(contract_address)
        if pos_units > 0 and price is not None and price > 0:
            return pos_units, pos_units * price
        elif pos_units > 0:
            logger.warning(f"Have {pos_units} units of {contract_address} but cannot get a valid price.")
            return pos_units, 0.0
        return 0.0, 0.0

    def _handle_market_buy_loop(self, contract_address: str, target_pos_usd: float, max_order_usd: float):
        """Handles the logic to reach a target USD position by market buying in chunks."""
        pos_units, pos_usd = self._get_current_position_usd(contract_address)
        
        fill_ratio = self.settings.EZ_BOT_TARGET_FILL_RATIO
        min_buy_threshold = self.settings.EZ_BOT_MIN_BUY_USD_THRESHOLD
        
        while pos_usd < (fill_ratio * target_pos_usd):
            logger.info(f"Current position: {pos_units:.4f} units (${pos_usd:.2f} USD). Target: ${target_pos_usd:.2f} USD.")
            size_needed_usd = target_pos_usd - pos_usd
            
            if size_needed_usd <= min_buy_threshold:
                logger.info(f"Position very close to target (need ${size_needed_usd:.2f}). Stopping buys.")
                break

            buy_chunk_usd = min(size_needed_usd, max_order_usd)
            usdc_lamports = int(buy_chunk_usd * (10**6)) # USDC has 6 decimals

            if usdc_lamports <= 0:
                logger.warning(f"Calculated buy chunk in lamports is {usdc_lamports}. Skipping.")
                break

            logger.info(f"Attempting to buy chunk of ${buy_chunk_usd:.2f} ({usdc_lamports} lamports).")
            try:
                tx_id = self.jupiter.market_buy(contract_address, usdc_lamports)
                if tx_id:
                    logger.info(f"Market buy order processed for {contract_address}. Tx: {tx_id}")
                else:
                    logger.error(f"Market buy attempt for {contract_address} failed after retries.")
                
                logger.info(f"Sleeping for {self.settings.EZ_BOT_SLEEP_AFTER_BUY_ATTEMPT}s after buy attempt.")
                time.sleep(self.settings.EZ_BOT_SLEEP_AFTER_BUY_ATTEMPT)

                # Re-check position after the buy attempt and sleep
                pos_units, pos_usd = self._get_current_position_usd(contract_address)
            
            except Exception as e:
                logger.error(f"Error during market buy loop for {contract_address}: {e}", exc_info=True)
                time.sleep(self.settings.EZ_BOT_GENERAL_RETRY_SLEEP)
        
        logger.info(f"Market buy loop for {contract_address} finished. Final position: ${pos_usd:.2f} USD.")
        return {"status": "completed", "final_usd_value": pos_usd}

    def close_full_position(self, contract_address: str):
        """Action 0: Closes the full position for a given contract address."""
        logger.info(f"Action 0: Closing full position for {contract_address}.")
        pos_units, pos_usd = self._get_current_position_usd(contract_address)
        logger.info(f"Current position: {pos_units:.4f} units (${pos_usd:.2f} USD).")

        if pos_units > self.settings.EZ_BOT_MIN_SIGNIFICANT_UNITS:
            self.jupiter.chunk_kill(
                contract_address,
                self.settings.EZ_BOT_MAX_USD_ORDER_SIZE,
                self.settings.EZ_BOT_SLEEP_AFTER_BUY_ATTEMPT # Re-using sleep setting
            )
            time.sleep(15)
            pos_units_after, _ = self._get_current_position_usd(contract_address)
            if pos_units_after < self.settings.EZ_BOT_MIN_SIGNIFICANT_UNITS:
                logger.info(f"Position for {contract_address} successfully closed.")
                return {"status": "success", "remaining_units": pos_units_after}
            else:
                logger.warning(f"Position for {contract_address} may not be fully closed. Remaining units: {pos_units_after}")
                return {"status": "warning", "detail": "Position may not be fully closed", "remaining_units": pos_units_after}
        else:
            logger.info(f"No significant position to close for {contract_address}.")
            return {"status": "success", "detail": "No significant position to close."}

    def market_buy_to_target(self, contract_address: str, target_usd_size: float = None):
        """Action 1: Market buys a token to a specific target USD size."""
        target = target_usd_size if target_usd_size is not None else self.settings.EZ_BOT_TOTAL_USD_POSITION_SIZE
        logger.info(f"Action 1: Market buying {contract_address} to target ${target:.2f} USD.")
        return self._handle_market_buy_loop(
            contract_address,
            target,
            self.settings.EZ_BOT_MAX_USD_ORDER_SIZE
        )

    def demand_zone_buy(self, contract_address: str):
        """Action 2: Waits for price to enter a demand zone, then buys to target."""
        logger.info(f"Action 2: Monitoring {contract_address} to buy in demand zone.")
        pos_units, pos_usd = self._get_current_position_usd(contract_address)
        target_usd = self.settings.EZ_BOT_TOTAL_USD_POSITION_SIZE
        
        if pos_usd >= (self.settings.EZ_BOT_TARGET_FILL_RATIO * target_usd):
            msg = f"Position for {contract_address} already filled. Skipping S/D buy."
            logger.info(msg)
            return {"status": "skipped", "detail": msg, "current_usd_value": pos_usd}

        sd_zones = self.jupiter.get_supply_demand_zones(
            contract_address,
            self.settings.EZ_BOT_SDZ_DAYS_BACK,
            self.settings.EZ_BOT_SDZ_TIMEFRAME
        )
        current_price = self.jupiter.get_token_price(contract_address)
        
        if not sd_zones or not current_price:
            msg = "Could not get valid S/D zones or price. Cannot perform S/D buy."
            logger.error(msg)
            return {"status": "error", "detail": msg}

        logger.info(f"S/D Check: Price ${current_price:.6f}. DZ: ${sd_zones['dz_low']:.6f} - ${sd_zones['dz_high']:.6f}")
        
        if sd_zones['dz_low'] <= current_price <= sd_zones['dz_high']:
            logger.info(f"Price is in demand zone. Initiating buys.")
            return self._handle_market_buy_loop(
                contract_address,
                target_usd,
                self.settings.EZ_BOT_MAX_USD_ORDER_SIZE
            )
        else:
            msg = f"Price ${current_price:.6f} is NOT in demand zone."
            logger.info(msg)
            return {"status": "monitoring", "detail": msg}
            
    def supply_zone_close(self, contract_address: str):
        """Action 3: Waits for price to enter a supply zone, then closes the position."""
        logger.info(f"Action 3: Monitoring {contract_address} to close in supply zone.")
        pos_units, _ = self._get_current_position_usd(contract_address)
        
        if pos_units < self.settings.EZ_BOT_MIN_SIGNIFICANT_UNITS:
            msg = f"No significant position for {contract_address} to close."
            logger.info(msg)
            return {"status": "skipped", "detail": msg}

        sd_zones = self.jupiter.get_supply_demand_zones(
            contract_address,
            self.settings.EZ_BOT_SDZ_DAYS_BACK,
            self.settings.EZ_BOT_SDZ_TIMEFRAME
        )
        current_price = self.jupiter.get_token_price(contract_address)

        if not sd_zones or not current_price:
            msg = "Could not get valid S/D zones or price. Cannot perform S/Z close."
            logger.error(msg)
            return {"status": "error", "detail": msg}

        logger.info(f"S/Z Check: Price ${current_price:.6f}. SZ: ${sd_zones['sz_low']:.6f} - ${sd_zones['sz_high']:.6f}")
        
        if sd_zones['sz_low'] <= current_price <= sd_zones['sz_high']:
            logger.info(f"Price is in supply zone. Initiating close.")
            return self.close_full_position(contract_address)
        else:
            msg = f"Price ${current_price:.6f} is NOT in supply zone."
            logger.info(msg)
            return {"status": "monitoring", "detail": msg}
