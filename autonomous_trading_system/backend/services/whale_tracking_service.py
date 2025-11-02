# autonomous_trading_system/backend/services/whale_tracking_service.py

import asyncio
import pandas as pd
import numpy as np
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List
from .moondev_api_service import MoonDevAPIService
from .hyperliquid_service import HyperliquidService

logger = logging.getLogger(__name__)

class WhaleTrackingService:
    def __init__(self, settings: Dict[str, Any], moondev_api_service: MoonDevAPIService, hyperliquid_service: HyperliquidService):
        self.settings = settings
        self.moondev_api_service = moondev_api_service
        self.hyperliquid_service = hyperliquid_service
        
        # Configuration from settings or defaults
        self.data_dir = Path(self.settings.get("WHALE_TRACKING_DATA_DIR", "data/whale_tracking"))
        self.min_position_value = self.settings.get("WHALE_TRACKING_MIN_POS_VALUE", 50000)
        self.top_n_positions = self.settings.get("WHALE_TRACKING_TOP_N", 30)
        self.tokens_to_analyze = self.settings.get("WHALE_TRACKING_TOKENS", ['BTC', 'ETH', 'SOL', 'XRP'])
        self.highlight_threshold = self.settings.get("WHALE_TRACKING_HIGHLIGHT_THRESHOLD", 2000000)
        self.schedule_interval_minutes = self.settings.get("WHALE_TRACKING_INTERVAL_MINUTES", 5)
        
        self.latest_analysis = {}
        
        self._ensure_data_dir()
        logger.info("WhaleTrackingService initialized with advanced analysis capabilities.")

    def _ensure_data_dir(self):
        try:
            self.data_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.error(f"Error creating data directory {self.data_dir}: {e}")

    async def start(self):
        """Starts the periodic analysis loop for the whale tracking service."""
        logger.info(f"Starting WhaleTrackingService analysis loop. Interval: {self.schedule_interval_minutes} minutes.")
        while True:
            try:
                await self.run_full_analysis()
            except Exception as e:
                logger.error(f"Error during whale tracking analysis cycle: {e}", exc_info=True)
            
            await asyncio.sleep(self.schedule_interval_minutes * 60)

    async def run_full_analysis(self):
        """Runs the complete whale position and liquidation risk analysis."""
        logger.info("Starting new whale analysis cycle...")
        start_time = datetime.now()
        
        # 1. Fetch latest positions from MoonDev API
        positions_df = self.moondev_api_service.get_positions_hlp()
        if positions_df is None or positions_df.empty:
            logger.warning("No positions data received from MoonDev API. Skipping analysis cycle.")
            return

        # 2. Process and filter positions
        positions_df = self._process_positions(positions_df)

        # 3. Get current prices for analysis
        current_prices = {}
        for coin in self.tokens_to_analyze:
            ask, bid, _ = self.hyperliquid_service.get_order_book(coin)
            if ask and bid:
                current_prices[coin] = (ask + bid) / 2
        
        if not current_prices:
            logger.error("Could not fetch current prices for any tokens. Aborting analysis.")
            return
            
        # 4. Perform analyses
        funding_rates = self.hyperliquid_service.get_funding_rates(self.tokens_to_analyze)
        liquidation_risk = self.get_liquidation_risk_analysis(positions_df, current_prices)
        liquidation_impact = self.get_liquidation_impact_analysis(positions_df, current_prices)
        liquidation_thresholds = self.get_liquidation_thresholds_table(positions_df, current_prices)
        highlighted_positions = self.get_highlighted_positions(positions_df, current_prices)
        market_direction_rec = self.get_market_direction_recommendation(liquidation_impact)

        self.latest_analysis = {
            "last_updated": datetime.now().isoformat(),
            "funding_rates": funding_rates,
            "liquidation_risk": liquidation_risk,
            "liquidation_impact": liquidation_impact,
            "liquidation_thresholds": liquidation_thresholds,
            "highlighted_positions": highlighted_positions,
            "market_direction": market_direction_rec,
            "top_positions": self.get_top_positions(positions_df),
            "raw_data_summary": {
                "total_positions": len(positions_df),
                "total_value": positions_df['position_value'].sum(),
            }
        }
        
        # 5. Save artifacts to CSV
        self._save_artifacts(self.latest_analysis)
        
        execution_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"Whale analysis cycle completed in {execution_time:.2f} seconds.")
        return self.latest_analysis

    def _process_positions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Internal method to clean, filter, and correct raw position data."""
        if df is None or df.empty:
            return pd.DataFrame()
        
        logger.info(f"Processing {len(df)} raw positions...")
        filtered_df = df[df['position_value'] >= self.min_position_value].copy()
        
        numeric_cols = ['entry_price', 'position_value', 'unrealized_pnl', 'liquidation_price', 'leverage']
        for col in numeric_cols:
            if col in filtered_df.columns:
                filtered_df[col] = pd.to_numeric(filtered_df[col], errors='coerce')
        
        if 'is_long' in filtered_df.columns and filtered_df['is_long'].dtype != bool:
            filtered_df['is_long'] = filtered_df['is_long'].apply(lambda x: x if isinstance(x, bool) else str(x).lower() in ['true', '1'])
        
        filtered_df.dropna(subset=numeric_cols, inplace=True)

        # Add position type correction logic from ppls_pos_3perc.py
        valid_liq_df = filtered_df[filtered_df['liquidation_price'] > 0].copy()
        if not valid_liq_df.empty:
            is_actually_long = valid_liq_df['liquidation_price'] < valid_liq_df['entry_price']
            inconsistent_mask = valid_liq_df['is_long'] != is_actually_long
            inconsistent_count = inconsistent_mask.sum()
            if inconsistent_count > 0:
                logger.warning(f"Found and correcting {inconsistent_count} positions with inconsistent types.")
                filtered_df.loc[inconsistent_mask[inconsistent_mask].index, 'is_long'] = is_actually_long[inconsistent_mask]

        logger.info(f"Finished processing. {len(filtered_df)} positions remain after filtering.")
        return filtered_df

    def get_highlighted_positions(self, df: pd.DataFrame, current_prices: Dict) -> Dict:
        """Creates a table of large positions (value > threshold) closest to liquidation."""
        if df is None or df.empty: return {"longs": [], "shorts": []}
            
        highlight_df = df[(df['position_value'] > self.highlight_threshold) & (df['coin'].isin(self.tokens_to_analyze))].copy()
        if highlight_df.empty: return {"longs": [], "shorts": []}

        highlight_df['current_price'] = highlight_df['coin'].map(current_prices)
        highlight_df.dropna(subset=['current_price'], inplace=True)
        
        highlight_df['distance_to_liq_pct'] = np.where(
            highlight_df['is_long'],
            abs((highlight_df['current_price'] - highlight_df['liquidation_price']) / highlight_df['current_price'] * 100),
            abs((highlight_df['liquidation_price'] - highlight_df['current_price']) / highlight_df['current_price'] * 100)
        )
        
        longs = highlight_df[highlight_df['is_long']].sort_values('distance_to_liq_pct').head(4)
        shorts = highlight_df[~highlight_df['is_long']].sort_values('distance_to_liq_pct').head(4)

        for i, row in longs.iterrows(): longs.loc[i, 'usdc_balance'] = self.hyperliquid_service.get_spot_usdc_balance(row['address'])
        for i, row in shorts.iterrows(): shorts.loc[i, 'usdc_balance'] = self.hyperliquid_service.get_spot_usdc_balance(row['address'])
            
        return {"longs": longs.to_dict('records'), "shorts": shorts.to_dict('records')}

    def get_liquidation_risk_analysis(self, df: pd.DataFrame, current_prices: Dict) -> Dict:
        """Analyzes positions closest to liquidation."""
        risk_df = df[(df['liquidation_price'] > 0) & (df['coin'].isin(self.tokens_to_analyze))].copy()
        if risk_df.empty: return {"longs": [], "shorts": []}

        risk_df['current_price'] = risk_df['coin'].map(current_prices)
        risk_df.dropna(subset=['current_price'], inplace=True)

        risk_df['distance_to_liq_pct'] = np.where(
            risk_df['is_long'],
            abs((risk_df['current_price'] - risk_df['liquidation_price']) / risk_df['current_price'] * 100),
            abs((risk_df['liquidation_price'] - risk_df['current_price']) / risk_df['current_price'] * 100)
        )
        
        risky_longs = risk_df[risk_df['is_long']].sort_values('distance_to_liq_pct').head(self.top_n_positions)
        risky_shorts = risk_df[~risk_df['is_long']].sort_values('distance_to_liq_pct').head(self.top_n_positions)
        
        for i, row in risky_longs.head(4).iterrows(): risky_longs.loc[i, 'usdc_balance'] = self.hyperliquid_service.get_spot_usdc_balance(row['address'])
        for i, row in risky_shorts.head(4).iterrows(): risky_shorts.loc[i, 'usdc_balance'] = self.hyperliquid_service.get_spot_usdc_balance(row['address'])

        return {"longs": risky_longs.to_dict('records'), "shorts": risky_shorts.to_dict('records')}

    def get_liquidation_impact_analysis(self, df: pd.DataFrame, current_prices: Dict) -> Dict:
        """Calculates the value of positions that would be liquidated by a 3% price move."""
        analysis = {"summary": {}, "by_coin": {}}
        total_long_liq_value, total_short_liq_value = 0, 0

        for coin in self.tokens_to_analyze:
            if coin not in current_prices: continue
            current_price = current_prices[coin]
            price_3pct_down, price_3pct_up = current_price * 0.97, current_price * 1.03

            coin_df = df[df['coin'] == coin]
            long_liq_df = coin_df[(coin_df['is_long']) & (coin_df['liquidation_price'].between(price_3pct_down, current_price))]
            short_liq_df = coin_df[(~coin_df['is_long']) & (coin_df['liquidation_price'].between(current_price, price_3pct_up))]

            long_value, short_value = long_liq_df['position_value'].sum(), short_liq_df['position_value'].sum()
            analysis["by_coin"][coin] = {"long_liquidation_value": long_value, "short_liquidation_value": short_value}
            total_long_liq_value += long_value
            total_short_liq_value += short_value

        analysis["summary"] = {"total_long_liquidation_value": total_long_liq_value, "total_short_liquidation_value": total_short_liq_value}
        return analysis
    
    def get_market_direction_recommendation(self, liquidation_impact: Dict) -> Dict:
        """Generates trading recommendations based on liquidation imbalance."""
        recs = {"overall": "NEUTRAL", "by_coin": {}}
        summary = liquidation_impact.get("summary", {})
        total_longs = summary.get("total_long_liquidation_value", 0)
        total_shorts = summary.get("total_short_liquidation_value", 0)

        if total_longs > total_shorts * 1.1: recs["overall"] = "SHORT"
        elif total_shorts > total_longs * 1.1: recs["overall"] = "LONG"
            
        for coin, values in liquidation_impact.get("by_coin", {}).items():
            long_liq, short_liq = values.get("long_liquidation_value", 0), values.get("short_liquidation_value", 0)
            if long_liq > short_liq * 1.1: recs["by_coin"][coin] = "SHORT"
            elif short_liq > long_liq * 1.1: recs["by_coin"][coin] = "LONG"
            else: recs["by_coin"][coin] = "NEUTRAL"
        return recs

    def get_liquidation_thresholds_table(self, df: pd.DataFrame, current_prices: Dict) -> List[Dict]:
        """Creates a table of liquidation values at different percentage thresholds."""
        thresholds = [0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0]
        table_data = []

        for t in thresholds:
            total_longs, total_shorts = 0, 0
            for coin in self.tokens_to_analyze:
                if coin not in current_prices: continue
                current_price = current_prices[coin]
                price_down, price_up = current_price * (1 - t / 100), current_price * (1 + t / 100)
                
                coin_df = df[df['coin'] == coin]
                total_longs += coin_df[coin_df['is_long'] & coin_df['liquidation_price'].between(price_down, current_price)]['position_value'].sum()
                total_shorts += coin_df[~coin_df['is_long'] & coin_df['liquidation_price'].between(current_price, price_up)]['position_value'].sum()
            
            total_liqs = total_longs + total_shorts
            imbalance = ((total_longs - total_shorts) / total_liqs * 100) if total_liqs > 0 else 0
            
            table_data.append({
                "threshold_pct": t, "long_liquidation_usd": total_longs, "short_liquidation_usd": total_shorts,
                "total_liquidation_usd": total_liqs, "imbalance_pct": imbalance
            })
        return table_data

    def get_top_positions(self, df: pd.DataFrame) -> Dict:
        """Gets top N long and short positions by value."""
        longs = df[df['is_long']].sort_values('position_value', ascending=False).head(self.top_n_positions)
        shorts = df[~df['is_long']].sort_values('position_value', ascending=False).head(self.top_n_positions)
        return {"longs": longs.to_dict('records'), "shorts": shorts.to_dict('records')}

    def _save_artifacts(self, analysis_data: Dict):
        """Saves all analysis artifacts to CSV files."""
        try:
            pd.DataFrame(analysis_data['funding_rates']).to_csv(self.data_dir / "funding_rates.csv")
            pd.DataFrame(analysis_data['liquidation_risk']['longs']).to_csv(self.data_dir / "liquidation_risk_longs.csv", index=False)
            pd.DataFrame(analysis_data['liquidation_risk']['shorts']).to_csv(self.data_dir / "liquidation_risk_shorts.csv", index=False)
            pd.DataFrame(analysis_data['highlighted_positions']['longs']).to_csv(self.data_dir / "highlighted_positions_longs.csv", index=False)
            pd.DataFrame(analysis_data['highlighted_positions']['shorts']).to_csv(self.data_dir / "highlighted_positions_shorts.csv", index=False)
            pd.DataFrame(analysis_data['top_positions']['longs']).to_csv(self.data_dir / "top_whale_positions_longs.csv", index=False)
            pd.DataFrame(analysis_data['top_positions']['shorts']).to_csv(self.data_dir / "top_whale_positions_shorts.csv", index=False)
            pd.DataFrame(analysis_data['liquidation_thresholds']).to_csv(self.data_dir / "liquidation_thresholds_table.csv", index=False)
            
            logger.info(f"Successfully saved analysis artifacts to {self.data_dir}")
        except Exception as e:
            logger.error(f"Failed to save analysis artifacts: {e}", exc_info=True)
