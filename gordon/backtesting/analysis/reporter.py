"""Report generation for backtest results"""

import logging
import json
from typing import List
from ..base import BacktestResult

logger = logging.getLogger(__name__)


class ResultsReporter:
    """Generate various reports from backtest results"""
    
    @staticmethod
    def export_json(results: List[BacktestResult], filepath: str):
        """
        Export results to JSON file
        
        Args:
            results: List of BacktestResult objects
            filepath: Output file path
        """
        try:
            data = [r.to_dict() for r in results]
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            logger.info(f"Results exported to {filepath}")
        except Exception as e:
            logger.error(f"Error exporting to JSON: {e}")
    
    @staticmethod
    def export_csv(results: List[BacktestResult], filepath: str):
        """
        Export results summary to CSV file
        
        Args:
            results: List of BacktestResult objects
            filepath: Output file path
        """
        try:
            import pandas as pd
            data = []
            for result in results:
                metrics = result.metrics
                data.append({
                    'Strategy': result.strategy_name,
                    'Framework': result.framework,
                    'Initial Value': metrics.initial_value,
                    'Final Value': metrics.final_value,
                    'Total Return %': metrics.total_return,
                    'Sharpe Ratio': metrics.sharpe_ratio,
                    'Max Drawdown %': metrics.max_drawdown,
                    'Trades': metrics.num_trades,
                    'Win Rate %': metrics.win_rate,
                    'Execution Time s': result.execution_time
                })
            
            df = pd.DataFrame(data)
            df.to_csv(filepath, index=False)
            logger.info(f"Results exported to {filepath}")
        except Exception as e:
            logger.error(f"Error exporting to CSV: {e}")
    
    @staticmethod
    def generate_html_report(results: List[BacktestResult], filepath: str):
        """
        Generate HTML report
        
        Args:
            results: List of BacktestResult objects
            filepath: Output file path
        """
        try:
            html = "<html><head><title>Backtest Report</title></head><body>"
            html += "<h1>Backtest Results Report</h1>"
            html += "<table border='1' cellpadding='5'>"
            html += "<tr>"
            html += "<th>Strategy</th><th>Return %</th><th>Sharpe</th>"
            html += "<th>Max DD %</th><th>Trades</th><th>Win Rate %</th>"
            html += "</tr>"
            
            for result in results:
                metrics = result.metrics
                html += "<tr>"
                html += f"<td>{result.strategy_name}</td>"
                html += f"<td>{metrics.total_return:.2f}%</td>"
                html += f"<td>{metrics.sharpe_ratio:.3f}</td>"
                html += f"<td>{metrics.max_drawdown:.2f}%</td>"
                html += f"<td>{metrics.num_trades}</td>"
                html += f"<td>{metrics.win_rate:.2f}%</td>"
                html += "</tr>"
            
            html += "</table></body></html>"
            
            with open(filepath, 'w') as f:
                f.write(html)
            logger.info(f"HTML report generated: {filepath}")
        except Exception as e:
            logger.error(f"Error generating HTML report: {e}")
