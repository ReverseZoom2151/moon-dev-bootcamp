# bitfinex_top_rrs.py
import pandas as pd
import logging
from pathlib import Path
from typing import List, Dict
from datetime import datetime
from bitfinex_rrs_config import RESULTS_DIR

# Setup logger for this module
logger = logging.getLogger(__name__)


def process_bitfinex_professional_results_files(results_dir: str = RESULTS_DIR) -> pd.DataFrame:
    """Processes individual Bitfinex professional RRS result CSV files to find top performers across timeframes.

    Reads all CSV files (except specified exclusions) from the Bitfinex professional results directory,
    extracts the top institutional-grade performers based on professional RRS score from each timeframe analysis,
    compiles them into a single DataFrame, sorts by professional score, and saves
    the consolidated list to 'Bitfinex_Professional_Top_RRS.csv'.

    Args:
        results_dir: The path to the directory containing the Bitfinex professional result CSV files.
                     Defaults to RESULTS_DIR from bitfinex_rrs_config.py.

    Returns:
        A pandas DataFrame containing the top professional performers across all processed files,
        sorted by professional RRS score (descending).
    """
    results_path = Path(results_dir)
    if not results_path.is_dir():
        logger.error(f"Bitfinex professional results directory not found: {results_dir}")
        return pd.DataFrame()  # Return empty if directory doesn't exist

    all_professional_performers: List[Dict] = []
    files_processed_count = 0
    excluded_files = ['bitfinex_professional_historical_summary.csv', 'Bitfinex_Professional_Top_RRS.csv', 'bitfinex_market_report.csv']  # Files to skip

    logger.info(f"Processing Bitfinex professional RRS result files in: {results_dir}")

    # Process each CSV file in the results directory
    for file_path in results_path.glob('*.csv'):
        if file_path.name in excluded_files:
            logger.debug(f"Skipping excluded professional file: {file_path.name}")
            continue

        # Only process Bitfinex-specific professional result files
        if not file_path.name.startswith('bitfinex_'):
            logger.debug(f"Skipping non-Bitfinex professional file: {file_path.name}")
            continue

        logger.debug(f"Processing Bitfinex professional file: {file_path.name}")
        try:
            df = pd.read_csv(file_path)
            files_processed_count += 1

            # Check if required professional columns exist
            required_columns = ['symbol', 'current_rrs', 'primary_signal', 'signal_confidence']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                logger.warning(f"Missing required professional columns {missing_columns} in {file_path.name}")
                continue

            # Handle different possible professional RRS column names
            rrs_column = None
            possible_rrs_columns = ['current_rrs', 'rrs_score', 'rrs', 'smoothed_rrs', 'professional_score']
            for col in possible_rrs_columns:
                if col in df.columns:
                    rrs_column = col
                    break
            
            if rrs_column is None:
                logger.warning(f"No professional RRS score column found in {file_path.name}")
                continue

            # Sort by professional RRS score and get top 5 performers from this file
            df_sorted = df.sort_values(by=rrs_column, ascending=False)
            top_performers = df_sorted.head(5)

            # Extract timeframe from filename (e.g., bitfinex_professional_rrs_results_4h.csv -> 4h)
            timeframe = "unknown"
            if "_results_" in file_path.name:
                try:
                    timeframe = file_path.name.split("_results_")[1].replace(".csv", "")
                except IndexError:
                    timeframe = "unknown"

            # Add professional top performers to the list
            for _, row in top_performers.iterrows():
                performer_data = {
                    'symbol': row['symbol'],
                    'timeframe': timeframe,
                    'rrs_score': row[rrs_column],
                    'signal': row.get('primary_signal', 'N/A'),
                    'confidence': row.get('signal_confidence', 0),
                    'risk_level': row.get('risk_level', 'Medium'),
                    'rank': row.get('rank', 0),
                    'professional_grade': row.get('professional_grade', 'Standard'),
                    'position_size_suggestion': row.get('position_size_suggestion', 1.0),
                    'source_file': file_path.name,
                    'last_price': row.get('last_price', 0),
                    'volume': row.get('volume', 0),
                    'beta': row.get('beta', 1.0),
                    'alpha': row.get('alpha', 0.0),
                    'correlation': row.get('correlation', 0.0),
                    'max_drawdown': row.get('max_drawdown', 0),
                    'tracking_error': row.get('tracking_error', 0),
                    'information_ratio': row.get('information_ratio', 0),
                    'var_5pct': row.get('var_5pct', 0),
                    'volatility': row.get('volatility', 0),
                    'momentum_score': row.get('momentum_score', 0),
                    'trend_consistency': row.get('trend_consistency', 0.5),
                    'rrs_stability': row.get('rrs_stability', 0.5)
                }
                all_professional_performers.append(performer_data)

            logger.info(f"Extracted {len(top_performers)} professional performers from {file_path.name}")

        except pd.errors.EmptyDataError:
            logger.warning(f"Empty professional file encountered: {file_path.name}")
        except pd.errors.ParserError as e:
            logger.error(f"Parse error in professional file {file_path.name}: {e}")
        except Exception as e:
            logger.error(f"Unexpected error processing professional file {file_path.name}: {e}")

    logger.info(f"Processed {files_processed_count} Bitfinex professional files")

    if not all_professional_performers:
        logger.warning("No professional top performers found across all processed files")
        return pd.DataFrame()

    # Create consolidated professional DataFrame
    consolidated_df = pd.DataFrame(all_professional_performers)
    
    # Sort by professional RRS score (descending) and reset index
    consolidated_df = consolidated_df.sort_values(by='rrs_score', ascending=False).reset_index(drop=True)
    
    # Add overall professional ranking
    consolidated_df['overall_rank'] = range(1, len(consolidated_df) + 1)
    
    # Calculate additional professional metrics
    consolidated_df['exchange'] = 'Bitfinex'
    consolidated_df['analysis_type'] = 'professional_institutional'
    consolidated_df['analysis_timestamp'] = datetime.utcnow()
    
    # Professional composite score calculation
    consolidated_df['institutional_score'] = (
        consolidated_df['rrs_score'] * 0.25 +
        consolidated_df['confidence'] * 0.20 +
        consolidated_df['information_ratio'].fillna(0) * 0.15 +
        consolidated_df['trend_consistency'] * 0.15 +
        consolidated_df['rrs_stability'] * 0.10 +
        (1 - abs(consolidated_df['beta'] - 1.0)) * 0.10 +  # Prefer beta close to 1
        consolidated_df['alpha'].fillna(0) * 0.05
    )
    
    # Re-sort by institutional score
    consolidated_df = consolidated_df.sort_values('institutional_score', ascending=False).reset_index(drop=True)
    consolidated_df['institutional_rank'] = range(1, len(consolidated_df) + 1)
    
    # Save to professional CSV
    output_file = results_path / 'Bitfinex_Professional_Top_RRS.csv'
    try:
        consolidated_df.to_csv(output_file, index=False)
        logger.info(f"Saved consolidated Bitfinex professional performers to: {output_file}")
        logger.info(f"Total professional performers: {len(consolidated_df)}")
        
        # Log professional summary statistics
        if len(consolidated_df) > 0:
            logger.info("=== BITFINEX PROFESSIONAL TOP RRS SUMMARY ===")
            logger.info(f"Best Professional RRS Score: {consolidated_df['rrs_score'].max():.4f}")
            logger.info(f"Worst Professional RRS Score: {consolidated_df['rrs_score'].min():.4f}")
            logger.info(f"Average Professional RRS Score: {consolidated_df['rrs_score'].mean():.4f}")
            logger.info(f"Average Professional Beta: {consolidated_df['beta'].mean():.3f}")
            logger.info(f"Average Professional Alpha: {consolidated_df['alpha'].mean():.4f}")
            
            # Professional signal distribution
            signal_counts = consolidated_df['signal'].value_counts()
            grade_counts = consolidated_df['professional_grade'].value_counts()
            logger.info(f"Professional Signal Distribution: {dict(signal_counts)}")
            logger.info(f"Professional Grade Distribution: {dict(grade_counts)}")
            
            # Top 10 professional performers
            logger.info("TOP 10 BITFINEX PROFESSIONAL PERFORMERS:")
            for i, (_, row) in enumerate(consolidated_df.head(10).iterrows(), 1):
                logger.info(f"  {i:2d}. {row['symbol']:8s} | {row['timeframe']:3s} | "
                           f"RRS: {row['rrs_score']:6.3f} | Signal: {row['signal']:10s} | "
                           f"Grade: {row['professional_grade']:12s} | Beta: {row['beta']:5.2f} | "
                           f"Alpha: {row['alpha']:6.3f}")
                
    except Exception as e:
        logger.error(f"Failed to save consolidated professional results: {e}")

    return consolidated_df

def analyze_bitfinex_professional_performance_trends(df: pd.DataFrame) -> Dict:
    """Analyzes professional performance trends across Bitfinex timeframes and symbols for institutional insight."""
    if df.empty:
        return {}
    
    analysis = {}
    
    try:
        # Professional timeframe performance analysis
        timeframe_stats = df.groupby('timeframe').agg({
            'rrs_score': ['count', 'mean', 'max', 'min', 'std'],
            'beta': ['mean', 'std'],
            'alpha': ['mean', 'std'],
            'information_ratio': ['mean', 'std'],
            'institutional_score': ['mean', 'max']
        }).round(4)
        analysis['timeframe_performance'] = timeframe_stats.to_dict()
        
        # Professional symbol consistency analysis (which symbols appear most often in top lists)
        symbol_frequency = df['symbol'].value_counts()
        symbol_performance = df.groupby('symbol').agg({
            'rrs_score': ['mean', 'std'],
            'institutional_score': 'mean',
            'professional_grade': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'Standard'
        }).round(4)
        analysis['most_consistent_professional_performers'] = symbol_frequency.head(10).to_dict()
        analysis['symbol_performance_metrics'] = symbol_performance.head(10).to_dict()
        
        # Professional signal effectiveness analysis
        signal_performance = df.groupby('signal').agg({
            'rrs_score': ['count', 'mean', 'std'],
            'confidence': ['mean', 'std'],
            'institutional_score': 'mean'
        }).round(4)
        analysis['professional_signal_effectiveness'] = signal_performance.to_dict()
        
        # Professional grade distribution and performance
        grade_performance = df.groupby('professional_grade').agg({
            'rrs_score': ['count', 'mean'],
            'confidence': 'mean',
            'institutional_score': 'mean'
        }).round(4)
        analysis['professional_grade_performance'] = grade_performance.to_dict()
        
        # Risk-adjusted performance analysis
        risk_analysis = {
            'high_beta_performers': len(df[df['beta'] > 1.5]),
            'low_beta_performers': len(df[df['beta'] < 0.5]),
            'positive_alpha_count': len(df[df['alpha'] > 0]),
            'negative_alpha_count': len(df[df['alpha'] < 0]),
            'high_information_ratio': len(df[df['information_ratio'] > 0.5]),
            'avg_tracking_error': df['tracking_error'].mean()
        }
        analysis['professional_risk_analysis'] = risk_analysis
        
        # Institutional quality metrics
        institutional_metrics = {
            'institutional_grade_count': len(df[df['professional_grade'] == 'Institutional']),
            'professional_grade_count': len(df[df['professional_grade'] == 'Professional']),
            'avg_position_size_suggestion': df['position_size_suggestion'].mean(),
            'high_confidence_signals': len(df[df['confidence'] > 0.8])
        }
        analysis['institutional_quality_metrics'] = institutional_metrics
        
        logger.info("Bitfinex professional performance trends analysis completed")
        
    except Exception as e:
        logger.error(f"Error in professional trend analysis: {e}")
    
    return analysis

def export_bitfinex_professional_performers_json(df: pd.DataFrame, results_dir: str = RESULTS_DIR):
    """Exports top Bitfinex professional performers to JSON format for institutional API consumption."""
    if df.empty:
        return
    
    try:
        # Create professional JSON-friendly format
        json_data = {
            'metadata': {
                'exchange': 'Bitfinex',
                'analysis_type': 'professional_institutional_rrs',
                'analysis_timestamp': datetime.utcnow().isoformat(),
                'total_performers': len(df),
                'data_source': 'Bitfinex Professional RRS Analysis System',
                'institutional_features': [
                    'Beta Analysis', 'Alpha Calculation', 'Information Ratio',
                    'Professional Grade Classification', 'Risk-Adjusted Scoring',
                    'Position Size Recommendations', 'Tracking Error Analysis'
                ]
            },
            'top_performers': df.head(25).to_dict('records'),  # Top 25 for professional JSON
            'institutional_performers': df[df['professional_grade'] == 'Institutional'].head(10).to_dict('records'),
            'performance_trends': analyze_bitfinex_professional_performance_trends(df),
            'risk_metrics_summary': {
                'avg_beta': df['beta'].mean(),
                'avg_alpha': df['alpha'].mean(),
                'avg_information_ratio': df['information_ratio'].mean(),
                'avg_tracking_error': df['tracking_error'].mean(),
                'high_grade_percentage': len(df[df['professional_grade'].isin(['Professional', 'Institutional'])]) / len(df)
            }
        }
        
        output_file = Path(results_dir) / 'Bitfinex_Professional_Top_RRS.json'
        
        import json
        with open(output_file, 'w') as f:
            json.dump(json_data, f, indent=2, default=str)
        
        logger.info(f"Exported Bitfinex professional performers JSON to: {output_file}")
        
    except Exception as e:
        logger.error(f"Failed to export professional JSON: {e}")

def generate_professional_trading_recommendations(df: pd.DataFrame) -> pd.DataFrame:
    """Generate professional trading recommendations based on institutional analysis."""
    if df.empty:
        return df
    
    try:
        recommendations_df = df.head(15).copy()  # Top 15 for professional recommendations
        
        # Professional recommendation logic
        recommendations_df['trading_recommendation'] = 'MONITOR'
        
        for i, row in recommendations_df.iterrows():
            grade = row['professional_grade']
            signal = row['signal']
            confidence = row['confidence']
            beta = row['beta']
            alpha = row['alpha']
            info_ratio = row['information_ratio']
            
            if (grade == 'Institutional' and 
                signal in ['STRONG_BUY', 'BUY'] and 
                confidence > 0.75 and 
                alpha > 0 and 
                info_ratio > 0.3):
                recommendations_df.at[i, 'trading_recommendation'] = 'STRONG_INSTITUTIONAL_BUY'
                
            elif (grade in ['Professional', 'Institutional'] and 
                  signal in ['BUY', 'STRONG_BUY'] and 
                  confidence > 0.65):
                recommendations_df.at[i, 'trading_recommendation'] = 'PROFESSIONAL_BUY'
                
            elif (signal in ['SELL', 'STRONG_SELL'] and 
                  confidence > 0.7):
                recommendations_df.at[i, 'trading_recommendation'] = 'PROFESSIONAL_SELL'
        
        return recommendations_df
        
    except Exception as e:
        logger.error(f"Error generating professional recommendations: {e}")
        return df

def main():
    """Main function to process Bitfinex professional RRS results and generate top performers list."""
    # Configure professional logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('bitfinex_professional_top_rrs.log')
        ]
    )
    
    logger.info("🚀 Starting Bitfinex Professional Top RRS Analysis")
    logger.info("🏛️ Institutional-grade analysis with professional risk metrics")
    
    try:
        # Process professional results files
        top_performers_df = process_bitfinex_professional_results_files()
        
        if not top_performers_df.empty:
            # Export professional data to JSON
            export_bitfinex_professional_performers_json(top_performers_df)
            
            # Generate professional trading recommendations
            recommendations_df = generate_professional_trading_recommendations(top_performers_df)
            
            # Save professional recommendations
            recommendations_file = Path(RESULTS_DIR) / 'Bitfinex_Professional_Trading_Recommendations.csv'
            recommendations_df.to_csv(recommendations_file, index=False)
            logger.info(f"Generated professional trading recommendations: {recommendations_file}")
            
            logger.info("✅ Bitfinex Professional Top RRS analysis completed successfully")
            logger.info("🏛️ Professional features: Beta, Alpha, Information Ratio, Risk Analysis")
            return top_performers_df
        else:
            logger.warning("⚠️ No professional top performers found")
            return pd.DataFrame()
            
    except Exception as e:
        logger.error(f"❌ Fatal error in Bitfinex professional top RRS analysis: {e}", exc_info=True)
        raise

if __name__ == '__main__':
    main()
