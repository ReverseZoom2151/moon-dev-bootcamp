import os
import json
import random
import time
import warnings
import logging
import pandas as pd
import numpy as np
import pandas_ta as ta
from talib import abstract as talib_abstract
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.inspection import permutation_importance
from typing import List, Tuple, Any, Optional
from backend.core.config import config

# Setup logging
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MLEvaluationService:
    def __init__(self):
        self.data_path = config.get('DATA_PATH', './data/BTC-5m-30wks-data.csv')
        self.results_dir = config.get('RESULTS_BASE_DIR_PATH', './ml/results')
        self.pandas_ta_indicators_path = config.get('PANDAS_TA_INDICATORS_PATH', os.path.join(self.results_dir, 'pandas_ta_indicators.json'))
        self.talib_indicators_path = config.get('TALIB_INDICATORS_PATH', os.path.join(self.results_dir, 'talib_indicators.json'))
        self.performance_filepath = os.path.join(self.results_dir, config.get('PERFORMANCE_FILENAME', 'model_performance.csv'))
        self.importance_filepath = os.path.join(self.results_dir, config.get('IMPORTANCE_FILENAME', 'feature_importance.csv'))
        os.makedirs(self.results_dir, exist_ok=True)
        self.generations = config.get('GENERATIONS', 10)
        self.indicators_per_lib = config.get('INDICATORS_PER_LIB', 3)
        self.target_column = config.get('TARGET_COLUMN', 'close')
        self.target_shift = config.get('TARGET_SHIFT', -1)
        self.test_size = config.get('TEST_SIZE', 0.2)
        self.shuffle_split = config.get('SHUFFLE_SPLIT', False)
        self.models = config.get('MODELS', {
            'LinearRegression': LinearRegression(),
            'RandomForestRegressor': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
            'XGBRegressor': XGBRegressor(n_estimators=100, random_state=42, verbosity=0, n_jobs=-1)
        })
        self.calc_permutation_importance = config.get('CALC_PERMUTATION_IMPORTANCE', False)
        self.permutation_n_repeats = config.get('PERMUTATION_N_REPEATS', 5)
        self.permutation_sample_size = config.get('PERMUTATION_SAMPLE_SIZE', 1000)

    def append_to_csv(self, df: pd.DataFrame, filepath: str) -> None:
        """Appends a DataFrame to a CSV file, creating file/header if needed."""
        try:
            file_exists = os.path.exists(filepath)
            df.to_csv(filepath, mode='a', header=not file_exists, index=False)
        except Exception as e:
            logger.error(f"Error appending DataFrame to {filepath}: {e}", exc_info=True)

    def save_dataframe(self, df: pd.DataFrame, filepath: str) -> None:
        """Saves a DataFrame to a CSV file, creating directories if needed."""
        try:
            output_dir = os.path.dirname(filepath)
            os.makedirs(output_dir, exist_ok=True)
            df.to_csv(filepath, index=False)
            logger.info(f"DataFrame saved successfully to: {filepath}")
        except Exception as e:
            logger.error(f"Error saving DataFrame to {filepath}: {e}", exc_info=True)

    def load_data(self, filepath: str) -> Optional[pd.DataFrame]:
        """Loads data from CSV, parsing datetime."""
        if not os.path.exists(filepath):
            logger.error(f"Data file not found: {filepath}")
            return None
        try:
            df = pd.read_csv(filepath, parse_dates=['datetime'])
            logger.info(f"Data loaded from {filepath}. Shape: {df.shape}")
            return df
        except (FileNotFoundError, pd.errors.EmptyDataError) as e:
            logger.error(f"Error loading data: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error loading data: {e}", exc_info=True)
            return None

    def load_indicator_lists(self, pandas_path: str, talib_path: str) -> Tuple[Optional[List[str]], Optional[List[str]]]:
        """Loads indicator names from JSON files."""
        pandas_ta_indicators, talib_indicators = None, None
        try:
            with open(pandas_path, 'r') as f:
                pandas_ta_indicators = json.load(f)
            logger.info(f"Loaded {len(pandas_ta_indicators)} pandas_ta indicators from {pandas_path}")
        except Exception as e:
            logger.error(f"Error loading pandas_ta indicators from {pandas_path}: {e}", exc_info=True)

        try:
            with open(talib_path, 'r') as f:
                talib_indicators = json.load(f)
            logger.info(f"Loaded {len(talib_indicators)} talib indicators from {talib_path}")
        except Exception as e:
            logger.error(f"Error loading talib indicators from {talib_path}: {e}", exc_info=True)

        return pandas_ta_indicators, talib_indicators

    def prepare_target(self, df: pd.DataFrame, target_col: str, shift: int) -> Tuple[pd.DataFrame, str]:
        """Creates the future target column and removes initial NaN rows."""
        logger.info(f"Preparing target column '{target_col}' shifted by {shift}")
        target_name = f"{target_col}_future_{abs(shift)}"
        df[target_name] = df[target_col].shift(shift)
        initial_rows = len(df)
        df = df.iloc[:-abs(shift)]  # Drop last rows with NaNs from target shift
        logger.info(f"Dropped {initial_rows - len(df)} rows due to target shifting.")
        return df, target_name

    def calculate_selected_indicators(self, df: pd.DataFrame, selected_pandas_ta: List[str], selected_talib: List[str]) -> Tuple[pd.DataFrame, List[str]]:
        """Calculates selected indicators for the given DataFrame generation."""
        df_out = df.copy()
        features_added = []
        logger.debug(f"Calculating pandas_ta: {selected_pandas_ta}")
        logger.debug(f"Calculating talib: {selected_talib}")

        # Pandas TA
        if selected_pandas_ta:
            strategy = ta.Strategy(
                name="Selected_PandasTA",
                ta=[{'kind': ind} for ind in selected_pandas_ta]
            )
            try:
                df_out.ta.strategy(strategy, append=True)
                added_cols = [col for col in df_out.columns if col not in df.columns]
                features_added.extend(added_cols)
                logger.debug(f"pandas_ta added columns: {added_cols}")
            except Exception as e:
                logger.warning(f"Error applying pandas_ta strategy: {e}")

        # Talib
        required_talib_cols = {'open', 'high', 'low', 'close', 'volume'}
        df_cols_lower = {col.lower() for col in df_out.columns}
        if not required_talib_cols.issubset(df_cols_lower):
            logger.warning(f"Skipping Talib indicators: Missing required columns {required_talib_cols - df_cols_lower}")
        else:
            col_map = {c.lower(): c for c in df_out.columns}
            inputs = {k: df_out[col_map[k]] for k in required_talib_cols if k in col_map}
            for indi_name in selected_talib:
                try:
                    func = talib_abstract.Function(indi_name)
                    output = func(inputs)
                    if isinstance(output, pd.Series):
                        col_name = indi_name
                        df_out[col_name] = output
                        features_added.append(col_name)
                        logger.debug(f"Talib added Series: {col_name}")
                    elif isinstance(output, pd.DataFrame):
                        df_out = df_out.join(output)
                        features_added.extend(output.columns.tolist())
                        logger.debug(f"Talib added DataFrame columns: {output.columns.tolist()}")
                    elif isinstance(output, np.ndarray):
                        col_name = indi_name
                        df_out[col_name] = output
                        features_added.append(col_name)
                        logger.debug(f"Talib added ndarray: {col_name}")
                    else:
                        logger.warning(f"Talib indicator '{indi_name}' produced unexpected output type: {type(output)}")
                except Exception as e:
                    logger.warning(f"Could not calculate talib indicator '{indi_name}': {e}")

        final_features = sorted(list(set(f for f in features_added if f in df_out.columns)))
        logger.debug(f"Features added in this generation: {final_features}")
        return df_out, final_features

    def prepare_features_target_gen(self, df_gen: pd.DataFrame, feature_cols: List[str], target_name: str) -> Optional[Tuple[pd.DataFrame, pd.Series]]:
        """Prepares features (X) and target (y) for a generation, drops NaNs."""
        logger.debug("Preparing features and target for generation...")
        cols_to_keep = feature_cols + [target_name]
        missing_cols = [col for col in cols_to_keep if col not in df_gen.columns]
        if missing_cols:
            logger.error(f"Columns missing after indicator calculation: {missing_cols}")
            return None

        df_prepared = df_gen[cols_to_keep].copy()
        initial_rows = len(df_prepared)
        df_prepared.dropna(inplace=True)
        rows_dropped = initial_rows - len(df_prepared)
        if rows_dropped > 0:
            logger.debug(f"Dropped {rows_dropped} rows with NaN values from indicators.")

        if df_prepared.empty:
            logger.warning("DataFrame is empty after preparing features and target for this generation.")
            return None
        if not feature_cols:
            logger.warning("No feature columns remain after processing. Skipping generation.")
            return None

        final_feature_cols = [f for f in feature_cols if f in df_prepared.columns]
        if not final_feature_cols:
            logger.warning("No usable feature columns found after NaN drop. Skipping generation.")
            return None

        X = df_prepared[final_feature_cols]
        y = df_prepared[target_name]
        logger.debug(f"Final shape for modeling: {X.shape}, {y.shape}")
        return X, y

    def split_data(self, X: pd.DataFrame, y: pd.Series, test_size: float, shuffle: bool) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Splits data into training and testing sets."""
        logger.debug(f"Splitting data: test_size={test_size}, shuffle={shuffle}")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=shuffle)
        logger.debug(f"Train set size: {len(X_train)}, Test set size: {len(X_test)}")
        return X_train, X_test, y_train, y_test

    def get_feature_importance(self, model_name: str, model: Any, X: pd.DataFrame, y: pd.Series, features: List[str]) -> Optional[np.ndarray]:
        """Extracts or calculates feature importance for a given model."""
        importance_values = None
        if model_name in ['RandomForestRegressor', 'XGBRegressor']:
            try:
                importance_values = model.feature_importances_
            except AttributeError:
                pass
        elif model_name == 'LinearRegression':
            try:
                importance_values = np.abs(model.coef_)
            except AttributeError:
                pass

        if importance_values is None and self.calc_permutation_importance:
            logger.info(f"Calculating permutation importance for {model_name}...")
            try:
                sample_size = min(self.permutation_sample_size, len(X))
                if sample_size < len(X):
                    logger.debug(f"Sampling {sample_size} rows for permutation importance.")
                    X_sample = X.sample(n=sample_size, random_state=42)
                    y_sample = y.loc[X_sample.index]
                else:
                    X_sample, y_sample = X, y

                perm_importance = permutation_importance(
                    model, X_sample, y_sample,
                    n_repeats=self.permutation_n_repeats,
                    random_state=42, scoring='r2', n_jobs=-1
                )
                importance_values = perm_importance.importances_mean
            except Exception as e:
                logger.error(f"Error calculating permutation importance for {model_name}: {e}", exc_info=True)
                return None
        elif importance_values is None:
            logger.info(f"Feature importance calculation not available or configured for {model_name}.")
            return None

        if importance_values is not None and len(importance_values) != len(features):
            logger.error(f"Feature importance length mismatch for {model_name}. Expected {len(features)}, got {len(importance_values)}")
            return None

        return importance_values

    def train_evaluate_save(self, gen: int, name: str, model: Any, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series, features: List[str], selected_indicators: List[str]) -> None:
        """Trains a single model, evaluates, gets importance, and saves results."""
        logger.info(f"--- Training {name} (Gen: {gen}) ---")
        try:
            start_time = time.time()
            model.fit(X_train, y_train)
            train_time = time.time() - start_time
            logger.info(f"{name} trained in {train_time:.2f}s")

            logger.info(f"Evaluating {name}...")
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            logger.info(f"{name} - MSE: {mse:.4f}, R2: {r2:.4f}")

            perf_data = {
                'Generation': gen, 'Model': name, 'MSE': mse, 'R2': r2,
                'NumFeatures': len(features),
                'Indicators': ','.join(selected_indicators)
            }
            self.append_to_csv(pd.DataFrame([perf_data]), self.performance_filepath)

            importances = self.get_feature_importance(name, model, X_test, y_test, features)
            if importances is not None:
                if len(importances) == len(features):
                    imp_data = pd.DataFrame({
                        'Generation': gen,
                        'Model': name,
                        'Feature': features,
                        'Importance': importances,
                        'MSE': mse,
                        'R2': r2
                    })
                    self.append_to_csv(imp_data, self.importance_filepath)
                else:
                    logger.error(f"Feature/Importance length mismatch for {name} in Gen {gen}. Skipping importance save.")
        except Exception as e:
            logger.error(f"Failed to train/evaluate/save model {name} in Gen {gen}: {e}", exc_info=True)
            perf_data = {
                'Generation': gen, 'Model': name, 'MSE': float('nan'), 'R2': float('nan'),
                'NumFeatures': len(features), 'Indicators': ','.join(selected_indicators)
            }
            self.append_to_csv(pd.DataFrame([perf_data]), self.performance_filepath)

    def run_generation(self, gen: int, df_base: pd.DataFrame, target_name: str, pandas_ta_list: List[str], talib_list: List[str]) -> None:
        """Runs a single generation of indicator selection, training, and evaluation."""
        logger.info(f"=== Starting Generation {gen}/{self.generations} ===")

        num_to_select = self.indicators_per_lib
        selected_pandas_ta = random.sample(pandas_ta_list, min(num_to_select, len(pandas_ta_list)))
        selected_talib = random.sample(talib_list, min(num_to_select, len(talib_list)))
        selected_indicators_all = selected_pandas_ta + selected_talib
        logger.info(f"Selected pandas_ta: {selected_pandas_ta}")
        logger.info(f"Selected talib: {selected_talib}")

        df_gen, features_added = self.calculate_selected_indicators(df_base, selected_pandas_ta, selected_talib)
        if not features_added:
            logger.warning(f"No features were successfully calculated in Gen {gen}. Skipping.")
            return

        prep_result = self.prepare_features_target_gen(df_gen, features_added, target_name)
        if prep_result is None:
            logger.warning(f"Feature/Target preparation failed for Gen {gen}. Skipping.")
            return
        X, y = prep_result
        final_features = X.columns.tolist()

        if len(X) < 10:
            logger.warning(f"Insufficient data ({len(X)} rows) after processing for Gen {gen}. Skipping.")
            return
        X_train, X_test, y_train, y_test = self.split_data(X, y, self.test_size, self.shuffle_split)

        for name, model_instance in self.models.items():
            self.train_evaluate_save(gen, name, model_instance, X_train, y_train, X_test, y_test, final_features, selected_indicators_all)

        logger.info(f"=== Completed Generation {gen} ===")

    async def run_evaluation(self) -> None:
        """Main function to run the ML evaluation pipeline across generations."""
        logger.info("=== ML Model Evaluation Script (Generational) Started ===")

        logger.info(f"Results will be saved to directory: {self.results_dir}")
        logger.info(f"Performance data: {self.performance_filepath}")
        logger.info(f"Importance data: {self.importance_filepath}")

        df_base = self.load_data(self.data_path)
        if df_base is None:
            return

        pandas_ta_list, talib_list = self.load_indicator_lists(self.pandas_ta_indicators_path, self.talib_indicators_path)
        if pandas_ta_list is None or talib_list is None:
            logger.error("Failed to load one or both indicator lists. Exiting.")
            return

        df_base_with_target, target_name = self.prepare_target(df_base, self.target_column, self.target_shift)

        if not self.models:
            logger.error("No models defined in configuration. Exiting.")
            return

        for gen in range(1, self.generations + 1):
            self.run_generation(gen, df_base_with_target.copy(), target_name, pandas_ta_list, talib_list)

        logger.info("=== ML Model Evaluation (Generational) Complete ===")
        logger.info(f"Final results saved in: {self.results_dir}") 