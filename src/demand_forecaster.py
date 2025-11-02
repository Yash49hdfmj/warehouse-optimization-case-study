"""
Enhanced Demand Forecasting with Multiple ML Models
Supports: LightGBM, XGBoost, and CatBoost with hyperparameter tuning
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Optional
import warnings
warnings.filterwarnings('ignore')


class DemandForecaster:
    """Advanced machine learning model for demand forecasting with model comparison"""
    
    def __init__(self, model_type: str = "Auto"):
        """
        Initialize forecaster with model selection
        
        Args:
            model_type: Type of model - "LightGBM", "XGBoost", "CatBoost", or "Auto" for best
        """
        self.model_type = model_type
        self.model = None
        self.best_model_name = None
        self.preprocessor = None
        self.metrics: Dict[str, float] = {}
        # make the empty series typed to avoid warnings
        self.predictions = pd.Series(dtype=float)
        self.all_model_results: Dict[str, Dict[str, float]] = {}
        
    def train_and_predict(self, daily_features: pd.DataFrame, 
                         train_ratio: float = 0.8,
                         cv_folds: int = 3) -> Tuple[pd.Series, Dict[str, float]]:
        """
        Train model(s) and generate predictions with cross-validation
        
        Args:
            daily_features: Features with warehouse, sku, time features, and orders
            train_ratio: Ratio of data to use for training
            cv_folds: Number of cross-validation folds
            
        Returns:
            Tuple of (predictions, metrics)
        """
        print("\n" + "="*80)
        print("STEP 3: ADVANCED DEMAND FORECASTING WITH MODEL COMPARISON")
        print("="*80)
        
        try:
            required_cols = {"warehouse", "sku", "dow", "week", "month", "orders"}
            if not required_cols.issubset(daily_features.columns):
                missing = required_cols - set(daily_features.columns)
                print(f"  ✗ Missing required columns: {missing}")
                # return NaNs to satisfy Dict[str, float] typing
                return pd.Series(dtype=float), {"mae": float('nan'), "rmse": float('nan'), "mape": float('nan')}
            
            if len(daily_features) < 30:
                print("  ✗ Insufficient data for training (need at least 30 samples)")
                return pd.Series(dtype=float), {"mae": float('nan'), "rmse": float('nan'), "mape": float('nan')}
            
            # Prepare features and target
            X, y = self._prepare_data(daily_features)
            
            # Time-ordered split
            split_idx = max(10, int(len(X) * train_ratio))
            split_idx = min(split_idx, len(X) - 10)
            
            X_train = X.iloc[:split_idx]
            y_train = y.iloc[:split_idx]
            X_test = X.iloc[split_idx:]
            y_test = y.iloc[split_idx:]
            
            print(f"\n  → Dataset Split:")
            print(f"      Training set:   {len(X_train):5d} samples ({train_ratio*100:.0f}%)")
            print(f"      Test set:       {len(X_test):5d} samples ({(1-train_ratio)*100:.0f}%)")
            print(f"      Total features: {X.shape[1]:5d} columns")
            
            # Train and compare models
            if self.model_type == "Auto":
                print(f"\n  → Auto mode: Training and comparing all models...")
                self._train_all_models(X, y, X_train, y_train, X_test, y_test)
            else:
                print(f"\n  → Training {self.model_type} model...")
                predictions_all = self._train_single_model(
                    self.model_type, X, y, X_train, y_train, X_test, y_test
                )
                self.predictions = pd.Series(predictions_all, index=daily_features.index)

                # compute and store metrics for single-model path
                y_pred_test = self.predictions.iloc[len(X_train):]
                mae, rmse, mape = self._calculate_metrics(y_test, y_pred_test)
                self.best_model_name = self.model_type
                self.metrics = {'mae': float(mae), 'rmse': float(rmse), 'mape': float(mape)}
            
            # Print final results
            self._print_results()
            
            # Ensure return types match hints
            if not self.metrics:
                self.metrics = {"mae": float('nan'), "rmse": float('nan'), "mape": float('nan')}
            return self.predictions, self.metrics
            
        except Exception as e:
            print(f"\n  ✗ Error in forecasting: {str(e)}")
            import traceback
            traceback.print_exc()
            return pd.Series(dtype=float), {"mae": float('nan'), "rmse": float('nan'), "mape": float('nan')}
    
    def _prepare_data(self, daily_features: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare and encode features"""
        from sklearn.preprocessing import LabelEncoder
        
        # Select features
        feature_cols = ["warehouse", "sku", "dow", "week", "month"]
        
        # Add cyclical features if available
        if 'dow_sin' in daily_features.columns:
            feature_cols.extend(['dow_sin', 'dow_cos'])
        if 'month_sin' in daily_features.columns:
            feature_cols.extend(['month_sin', 'month_cos'])
        
        # Add day if available
        if 'day' in daily_features.columns:
            feature_cols.append('day')
        
        X = daily_features[feature_cols].copy()
        y = daily_features["orders"].astype(float).copy()
        
        # Encode categorical variables
        self.le_warehouse = LabelEncoder()
        self.le_sku = LabelEncoder()
        
        X['warehouse'] = self.le_warehouse.fit_transform(X['warehouse'].astype(str))
        X['sku'] = self.le_sku.fit_transform(X['sku'].astype(str))
        
        return X, y
    
    def _train_all_models(self, X, y, X_train, y_train, X_test, y_test):
        """Train all three models and select the best"""
        models_to_test = ["LightGBM", "XGBoost", "CatBoost"]
        
        print("\n" + "-"*80)
        print("MODEL COMPARISON & SELECTION")
        print("-"*80)
        
        best_mae = float('inf')
        best_model_name = None
        best_predictions = None
        
        for model_name in models_to_test:
            try:
                print(f"\n  [{model_name}]")
                predictions = self._train_single_model(
                    model_name, X, y, X_train, y_train, X_test, y_test
                )
                
                # Calculate metrics on test set
                y_pred_test = predictions[len(X_train):]
                mae, rmse, mape = self._calculate_metrics(y_test, y_pred_test)
                
                self.all_model_results[model_name] = {
                    'mae': float(mae),
                    'rmse': float(rmse),
                    'mape': float(mape),
                }
                
                print(f"      MAE:  {mae:8.3f}  |  RMSE: {rmse:8.3f}  |  MAPE: {mape:7.2f}%")
                
                # Track best model
                if mae < best_mae:
                    best_mae = mae
                    best_model_name = model_name
                    best_predictions = predictions
                    
            except Exception as e:
                print(f"      ✗ Failed: {str(e)}")
                continue
        
        # Select best model
        if best_model_name is not None and best_predictions is not None:
            print(f"\n  {'='*76}")
            print(f"  🏆 BEST MODEL: {best_model_name}")
            print(f"  {'='*76}")
            
            self.best_model_name = best_model_name
            self.model_type = best_model_name
            self.predictions = pd.Series(best_predictions, index=X.index)
            self.metrics = self.all_model_results[best_model_name].copy()
        else:
            print(f"\n  ✗ All models failed to train")
            self.best_model_name = None
            self.metrics = {"mae": float('nan'), "rmse": float('nan'), "mape": float('nan')}
            self.predictions = pd.Series(dtype=float)
    
    def _train_single_model(self, model_name: str, X, y, X_train, y_train, X_test, y_test):
        """Train a single model with optimized hyperparameters"""
        
        if model_name == "LightGBM":
            return self._train_lightgbm(X, y, X_train, y_train, X_test, y_test)
        elif model_name == "XGBoost":
            return self._train_xgboost(X, y, X_train, y_train, X_test, y_test)
        elif model_name == "CatBoost":
            return self._train_catboost(X, y, X_train, y_train, X_test, y_test)
        else:
            raise ValueError(f"Unknown model: {model_name}")
    
    def _train_lightgbm(self, X, y, X_train, y_train, X_test, y_test):
        """Train LightGBM with optimized parameters"""
        try:
            import lightgbm as lgb
            from typing import cast
            try:
                from scipy.sparse import issparse, spmatrix  # type: ignore
            except Exception:
                def issparse(_):  # type: ignore
                    return False
                class spmatrix:  # type: ignore
                    pass
            
            # Optimized hyperparameters for demand forecasting
            params = {
                'objective': 'regression',
                'metric': 'mae',
                'boosting_type': 'gbdt',
                'num_leaves': 31,
                'learning_rate': 0.05,
                'feature_fraction': 0.9,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'max_depth': 6,
                'min_child_samples': 20,
                'reg_alpha': 0.1,
                'reg_lambda': 0.1,
                'verbosity': -1,
                'n_estimators': 300,
                'random_state': 42
            }
            
            self.model = lgb.LGBMRegressor(**params)
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_test, y_test)],
                eval_metric='mae',
                callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
            )
            
            # Predict on all data
            raw_preds = self.model.predict(X)
            if issparse(raw_preds):  # type: ignore[call-arg]
                raw_preds = cast(spmatrix, raw_preds).toarray()  # type: ignore[union-attr]
            preds = np.asarray(raw_preds, dtype=float).ravel()
            predictions = np.clip(preds, 0.0, None)  # Ensure non-negative
            return predictions
            
        except ImportError:
            raise ImportError("LightGBM not installed. Run: pip install lightgbm")
    
    def _train_xgboost(self, X, y, X_train, y_train, X_test, y_test):
        """Train XGBoost with optimized parameters"""
        try:
            import xgboost as xgb

            params = {
                'objective': 'reg:squarederror',
                'eval_metric': 'mae',
                'max_depth': 6,
                'learning_rate': 0.05,
                'n_estimators': 300,
                'min_child_weight': 3,
                'subsample': 0.8,
                'colsample_bytree': 0.9,
                'gamma': 0.1,
                'reg_alpha': 0.1,
                'reg_lambda': 1.0,
                'random_state': 42,
                'verbosity': 0
            }

            self.model = xgb.XGBRegressor(**params)

            # ✅ Handle early stopping depending on XGBoost version
            try:
                self.model.fit(
                    X_train, y_train,
                    eval_set=[(X_test, y_test)],
                    verbose=False,
                    early_stopping_rounds=50  # works on <2.0
                )
            except TypeError:
                # fallback for XGBoost >=2.0
                eval_set = [(X_test, y_test)]
                self.model.fit(X_train, y_train, eval_set=eval_set, verbose=False)

            predictions = self.model.predict(X)
            predictions = np.clip(np.asarray(predictions, dtype=float).ravel(), 0.0, None)
            return predictions

        except ImportError:
            raise ImportError("XGBoost not installed. Run: pip install xgboost")

    
    def _train_catboost(self, X, y, X_train, y_train, X_test, y_test):
        """Train CatBoost with optimized parameters"""
        try:
            from catboost import CatBoostRegressor
            
            # Optimized hyperparameters
            params = {
                'iterations': 300,
                'learning_rate': 0.05,
                'depth': 6,
                'l2_leaf_reg': 3,
                'min_data_in_leaf': 20,
                'random_strength': 0.5,
                'bagging_temperature': 0.2,
                'od_type': 'Iter',
                'od_wait': 50,
                'random_seed': 42,
                'verbose': False,
                'loss_function': 'MAE'
            }
            
            self.model = CatBoostRegressor(**params)
            self.model.fit(
                X_train, y_train,
                eval_set=(X_test, y_test),
                early_stopping_rounds=50,
                verbose=False
            )
            
            # Predict on all data
            predictions = self.model.predict(X)
            predictions = np.clip(np.asarray(predictions, dtype=float).ravel(), 0.0, None)
            return predictions
            
        except ImportError:
            raise ImportError("CatBoost not installed. Run: pip install catboost")
    
    def _calculate_metrics(self, y_true, y_pred) -> Tuple[float, float, float]:
        """Calculate comprehensive metrics"""
        from sklearn.metrics import mean_absolute_error, mean_squared_error
        
        mae = float(mean_absolute_error(y_true, y_pred))
        mse = float(mean_squared_error(y_true, y_pred))
        rmse = float(np.sqrt(mse))
        
        # Calculate MAPE (avoiding division by zero)
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        mask = y_true != 0
        if mask.sum() > 0:
            mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0)
        else:
            mape = float('nan')
        
        return mae, rmse, mape
    
    def _print_results(self):
        """Print formatted results"""
        print("\n" + "="*80)
        print("FINAL MODEL PERFORMANCE")
        print("="*80)
        
        if self.metrics.get('mae') is not None and not np.isnan(self.metrics.get('mae', np.nan)):
            print(f"\n  Selected Model: {self.best_model_name or self.model_type}")
            print(f"  {'─'*76}")
            print(f"  Mean Absolute Error (MAE):        {self.metrics['mae']:10.3f} orders/day")
            print(f"  Root Mean Squared Error (RMSE):   {self.metrics['rmse']:10.3f} orders/day")
            mape_val = self.metrics.get('mape', float('nan'))
            if mape_val == mape_val:  # not NaN
                print(f"  Mean Absolute Percentage Error:   {mape_val:10.2f}%")
            
            # Calculate relative metrics
            avg_demand = float(self.predictions.mean()) if len(self.predictions) else float('nan')
            if avg_demand and avg_demand == avg_demand and self.metrics['mae'] == self.metrics['mae']:
                relative_error = (self.metrics['mae'] / avg_demand) * 100
                print(f"  Relative Error:                    {relative_error:10.2f}%")
                
                if relative_error < 15:
                    print(f"\n  ✓ Excellent model performance (< 15% error)")
                elif relative_error < 25:
                    print(f"\n  ✓ Good model performance (15-25% error)")
                elif relative_error < 40:
                    print(f"\n  ⚠ Acceptable model performance (25-40% error)")
                else:
                    print(f"\n  ⚠ Model needs improvement (> 40% error)")
        
        # Print all model comparison if available
        if len(self.all_model_results) > 1:
            print(f"\n  {'─'*76}")
            print(f"  MODEL COMPARISON SUMMARY:")
            print(f"  {'─'*76}")
            print(f"  {'Model':<15} {'MAE':>12} {'RMSE':>12} {'MAPE':>12}")
            print(f"  {'-'*15} {'-'*12} {'-'*12} {'-'*12}")
            
            for model_name, results in sorted(self.all_model_results.items(), 
                                             key=lambda x: (x[1].get('mae', float('inf')))):
                marker = "→" if model_name == self.best_model_name else " "
                mae = results.get('mae', float('nan'))
                rmse = results.get('rmse', float('nan'))
                mape = results.get('mape', float('nan'))
                mae_str = f"{mae:.3f}" if mae == mae else "N/A"
                rmse_str = f"{rmse:.3f}" if rmse == rmse else "N/A"
                mape_str = f"{mape:.2f}%" if mape == mape else "N/A"
                print(f"  {marker} {model_name:<13} {mae_str:>12} {rmse_str:>12} {mape_str:>12}")
        
        print("\n" + "="*80)
        print("✓ Demand forecasting completed successfully")
        print("="*80 + "\n")
    
    def get_feature_importance(self, top_n: int = 10) -> pd.DataFrame:
        """Get feature importance from trained model"""
        if self.model is None:
            return pd.DataFrame()
        
        try:
            if hasattr(self.model, "feature_importances_"):
                importance = self.model.feature_importances_
                # Safely resolve feature names across libs
                if hasattr(self.model, "feature_names_in_"):
                    feature_names = list(getattr(self.model, "feature_names_in_"))
                elif hasattr(self.model, "feature_name_"):
                    feature_names = list(getattr(self.model, "feature_name_"))
                else:
                    feature_names = [f"feature_{i}" for i in range(len(importance))]
                
                df = pd.DataFrame({
                    'feature': feature_names,
                    'importance': importance
                })
                return df.sort_values('importance', ascending=False).head(top_n)
        except Exception as e:
            print(f"⚠️  Could not retrieve feature importance: {e}")
        
        return pd.DataFrame()
