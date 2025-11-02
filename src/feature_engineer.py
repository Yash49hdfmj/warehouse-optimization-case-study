
import pandas as pd
import numpy as np


class FeatureEngineer:
    """Creates features for demand forecasting"""
    
    def __init__(self):
        self.daily_orders = pd.DataFrame()
        self.features = pd.DataFrame()
    
    def build_features(self, orders: pd.DataFrame) -> pd.DataFrame:
        """
        Build complete feature set from orders
        
        Args:
            orders: Orders dataframe
            
        Returns:
            Dataframe with features
        """
        print("\n" + "="*60)
        print("STEP 2: FEATURE ENGINEERING")
        print("="*60)
        
        try:
            # Step 1: Aggregate to daily level
            print("\n  → Aggregating orders to daily level...")
            self.daily_orders = self._build_daily_orders(orders)
            print(f"  ✓ Created {len(self.daily_orders):,} daily observations")
            
            # Step 2: Add time features
            print("\n  → Adding time-based features...")
            self.features = self._add_time_features(self.daily_orders)
            print(f"  ✓ Added time features (dow, month, week, cyclical)")
            
            print("\n  ✓ Feature engineering completed successfully")
            return self.features
            
        except Exception as e:
            print(f"\n  ✗ Error in feature engineering: {str(e)}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame()
    
    def _build_daily_orders(self, orders: pd.DataFrame) -> pd.DataFrame:
        """Aggregate orders to daily level by warehouse and SKU"""
        required_cols = {"order_date", "warehouse", "sku"}
        
        if not required_cols.issubset(orders.columns):
            missing = required_cols - set(orders.columns)
            print(f"  ✗ Missing columns: {missing}")
            return pd.DataFrame(columns=["date", "warehouse", "sku", "orders"])
        
        # Remove any rows with null values in key columns
        orders_clean = orders.dropna(subset=["order_date", "warehouse", "sku"])
        
        # Group by date, warehouse, and sku
        daily = orders_clean.groupby([
            orders_clean["order_date"].dt.date,
            "warehouse",
            "sku"
        ]).size().reset_index(name="orders")
        
        daily.rename(columns={"order_date": "date"}, inplace=True)
        daily["date"] = pd.to_datetime(daily["date"])
        
        return daily
    
    def _add_time_features(self, df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
        """Add time-based features for modeling"""
        if df.empty or date_col not in df.columns:
            return df
        
        result = df.copy()
        
        # Basic time features
        result["dow"] = result[date_col].dt.dayofweek  # 0=Monday, 6=Sunday
        result["month"] = result[date_col].dt.month
        result["week"] = result[date_col].dt.isocalendar().week.astype(int)
        result["day"] = result[date_col].dt.day
        
        # Cyclical features for better model performance
        result["dow_sin"] = np.sin(2 * np.pi * result["dow"] / 7)
        result["dow_cos"] = np.cos(2 * np.pi * result["dow"] / 7)
        result["month_sin"] = np.sin(2 * np.pi * result["month"] / 12)
        result["month_cos"] = np.cos(2 * np.pi * result["month"] / 12)
        
        return result