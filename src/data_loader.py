
import pandas as pd
import numpy as np
from typing import Dict, Any
import os
import warnings
warnings.filterwarnings('ignore')


class DataLoader:
    """Handles loading and preprocessing of warehouse data"""
    
    # Column mapping standards for data normalization
    COLUMN_MAPPINGS = {
        "orders": {
            "order_date": ["order_date", "order_dt", "date"],
            "warehouse": ["destination", "dest", "warehouse"],
            "sku": ["sku", "product_category", "category"]
        },
        "inventory": {
            "warehouse": ["warehouse", "location", "site"],
            "sku": ["sku", "product_category", "category"],
            "on_hand": ["on_hand", "current_stock_units", "stock"]
        }
    }
    
    def __init__(self, data_dir: str):
        """
        Initialize data loader
        
        Args:
            data_dir: Directory containing CSV files
        """
        self.data_dir = data_dir
        self.orders = pd.DataFrame()
        self.inventory = pd.DataFrame()
        self.routes = pd.DataFrame()
        self.delivery = pd.DataFrame()
        self.vehicles = pd.DataFrame()
        self.cost = pd.DataFrame()
        self.feedback = pd.DataFrame()
        self._loaded = False
    
    def load_all(self) -> bool:
        """
        Load all datasets from directory
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            print("\n" + "="*60)
            print("STEP 1: DATA LOADING")
            print("="*60)
            
            # Define file paths
            files = {
                'orders': 'cleaned_orders.csv',
                'inventory': 'cleaned_warehouse_inventory.csv',
                'routes': 'cleaned_routes.csv',
                'delivery': 'cleaned_delivery_performance.csv',
                'vehicles': 'cleaned_vehicles.csv',
                'cost': 'cleaned_cost_breakdown.csv',
                'feedback': 'cleaned_customer_feedback.csv'
            }
            
            # Load each file
            for name, filename in files.items():
                filepath = os.path.join(self.data_dir, filename)
                if os.path.exists(filepath):
                    df = pd.read_csv(filepath)
                    df = self._clean_dataframe(df)
                    setattr(self, name, df)
                    print(f"  ✓ Loaded {name:12s}: {len(df):5d} rows, {len(df.columns):2d} columns")
                else:
                    print(f"  ⚠ Warning: {filename} not found")
            
            # Apply transformations
            if not self.orders.empty and not self.inventory.empty:
                self._apply_transformations()
                self._loaded = True
                print("\n  ✓ Data loading completed successfully")
                return True
            else:
                print("\n  ✗ Critical files missing (orders or inventory)")
                return False
            
        except Exception as e:
            print(f"\n  ✗ Error loading data: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    def _clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean column names and basic formatting"""
        if df is None or df.empty:
            return pd.DataFrame()
        
        # Standardize column names
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
        return df
    
    def _apply_transformations(self):
        """Apply domain-specific transformations to loaded data"""
        print("\n  → Applying data transformations...")
        
        # Transform orders
        if not self.orders.empty:
            # Map columns
            if "product_category" in self.orders.columns and "sku" not in self.orders.columns:
                self.orders["sku"] = self.orders["product_category"]
            
            # Convert date column
            if "order_date" in self.orders.columns:
                self.orders["order_date"] = pd.to_datetime(
                    self.orders["order_date"], errors="coerce"
                )
            
            # Extract warehouse from destination
            if "destination" in self.orders.columns:
                if "warehouse" not in self.orders.columns:
                    self.orders["warehouse"] = self.orders["destination"]
                self.orders["warehouse"] = self.orders["warehouse"].astype(str).str.strip().str.title()
            
            # Standardize SKU
            if "sku" in self.orders.columns:
                self.orders["sku"] = self.orders["sku"].astype(str).str.strip()
        
        # Transform inventory
        if not self.inventory.empty:
            # Map columns
            if "product_category" in self.inventory.columns and "sku" not in self.inventory.columns:
                self.inventory["sku"] = self.inventory["product_category"]
            
            if "location" in self.inventory.columns and "warehouse" not in self.inventory.columns:
                self.inventory["warehouse"] = self.inventory["location"]
            
            if "current_stock_units" in self.inventory.columns and "on_hand" not in self.inventory.columns:
                self.inventory["on_hand"] = self.inventory["current_stock_units"]
            
            # Ensure numeric on_hand
            if "on_hand" in self.inventory.columns:
                self.inventory["on_hand"] = pd.to_numeric(
                    self.inventory["on_hand"], errors="coerce"
                ).fillna(0)
            
            # Standardize warehouse names
            if "warehouse" in self.inventory.columns:
                self.inventory["warehouse"] = self.inventory["warehouse"].astype(str).str.strip().str.title()
            
            # Standardize SKU
            if "sku" in self.inventory.columns:
                self.inventory["sku"] = self.inventory["sku"].astype(str).str.strip()
        
        print("  ✓ Transformations completed")
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics of loaded data"""
        if not self._loaded:
            return {"error": "Data not loaded"}
        
        summary = {
            "orders": {
                "total": len(self.orders),
                "date_range": (
                    str(self.orders["order_date"].min()) if "order_date" in self.orders.columns else None,
                    str(self.orders["order_date"].max()) if "order_date" in self.orders.columns else None
                ),
                "warehouses": self.orders["warehouse"].nunique() if "warehouse" in self.orders.columns else 0,
                "unique_skus": self.orders["sku"].nunique() if "sku" in self.orders.columns else 0
            },
            "inventory": {
                "total_units": float(self.inventory["on_hand"].sum()) if "on_hand" in self.inventory.columns else 0,
                "warehouses": self.inventory["warehouse"].nunique() if "warehouse" in self.inventory.columns else 0,
                "skus": self.inventory["sku"].nunique() if "sku" in self.inventory.columns else 0
            }
        }
        return summary
