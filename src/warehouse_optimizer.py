import pandas as pd
import numpy as np
from typing import Dict, Any, Optional

from config import OptimizerConfig, CITY_LOCATIONS
from data_loader import DataLoader
from demand_forecaster import DemandForecaster
from feature_engineer import FeatureEngineer
from optimizer import OptimizationEngine


class WarehouseOptimizer:
    """Main orchestrator for warehouse optimization system"""

    def __init__(self, data_dir: str, config: Optional[OptimizerConfig] = None):
        """
        Initialize warehouse optimizer

        Args:
            data_dir: Directory containing CSV data files
            config: OptimizerConfig instance (uses defaults if None)
        """
        self.data_dir = data_dir

        # Use passed config or create default OptimizerConfig
        if config is None:
            self.config = OptimizerConfig()
            self.config.validate()
        else:
            self.config = config

        # Pipeline components (initialized in _initialize_components)
        self.data_loader: Optional[DataLoader] = None
        self.feature_engineer: Optional[FeatureEngineer] = None
        self.forecaster: Optional[DemandForecaster] = None
        self.optimizer: Optional[OptimizationEngine] = None

        # Results storage
        self.daily_features: pd.DataFrame = pd.DataFrame()
        self.demand_predictions: pd.Series = pd.Series(dtype=float)
        self.rebalancing_plan: pd.DataFrame = pd.DataFrame()
        self.metrics: Dict[str, Any] = {}
        self.optimization_status: Optional[str] = None

    def run(self) -> Dict[str, Any]:
        """
        Execute complete optimization pipeline

        Returns:
            Dictionary with results and status
        """
        print("\n" + "="*70)
        print(" "*15 + "WAREHOUSE OPTIMIZATION SYSTEM")
        print(" "*20 + "MLOps Pipeline Execution")
        print("="*70)

        try:
            # Initialize components
            self._initialize_components()

            # Narrow optionals for type checker (and crash early if missing)
            assert self.data_loader is not None, "DataLoader not initialized"
            assert self.feature_engineer is not None, "FeatureEngineer not initialized"
            assert self.forecaster is not None, "DemandForecaster not initialized"
            assert self.optimizer is not None, "OptimizationEngine not initialized"

            # Step 1: Load Data
            if not self.data_loader.load_all():
                return {"status": "error", "message": "Data loading failed"}

            # Step 2: Feature Engineering
            self.daily_features = self.feature_engineer.build_features(
                self.data_loader.orders
            )

            if self.daily_features.empty:
                return {"status": "error", "message": "Feature engineering failed"}

            # Step 3: Demand Forecasting
            self.demand_predictions, self.metrics = self.forecaster.train_and_predict(
                self.daily_features,
                train_ratio=self.config.train_test_split
            )

            if not self.demand_predictions.empty:
                self.daily_features['predicted_orders'] = self.demand_predictions

            # Step 4: Optimization
            problem, self.optimization_status = self.optimizer.optimize(
                self.data_loader.inventory,
                self.daily_features
            )

            # Step 5: Extract Results
            print("\n" + "="*60)
            print("STEP 5: EXTRACTING RESULTS")
            print("="*60)

            self.rebalancing_plan = self.optimizer.extract_plan(problem)

            if not self.rebalancing_plan.empty:
                # Enrich plan with additional details
                self.rebalancing_plan = self._enrich_plan(self.rebalancing_plan)
                print(f"\n  ✓ Generated {len(self.rebalancing_plan)} transfer recommendations")
                print(f"  ✓ Total units to transfer: {self.rebalancing_plan['quantity'].sum():,.0f}")
            else:
                print("\n  ✓ No transfers needed - inventory is optimal")

            # Prepare results
            results = {
                'status': 'success',
                'optimization_status': self.optimization_status,
                'metrics': self.metrics,
                'rebalancing_plan': self.rebalancing_plan,
                'summary': self.data_loader.get_summary(),
                'daily_features': self.daily_features
            }

            print("\n" + "="*70)
            print(" "*20 + "✓ PIPELINE COMPLETED SUCCESSFULLY")
            print("="*70)

            return results

        except Exception as e:
            print(f"\n✗ Pipeline failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return {"status": "error", "message": str(e)}

    def _initialize_components(self) -> None:
        """Initialize all pipeline components"""
        # Create concrete instances so members are not None at runtime
        self.data_loader = DataLoader(self.data_dir)
        self.feature_engineer = FeatureEngineer()
        # You can switch this to "Auto" or keep "LightGBM" based on your needs
        self.forecaster = DemandForecaster(model_type="LightGBM")
        self.optimizer = OptimizationEngine(self.config, CITY_LOCATIONS)

    def _enrich_plan(self, plan: pd.DataFrame) -> pd.DataFrame:
        """Add distance and cost estimates to rebalancing plan"""
        if plan.empty:
            return plan

        # Ensure optimizer exists for type checker
        assert self.optimizer is not None, "OptimizationEngine not initialized"
        opt = self.optimizer

        enriched = plan.copy()

        # Calculate distances
        enriched['distance_km'] = enriched.apply(
            lambda row: opt.calculate_distance(
                row['from_warehouse'],
                row['to_warehouse']
            ),
            axis=1
        )

        # Calculate costs (assuming cost per ton-km, quantity in kg)
        enriched['estimated_cost'] = (
            enriched['distance_km'] *
            self.config.cost_per_km *
            (enriched['quantity'] / 1000.0)
        )

        # Calculate CO2 emissions (co2_per_ton_km in grams; convert to kg)
        enriched['estimated_co2_kg'] = (
            enriched['distance_km'] *
            (self.config.co2_per_ton_km / 1000.0) *
            (enriched['quantity'] / 1000.0)
        )

        return enriched

    def export_results(self, output_dir: str = "./optimization_results") -> None:
        """Export all results to CSV files"""
        import os
        import json

        os.makedirs(output_dir, exist_ok=True)

        print(f"\n📁 Exporting results to {output_dir}/")

        # Export rebalancing plan
        if not self.rebalancing_plan.empty:
            filepath = os.path.join(output_dir, "rebalancing_plan.csv")
            self.rebalancing_plan.to_csv(filepath, index=False)
            print(f"  ✓ rebalancing_plan.csv ({len(self.rebalancing_plan)} rows)")

        # Export demand forecast
        if not self.daily_features.empty:
            filepath = os.path.join(output_dir, "demand_forecast.csv")
            self.daily_features.to_csv(filepath, index=False)
            print(f"  ✓ demand_forecast.csv ({len(self.daily_features)} rows)")

        # Export inventory analysis
        inv_analysis = self.get_inventory_analysis()
        if not inv_analysis.empty:
            filepath = os.path.join(output_dir, "inventory_analysis.csv")
            inv_analysis.to_csv(filepath, index=False)
            print(f"  ✓ inventory_analysis.csv ({len(inv_analysis)} rows)")

        # Export metrics
        if self.metrics:
            filepath = os.path.join(output_dir, "metrics.json")
            with open(filepath, 'w') as f:
                json.dump(self.metrics, f, indent=2)
            print(f"  ✓ metrics.json")

        print(f"\n✓ All results exported successfully\n")

    def get_inventory_analysis(self) -> pd.DataFrame:
        """Analyze current inventory vs predicted demand"""
        # Ensure loader exists and has inventory
        if self.data_loader is None or self.data_loader.inventory.empty or self.daily_features.empty:
            return pd.DataFrame()

        # Calculate average daily demand
        demand_cols = ['warehouse', 'sku', 'orders']
        if 'predicted_orders' in self.daily_features.columns:
            demand_cols.append('predicted_orders')

        demand_avg = self.daily_features[demand_cols].groupby(['warehouse', 'sku']).mean().reset_index()
        demand_avg.columns = ['warehouse', 'sku', 'avg_daily_orders'] + (
            ['avg_predicted_orders'] if 'predicted_orders' in self.daily_features.columns else []
        )

        # Merge with inventory
        inv = self.data_loader.inventory[['warehouse', 'sku', 'on_hand']].copy()
        inv_analysis = inv.merge(demand_avg, on=['warehouse', 'sku'], how='left')
        inv_analysis['avg_daily_orders'] = inv_analysis['avg_daily_orders'].fillna(0)

        # Calculate days of supply
        inv_analysis['days_of_supply'] = np.where(
            inv_analysis['avg_daily_orders'] > 0,
            inv_analysis['on_hand'] / inv_analysis['avg_daily_orders'],
            999
        )

        # Calculate target inventory
        if 'avg_predicted_orders' in inv_analysis.columns:
            inv_analysis['target_inventory'] = (
                inv_analysis['avg_predicted_orders'] *
                self.config.target_days *
                (1 + (self.config.service_level - 0.9))
            )

            inv_analysis['shortage_surplus'] = (
                inv_analysis['on_hand'] - inv_analysis['target_inventory']
            )

        # Categorize inventory status
        inv_analysis['status'] = pd.cut(
            inv_analysis['days_of_supply'],
            bins=[0, 3, 7, 14, 999],
            labels=['Critical', 'Low', 'Adequate', 'Excess']
        )

        return inv_analysis
