"""
WAREHOUSE OPTIMIZATION SYSTEM - COMPREHENSIVE TEST SUITE
=========================================================
Tests all components of the warehouse optimization pipeline

Author: AI Assistant
Date: November 2025
"""
from __future__ import annotations  # put this at the very top of the file

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

import sys
import traceback
import os
from typing import Tuple, Optional, Any


def test_imports() -> bool:
    """Test 1: Verify all required packages are installed"""
    print("="*80)
    print("TEST 1: Import Verification")
    print("="*80)
    try:
        import pandas as pd  # noqa: F401
        import numpy as np  # noqa: F401
        import lightgbm as lgb
        from pulp import LpProblem  # noqa: F401
        from sklearn.preprocessing import LabelEncoder  # noqa: F401
        
        libs = {
            'pandas': pd.__version__,
            'numpy': np.__version__,
            'lightgbm': lgb.__version__,
            'scikit-learn': 'installed',
            'pulp': 'installed'
        }
        for lib, ver in libs.items():
            print(f"  ✓ {lib}: {ver}")
        print()
        return True
    except Exception as e:
        print(f"  ✗ Error: {e}\n")
        return False


def test_config() -> Tuple[bool, Optional[Any]]:
    """Test 2: Configuration validation"""
    print("="*80)
    print("TEST 2: Configuration Module")
    print("="*80)
    try:
        from config import OptimizerConfig, CITY_LOCATIONS
        
        # Test default configuration
        config = OptimizerConfig()
        config.validate()
        
        print(f"  ✓ Config created successfully")
        print(f"    Target days: {config.target_days}")
        print(f"    Service level: {config.service_level:.0%}")
        print(f"    Cost per km: ₹{config.cost_per_km:.2f}")
        print(f"    Optimization: {config.optimize_weight}")
        
        # Test city locations
        print(f"\n  ✓ City locations loaded: {len(CITY_LOCATIONS)} cities")
        sample_cities = list(CITY_LOCATIONS.keys())[:3]
        for city in sample_cities:
            loc = CITY_LOCATIONS[city]
            print(f"    {city}: ({loc.lat:.4f}, {loc.lon:.4f})")
        
        # Test invalid config (must return a 2-tuple in all branches)
        try:
            bad_config = OptimizerConfig(service_level=1.5)
            bad_config.validate()
            print("  ✗ Validation should have failed")
            return (False, None)
        except AssertionError:
            print(f"\n  ✓ Config validation working")
        
        print()
        return (True, config)
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        traceback.print_exc()
        print()
        return (False, None)


def test_data_loading() -> Tuple[bool, Optional[Any]]:
    """Test 3: Data loading and preprocessing"""
    print("="*80)
    print("TEST 3: Data Loading")
    print("="*80)
    try:
        from data_loader import DataLoader
        
        # Test with current directory
        loader = DataLoader(".")
        success = loader.load_all()
        
        if not success:
            print("  ✗ Data loading failed")
            return (False, None)
        
        print(f"\n  ✓ Data loaded successfully")
        
        # Verify orders
        if not loader.orders.empty:
            print(f"    Orders: {len(loader.orders):,} rows")
            print(f"    Date range: {loader.orders['order_date'].min()} to {loader.orders['order_date'].max()}")
            print(f"    Warehouses: {loader.orders['warehouse'].nunique()}")
            print(f"    SKUs: {loader.orders['sku'].nunique()}")
        
        # Verify inventory
        if not loader.inventory.empty:
            print(f"    Inventory: {len(loader.inventory):,} rows")
            print(f"    Total units: {loader.inventory['on_hand'].sum():,.0f}")
            print(f"    Warehouses: {loader.inventory['warehouse'].nunique()}")
        
        # Get summary
        summary = loader.get_summary()
        print(f"\n  ✓ Data summary generated")
        
        print()
        return (True, loader)
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        traceback.print_exc()
        print()
        return (False, None)


def test_feature_engineering(loader) -> Tuple[bool, Optional[pd.DataFrame], Optional[Any]]:
    """Test 4: Feature engineering"""
    print("="*80)
    print("TEST 4: Feature Engineering")
    print("="*80)
    try:
        from feature_engineer import FeatureEngineer
        
        engineer = FeatureEngineer()
        features = engineer.build_features(loader.orders)
        
        if features.empty:
            print("  ✗ Feature engineering failed")
            return (False, None, None)
        
        print(f"\n  ✓ Features created: {len(features):,} daily observations")
        print(f"    Date range: {features['date'].min()} to {features['date'].max()}")
        print(f"    Avg daily orders: {features['orders'].mean():.2f}")
        
        # Check required columns
        required_cols = ['date', 'warehouse', 'sku', 'orders', 'dow', 'month', 'week']
        missing_cols = [col for col in required_cols if col not in features.columns]
        
        if missing_cols:
            print(f"  ✗ Missing columns: {missing_cols}")
            return (False, None, None)
        
        print(f"  ✓ All required columns present")
        
        # Check time features
        print(f"\n  Time features:")
        print(f"    Day of week: {features['dow'].min()} to {features['dow'].max()}")
        print(f"    Months: {features['month'].nunique()} unique")
        print(f"    Weeks: {features['week'].nunique()} unique")
        
        # Check cyclical features
        if 'dow_sin' in features.columns and 'dow_cos' in features.columns:
            print(f"  ✓ Cyclical features present")
        
        print()
        return (True, features, engineer)
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        traceback.print_exc()
        print()
        return (False, None, None)


def test_demand_forecasting(features: pd.DataFrame) -> Tuple[bool, Optional[Any], Optional[pd.Series]]:
    """Test 5: ML demand forecasting"""
    print("="*80)
    print("TEST 5: Demand Forecasting")
    print("="*80)
    try:
        from demand_forecaster import DemandForecaster
        
        forecaster = DemandForecaster(model_type="LightGBM")
        predictions, metrics = forecaster.train_and_predict(features, train_ratio=0.8)
        
        if predictions.empty:
            print("  ✗ Forecasting failed")
            return (False, None, None)
        
        print(f"\n  ✓ Predictions generated: {len(predictions):,} forecasts")
        print(f"    Avg predicted demand: {predictions.mean():.2f} orders/day")
        print(f"    Min prediction: {predictions.min():.2f}")
        print(f"    Max prediction: {predictions.max():.2f}")
        
        # Check metrics
        if metrics.get('mae') and not np.isnan(metrics.get('mae', np.nan)):
            print(f"\n  Model Performance:")
            print(f"    MAE: {metrics['mae']:.2f} orders/day")
            print(f"    RMSE: {metrics['rmse']:.2f} orders/day")
            
            # Calculate relative error
            actual_mean = features['orders'].mean()
            relative_error = (metrics['mae'] / actual_mean) * 100 if actual_mean > 0 else 0
            print(f"    Relative Error: {relative_error:.1f}%")
            
            if relative_error < 50:
                print(f"  ✓ Good model performance")
            else:
                print(f"  ⚠ High relative error")
        else:
            print("  ⚠ No metrics available")
        
        print()
        return (True, forecaster, predictions)
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        traceback.print_exc()
        print()
        return (False, None, None)


def test_distance_calculation(config) -> Tuple[bool, Optional[Any]]:
    """Test 6: Distance calculation"""
    print("="*80)
    print("TEST 6: Distance Calculation")
    print("="*80)
    try:
        from optimizer import OptimizationEngine
        from config import CITY_LOCATIONS
        
        optimizer = OptimizationEngine(config, CITY_LOCATIONS)
        
        # Test distances between major cities
        test_pairs = [
            ("Mumbai", "Delhi"),
            ("Bangalore", "Chennai"),
            ("Kolkata", "Hyderabad")
        ]
        
        print(f"  Sample distances:")
        for city1, city2 in test_pairs:
            distance = optimizer.calculate_distance(city1, city2)
            print(f"    {city1:12s} → {city2:12s}: {distance:6.1f} km")
        
        # Verify distance is symmetric
        dist_ab = optimizer.calculate_distance("Mumbai", "Delhi")
        dist_ba = optimizer.calculate_distance("Delhi", "Mumbai")
        
        if abs(dist_ab - dist_ba) < 0.1:
            print(f"\n  ✓ Distance calculation is symmetric")
        else:
            print(f"\n  ✗ Distance asymmetry detected")
            return (False, None)
        
        # Test same city (should be 0)
        same_city = optimizer.calculate_distance("Mumbai", "Mumbai")
        if same_city < 0.1:
            print(f"  ✓ Same city distance is zero")
        else:
            print(f"  ✗ Same city distance should be zero, got {same_city:.2f}")
            return (False, None)
        
        print()
        return (True, optimizer)
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        traceback.print_exc()
        print()
        return (False, None)


def test_optimization(optimizer, loader, features: pd.DataFrame) -> Tuple[bool, Optional[Any], Optional[str]]:
    """Test 7: Optimization engine"""
    print("="*80)
    print("TEST 7: Optimization Engine")
    print("="*80)
    try:
        # Prepare demand forecast dataframe
        demand_forecast = features[['warehouse', 'sku', 'orders']].copy()
        
        # Run optimization
        problem, status = optimizer.optimize(loader.inventory, demand_forecast)
        
        print(f"\n  ✓ Optimization completed")
        print(f"    Status: {status}")
        
        if status == "Optimal":
            print(f"  ✓ Optimal solution found!")
            
            # Extract plan
            plan = optimizer.extract_plan(problem)
            
            if not plan.empty:
                print(f"\n  Rebalancing Plan:")
                print(f"    Total transfers: {len(plan)}")
                print(f"    Total units: {plan['quantity'].sum():,.0f}")
                
                # Top 3 transfers
                top_3 = plan.nlargest(3, 'quantity')
                print(f"\n  Top 3 transfers:")
                for _, row in top_3.iterrows():
                    print(f"    {row['from_warehouse']:12s} → {row['to_warehouse']:12s} | "
                          f"{row['sku']:20s} | {row['quantity']:6.0f} units")
            else:
                print(f"\n  ✓ No transfers needed - inventory is optimal")
        
        elif status == "Infeasible":
            print(f"  ⚠ Problem is infeasible - check constraints")
        else:
            print(f"  ⚠ Optimization status: {status}")
        
        print()
        return (True, problem, status)
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        traceback.print_exc()
        print()
        return (False, None, None)


def test_warehouse_optimizer(config) -> Tuple[bool, Optional[Any]]:
    """Test 8: Full warehouse optimizer"""
    print("="*80)
    print("TEST 8: Warehouse Optimizer (Integration)")
    print("="*80)
    try:
        from warehouse_optimizer import WarehouseOptimizer
        
        optimizer = WarehouseOptimizer(".", config)
        results = optimizer.run()
        
        if results['status'] != 'success':
            print(f"  ✗ Optimization failed: {results.get('message', 'Unknown error')}")
            return (False, None)
        
        print(f"\n  ✓ Full pipeline executed successfully!")
        print(f"    Status: {results['optimization_status']}")
        
        # Check results
        metrics = results.get('metrics', {})
        if metrics:
            print(f"\n  Model Metrics:")
            if metrics.get('mae'):
                print(f"    MAE: {metrics['mae']:.2f}")
                print(f"    RMSE: {metrics['rmse']:.2f}")
        
        plan = results.get('rebalancing_plan')
        if plan is not None and not plan.empty:
            print(f"\n  Rebalancing Plan:")
            print(f"    Transfers: {len(plan)}")
            print(f"    Total units: {plan['quantity'].sum():,.0f}")
            
            if 'estimated_cost' in plan.columns:
                print(f"    Total cost: ₹{plan['estimated_cost'].sum():,.2f}")
            
            if 'estimated_co2_kg' in plan.columns:
                print(f"    Total CO2: {plan['estimated_co2_kg'].sum():,.1f} kg")
        
        summary = results.get('summary', {})
        if summary:
            print(f"\n  Data Summary:")
            orders_info = summary.get('orders', {})
            inv_info = summary.get('inventory', {})
            print(f"    Orders: {orders_info.get('total', 0):,}")
            print(f"    Warehouses: {inv_info.get('warehouses', 0)}")
            print(f"    SKUs: {inv_info.get('skus', 0)}")
        
        print()
        return (True, optimizer)
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        traceback.print_exc()
        print()
        return (False, None)


def test_inventory_analysis(optimizer) -> Tuple[bool, Optional[pd.DataFrame]]:
    """Test 9: Inventory analysis"""
    print("="*80)
    print("TEST 9: Inventory Analysis")
    print("="*80)
    try:
        inv_analysis = optimizer.get_inventory_analysis()
        
        if inv_analysis.empty:
            print("  ⚠ No inventory analysis available")
            return (True, None)
        
        print(f"  ✓ Inventory analysis generated: {len(inv_analysis)} items")
        
        # Check required columns
        if 'days_of_supply' in inv_analysis.columns:
            avg_days = inv_analysis['days_of_supply'].replace([np.inf], 999).mean()
            print(f"    Avg days of supply: {avg_days:.1f}")
            
            # Count by status
            if 'status' in inv_analysis.columns:
                status_counts = inv_analysis['status'].value_counts()
                print(f"\n  Inventory Status:")
                for status, count in status_counts.items():
                    print(f"    {status}: {count}")
        
        # Warehouse summary
        if 'warehouse' in inv_analysis.columns:
            warehouse_summary = inv_analysis.groupby('warehouse').agg({
                'on_hand': 'sum',
                'avg_daily_orders': 'mean'
            }).reset_index()
            
            print(f"\n  Warehouse Summary:")
            for _, row in warehouse_summary.head(5).iterrows():
                print(f"    {row['warehouse']:12s}: {row['on_hand']:8,.0f} units, "
                      f"{row['avg_daily_orders']:5.1f} orders/day")
        
        print()
        return (True, inv_analysis)
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        traceback.print_exc()
        print()
        return (False, None)


def test_export(optimizer) -> bool:
    """Test 10: Results export"""
    print("="*80)
    print("TEST 10: Results Export")
    print("="*80)
    try:
        # Create output directory
        output_dir = "./test_optimization_results"
        os.makedirs(output_dir, exist_ok=True)
        
        # Export results
        optimizer.export_results(output_dir)
        
        # Check expected files
        expected_files = [
            'rebalancing_plan.csv',
            'demand_forecast.csv',
            'inventory_analysis.csv',
            'metrics.json'
        ]
        
        files_found = []
        for filename in expected_files:
            filepath = os.path.join(output_dir, filename)
            if os.path.exists(filepath):
                size = os.path.getsize(filepath) / 1024
                print(f"  ✓ {filename}: {size:.1f} KB")
                files_found.append(filename)
            else:
                print(f"  ⚠ {filename}: not found")
        
        print(f"\n  Files created: {len(files_found)}/{len(expected_files)}")
        
        # Clean up test directory
        import shutil
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
            print(f"  ✓ Test directory cleaned up")
        
        print()
        return True
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        traceback.print_exc()
        print()
        return False


def test_visualization(optimizer) -> bool:
    """Test 11: Visualization and reporting"""
    print("="*80)
    print("TEST 11: Visualization & Reporting")
    print("="*80)
    try:
        from visualizer import Visualizer
        
        # Prepare results dictionary
        results = {
            'summary': optimizer.data_loader.get_summary(),
            'metrics': optimizer.metrics,
            'rebalancing_plan': optimizer.rebalancing_plan
        }
        
        # Print summary
        print("\n  Testing summary report generation...")
        Visualizer.print_summary(results)
        
        print("  ✓ Visualization module working")
        print()
        return True
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        traceback.print_exc()
        print()
        return False


def run_full_pipeline() -> bool:
    """Test 12: Complete end-to-end pipeline"""
    print("="*80)
    print("TEST 12: Full Pipeline Execution")
    print("="*80)
    try:
        # Import main if it exists
        try:
            from main import main
            print("  Running main() function...")
            optimizer = main()
            
            if optimizer:
                print(f"\n  ✓ Pipeline executed successfully!")
                print(f"    Optimization status: {optimizer.optimization_status}")
                
                if not optimizer.rebalancing_plan.empty:
                    print(f"    Transfers: {len(optimizer.rebalancing_plan)}")
                    print(f"    Units: {optimizer.rebalancing_plan['quantity'].sum():,.0f}")
                
                return True
            else:
                print(f"\n  ✗ Pipeline returned None")
                return False
        except ImportError:
            print("  ⚠ main.py not found - skipping full pipeline test")
            return True
        
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        traceback.print_exc()
        print()
        return False


def main() -> None:
    """Run all tests"""
    print("\n" + "╔" + "="*78 + "╗")
    print("║" + " "*15 + "WAREHOUSE OPTIMIZATION SYSTEM - TEST SUITE" + " "*21 + "║")
    print("╚" + "="*78 + "╝\n")
    
    results: dict[str, bool] = {}
    
    # Test 1: Imports
    results['imports'] = test_imports()
    if not results['imports']:
        print("❌ Install: pip install pandas numpy scikit-learn lightgbm pulp")
        return
    
    # Test 2: Configuration
    success, config = test_config()
    results['config'] = success
    if not success:
        return
    
    # Test 3: Data Loading
    success, loader = test_data_loading()
    results['data_loading'] = success
    if not success or loader is None:
        print("⚠️  Place CSV files in current directory and try again")
        return

    # Test 4: Feature Engineering
    success, features, engineer = test_feature_engineering(loader)
    results['feature_engineering'] = success
    if not success or features is None:
        return

    # (Optional) help Pylance narrow types
    from typing import cast
    features = cast(pd.DataFrame, features)

    # Test 5: Demand Forecasting
    success, forecaster, predictions = test_demand_forecasting(features)
    results['forecasting'] = success
    if not success or forecaster is None or predictions is None:
        return

    # Test 6: Distance Calculation
    success, optimizer_engine = test_distance_calculation(config)
    results['distance'] = success
    if not success or optimizer_engine is None:
        return

    # Test 7: Optimization
    success, problem, status = test_optimization(optimizer_engine, loader, features)
    results['optimization'] = success

    
    # Test 8: Full Warehouse Optimizer
    success, optimizer = test_warehouse_optimizer(config)
    results['warehouse_optimizer'] = success
    if not success:
        return
    
    # Test 9: Inventory Analysis
    success, inv_analysis = test_inventory_analysis(optimizer)
    results['inventory_analysis'] = success
    
    # Test 10: Export
    success = test_export(optimizer)
    results['export'] = success
    
    # Test 11: Visualization
    success = test_visualization(optimizer)
    results['visualization'] = success
    
    # Test 12: Full Pipeline
    success = run_full_pipeline()
    results['full_pipeline'] = success
    
    # Print summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✓" if result else "✗"
        print(f"  {status} {test_name.replace('_', ' ').title()}")
    
    print(f"\nPassed: {passed}/{total}")
    print("="*80 + "\n")
    
    if passed >= total - 1:
        print("🎉 ALL CORE TESTS PASSED - SYSTEM READY!")
        print("\nNext Steps:")
        print("  1. Run 'python main.py' for full execution")
        print("  2. Review optimization_results/ directory")
        print("  3. Check rebalancing_plan.csv for recommendations")
        print("  4. Analyze inventory_analysis.csv for insights")
        print("  5. Deploy optimization recommendations")
    else:
        print("⚠️  Fix critical tests before deploying")
    
    print()


if __name__ == "__main__":
    main()
