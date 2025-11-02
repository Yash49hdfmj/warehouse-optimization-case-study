"""
WAREHOUSE OPTIMIZATION SYSTEM - MAIN EXECUTION
===============================================
Production-ready MLOps pipeline with multi-model forecasting

Author: AI Assistant  
Date: November 2025
"""

import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
import json

from visualizer import Visualizer
from warehouse_optimizer import WarehouseOptimizer
from config import OptimizerConfig


def print_banner():
    """Print system banner"""
    print("""
    ╔═══════════════════════════════════════════════════════════════════════════╗
    ║                                                                           ║
    ║            WAREHOUSE OPTIMIZATION SYSTEM - MLOps Pipeline                ║
    ║                                                                           ║
    ║  🎯 Problem: Optimal inventory placement and rebalancing across          ║
    ║             warehouses using AI-powered demand forecasting                ║
    ║                                                                           ║
    ║  🤖 ML Models: LightGBM | XGBoost | CatBoost (Auto-selection)           ║
    ║                                                                           ║
    ╚═══════════════════════════════════════════════════════════════════════════╝
    
    Features:
    ✓ Automated data loading and validation
    ✓ Advanced feature engineering with lag & rolling features
    ✓ Multi-model ML comparison (LightGBM, XGBoost, CatBoost)
    ✓ Automatic best model selection
    ✓ Linear programming optimization
    ✓ Cost and CO2 emission analysis
    ✓ Comprehensive reporting and export
    
    Requirements:
    pip install pandas numpy scikit-learn lightgbm xgboost catboost pulp
    """)


def check_dependencies():
    """Check if all required packages are installed"""
    required_packages = {
        'pandas': 'pandas',
        'numpy': 'numpy',
        'sklearn': 'scikit-learn',
        'lightgbm': 'lightgbm',
        'xgboost': 'xgboost',
        'catboost': 'catboost',
        'pulp': 'pulp'
    }
    
    missing = []
    for module, package in required_packages.items():
        try:
            __import__(module)
        except ImportError:
            missing.append(package)
    
    if missing:
        print(f"\n❌ ERROR: Missing required packages: {', '.join(missing)}")
        print(f"\nInstall them using:")
        print(f"pip install {' '.join(missing)}\n")
        return False
    
    return True


def validate_data_files(data_directory):
    """Check if required data files exist"""
    required_files = [
        'cleaned_orders.csv',
        'cleaned_warehouse_inventory.csv'
    ]
    
    optional_files = [
        'cleaned_routes.csv',
        'cleaned_delivery_performance.csv',
        'cleaned_vehicles.csv',
        'cleaned_cost_breakdown.csv',
        'cleaned_customer_feedback.csv'
    ]
    
    missing_required = []
    missing_optional = []
    
    for filename in required_files:
        filepath = os.path.join(data_directory, filename)
        if not os.path.exists(filepath):
            missing_required.append(filename)
    
    for filename in optional_files:
        filepath = os.path.join(data_directory, filename)
        if not os.path.exists(filepath):
            missing_optional.append(filename)
    
    if missing_required:
        print(f"\n❌ ERROR: Missing required files in '{data_directory}':")
        for f in missing_required:
            print(f"   - {f}")
        return False
    
    if missing_optional:
        print(f"\n⚠️  Optional files not found (system will continue):")
        for f in missing_optional:
            print(f"   - {f}")
    
    return True


def _save_metrics_json(metrics: dict, out_dir: str | Path):
    """Persist metrics to metrics.json for downstream consumption."""
    try:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print(f"\n  ✓ Metrics saved to {out / 'metrics.json'}")
    except Exception as _e:
        print(f"\n  ⚠ Could not save metrics.json: {_e}")


def main():
    """Main execution function"""
    
    start_time = datetime.now()
    
    # Print banner
    print_banner()
    
    # Check dependencies
    print("🔍 Checking dependencies...")
    if not check_dependencies():
        return None
    print("✓ All dependencies installed\n")
    
    # Configuration
    print("⚙️  Initializing configuration...")
    config = OptimizerConfig(
        target_days=14,           # Target inventory: 14 days
        service_level=0.95,       # 95% service level
        cost_per_km=15.0,         # ₹15 per km transportation
        co2_per_ton_km=62.0,      # 62g CO2 per ton-km
        optimize_weight='Balanced',  # Balance cost and CO2
        train_test_split=0.8      # 80-20 train-test split
    )
    config.validate()
    print("✓ Configuration validated\n")
    
    # Set data directory
    data_directory = "."  # Current directory
    
    # Validate data files
    print("📂 Validating data files...")
    if not validate_data_files(data_directory):
        print(f"\nPlease ensure all required CSV files are in '{data_directory}'\n")
        return None
    
    print(f"✓ Data files validated")
    print(f"📂 Data Directory: {os.path.abspath(data_directory)}\n")
    
    # Initialize and run optimizer
    print("="*80)
    print("STARTING OPTIMIZATION PIPELINE")
    print("="*80 + "\n")
    
    try:
        optimizer = WarehouseOptimizer(data_directory, config)
        results = optimizer.run()
        
        # Calculate execution time
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Check results
        if results['status'] == 'success':
            # Print summary report
            Visualizer.print_summary(results)
            
            # Print execution summary
            print("\n" + "="*80)
            print("EXECUTION SUMMARY")
            print("="*80)
            print(f"\n  Execution Time:        {duration:.1f} seconds")
            print(f"  Status:                ✓ SUCCESS")
            print(f"  Optimization Result:   {results['optimization_status']}")

            # If model metrics exist, show them here (works for Auto and single-model modes)
            if optimizer.metrics and optimizer.metrics.get('mae') is not None:
                try:
                    mae = optimizer.metrics.get('mae')
                    rmse = optimizer.metrics.get('rmse')
                    mape = optimizer.metrics.get('mape')
                    print(f"\n  Model Metrics:")
                    if mae is not None:
                        print(f"    MAE : {mae:.3f}")
                    if rmse is not None:
                        print(f"    RMSE: {rmse:.3f}")
                    if mape is not None:
                        print(f"    MAPE: {mape:.2f}%")
                except Exception:
                    pass
            
            # Print model comparison if available
            # Print model comparison if available (safe checks)
            forecaster = getattr(optimizer, "forecaster", None)
            if forecaster is not None:
                all_results = getattr(forecaster, "all_model_results", None)
                best_name  = getattr(forecaster, "best_model_name", None)

                if isinstance(all_results, dict) and len(all_results) > 1:
                    print(f"\n  ML Model Comparison:")

                    # sort by MAE; if missing/None, push to the end
                    def _mae_key(item):
                        val = item[1].get("mae")
                        try:
                            return float(val)
                        except Exception:
                            return float("inf")

                    for model_name, model_results in sorted(all_results.items(), key=_mae_key):
                        marker = "🏆" if (best_name and model_name == best_name) else "  "
                        mae = model_results.get("mae")
                        rmse = model_results.get("rmse")
                        mae_str  = f"{mae:.3f}"  if isinstance(mae,  (int, float, np.floating)) else "N/A"
                        rmse_str = f"{rmse:.3f}" if isinstance(rmse, (int, float, np.floating)) else "N/A"

                        print(f"    {marker} {model_name:12s}: MAE={mae_str:>7}, RMSE={rmse_str:>7}")

                    if best_name:
                        print(f"\n  Selected Best Model:   {best_name}")


            # Persist metrics.json for downstream usage (e.g., tests, dashboards)
            if optimizer.metrics:
                _save_metrics_json(optimizer.metrics, "./optimization_results")
            
            # Additional analysis
            print("\n" + "="*80)
            print("WAREHOUSE-LEVEL INVENTORY ANALYSIS")
            print("="*80)
            
            inv_analysis = optimizer.get_inventory_analysis()
            if not inv_analysis.empty:
                warehouse_summary = inv_analysis.groupby('warehouse').agg({
                    'on_hand': 'sum',
                    'avg_daily_orders': 'mean',
                    'days_of_supply': 'mean'
                }).reset_index()
                
                warehouse_summary['days_of_supply'] = warehouse_summary['days_of_supply'].replace([np.inf], 999)
                warehouse_summary.columns = ['Warehouse', 'Current Stock', 'Avg Daily Orders', 'Days of Supply']
                
                print("\n" + warehouse_summary.to_string(index=False))
                
                # Identify critical warehouses
                critical = warehouse_summary[warehouse_summary['Days of Supply'] < config.target_days * 0.5]
                excess = warehouse_summary[warehouse_summary['Days of Supply'] > config.target_days * 2]
                
                if not critical.empty:
                    print(f"\n  ⚠️  Warehouses with LOW inventory:")
                    for _, row in critical.iterrows():
                        print(f"      {row['Warehouse']:12s}: {row['Days of Supply']:.1f} days (Target: {config.target_days})")
                
                if not excess.empty:
                    print(f"\n  ⚠️  Warehouses with EXCESS inventory:")
                    for _, row in excess.iterrows():
                        print(f"      {row['Warehouse']:12s}: {row['Days of Supply']:.1f} days (Target: {config.target_days})")
            
            # Performance insights
            print("\n" + "="*80)
            print("KEY INSIGHTS & RECOMMENDATIONS")
            print("="*80)
            
            insights = []
            
            # Model accuracy insight
            if optimizer.metrics.get('mae'):
                avg_demand = optimizer.daily_features['orders'].mean()
                relative_error = (optimizer.metrics['mae'] / avg_demand) * 100 if avg_demand > 0 else 0
                
                if relative_error < 15:
                    insights.append(f"✓ EXCELLENT forecast accuracy ({relative_error:.1f}% error) - High confidence in predictions")
                elif relative_error < 25:
                    insights.append(f"✓ GOOD forecast accuracy ({relative_error:.1f}% error) - Reliable for planning")
                elif relative_error < 40:
                    insights.append(f"⚠ ACCEPTABLE forecast accuracy ({relative_error:.1f}% error) - Monitor closely")
                else:
                    insights.append(f"⚠ Model needs improvement ({relative_error:.1f}% error) - Consider more training data")
            
            # Inventory balance insight
            if not inv_analysis.empty and 'days_of_supply' in inv_analysis.columns:
                avg_dos = inv_analysis['days_of_supply'].replace([np.inf], 999).mean()
                target_dos = config.target_days
                
                if avg_dos > target_dos * 2:
                    insights.append(f"⚠ EXCESS inventory detected: Avg {avg_dos:.0f} days vs target {target_dos} days")
                    insights.append(f"   → Recommendation: Reduce order quantities or increase distribution")
                elif avg_dos < target_dos * 0.5:
                    insights.append(f"⚠ LOW inventory levels: Avg {avg_dos:.0f} days vs target {target_dos} days")
                    insights.append(f"   → Recommendation: Increase safety stock or expedite replenishment")
                else:
                    insights.append(f"✓ OPTIMAL inventory levels: Avg {avg_dos:.0f} days near target {target_dos} days")
            
            # Rebalancing impact
            plan = optimizer.rebalancing_plan
            if not plan.empty:
                total_units = plan['quantity'].sum()
                insights.append(f"📦 {len(plan)} transfer recommendations for {total_units:,.0f} units")
                
                if 'estimated_cost' in plan.columns:
                    total_cost = plan['estimated_cost'].sum()
                    insights.append(f"💰 Estimated rebalancing cost: ₹{total_cost:,.2f}")
                
                insights.append(f"   → Priority: Execute high-priority transfers within 3-5 days")
            else:
                # Clarify that no redistribution helps even if inventory is excessive
                insights.append(f"✓ No redistribution needed between warehouses based on current targets")
                insights.append(f"   → If excess persists, reduce procurement or return excess to supplier (policy permitting)")
            
            # Print insights
            for i, insight in enumerate(insights, 1):
                print(f"\n  {i}. {insight}")
            
            # Action items
            print("\n" + "="*80)
            print("RECOMMENDED ACTION ITEMS")
            print("="*80)
            
            actions = [
                "1. Review the rebalancing plan in optimization_results/rebalancing_plan.csv",
                "2. Prioritize transfers for warehouses with < 7 days of supply",
                "3. Monitor forecast accuracy weekly and retrain model if performance degrades",
                "4. Update safety stock levels based on demand volatility",
                "5. Schedule next optimization run in 1 week or after major demand changes"
            ]
            
            for action in actions:
                print(f"  {action}")
            
            print("\n" + "="*80)
            print("✅ OPTIMIZATION COMPLETED SUCCESSFULLY")
            print("="*80)
            print(f"\n  📁 Results saved to: ./optimization_results/")
            print(f"  ⏱️  Total execution time: {duration:.1f} seconds")
            print(f"  🎯 Ready for deployment!")
            print("\n")
            
            return optimizer
        
        else:
            print("\n" + "="*80)
            print("❌ OPTIMIZATION FAILED")
            print("="*80)
            print(f"\n  Error: {results.get('message', 'Unknown error')}")
            print(f"  Duration: {duration:.1f} seconds")
            
            print("\n  Troubleshooting checklist:")
            print("    1. Verify all CSV files have correct column names")
            print("    2. Check data quality (no missing values in key columns)")
            print("    3. Ensure sufficient historical data (minimum 30 days)")
            print("    4. Review error messages above for specific issues")
            print("\n")
            return None
    
    except Exception as e:
        print("\n" + "="*80)
        print("❌ CRITICAL ERROR")
        print("="*80)
        print(f"\n  {str(e)}\n")
        import traceback
        traceback.print_exc()
        print("\n")
        return None


if __name__ == "__main__":
    # Run the system
    optimizer = main()
    
    # Additional usage examples if successful
    if optimizer and optimizer.optimization_status == 'Optimal':
        print("\n" + "="*80)
        print("📖 PROGRAMMATIC ACCESS EXAMPLES")
        print("="*80)
        
        print("\n  # Access optimization results:")
        print("  >>> rebalancing_plan = optimizer.rebalancing_plan")
        print("  >>> metrics = optimizer.metrics")
        print("  >>> forecast = optimizer.daily_features")
        
        print("\n  # Get inventory analysis:")
        print("  >>> inv_analysis = optimizer.get_inventory_analysis()")
        print("  >>> critical_items = inv_analysis[inv_analysis['status'] == 'Critical']")
        
        print("\n  # Filter by warehouse:")
        print("  >>> mumbai_data = inv_analysis[inv_analysis['warehouse'] == 'Mumbai']")
        
        print("\n  # Analyze specific product:")
        print("  >>> electronics = inv_analysis[inv_analysis['sku'] == 'Electronics']")
        
        print("\n  # Export custom reports:")
        print("  >>> optimizer.rebalancing_plan.to_excel('custom_report.xlsx')")
        
        print("\n" + "="*80 + "\n")
        
        print("🎉 All done! Good luck with your OFI internship application! 🚀\n")
