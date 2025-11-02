
import pandas as pd
import numpy as np


class Visualizer:
    """Create reports and visualizations"""
    
    @staticmethod
    def print_summary(optimizer_results: dict):
        """Print comprehensive summary report"""
        print("\n" + "="*70)
        print(" "*20 + "OPTIMIZATION SUMMARY REPORT")
        print("="*70)
        
        # Data Summary
        print("\n📊 DATA SUMMARY")
        print("-" * 70)
        summary = optimizer_results.get('summary', {})
        orders_info = summary.get('orders', {})
        inv_info = summary.get('inventory', {})
        
        print(f"Total Orders:               {orders_info.get('total', 0):10,d}")
        print(f"Date Range:                 {orders_info.get('date_range', ('N/A', 'N/A'))[0]} to {orders_info.get('date_range', ('N/A', 'N/A'))[1]}")
        print(f"Warehouses:                 {inv_info.get('warehouses', 0):10d}")
        print(f"Unique SKUs:                {inv_info.get('skus', 0):10d}")
        print(f"Total Inventory:            {inv_info.get('total_units', 0):10,.0f} units")
        
        # Model Performance
        print("\n📈 DEMAND FORECAST PERFORMANCE")
        print("-" * 70)
        metrics = optimizer_results.get('metrics', {})
        if metrics.get('mae'):
            print(f"Mean Absolute Error (MAE):  {metrics['mae']:10.2f} orders/day")
            print(f"Root Mean Squared Error:    {metrics['rmse']:10.2f} orders/day")
            print(f"Model Type:                 LightGBM")
        else:
            print("No metrics available")
        
        # Rebalancing Plan
        print("\n🚚 REBALANCING RECOMMENDATIONS")
        print("-" * 70)
        plan = optimizer_results.get('rebalancing_plan')
        
        if plan is not None and not plan.empty:
            total_transfers = len(plan)
            total_units = plan['quantity'].sum()
            
            print(f"Total Transfers:            {total_transfers:10d}")
            print(f"Total Units to Move:        {total_units:10,.0f} units")
            
            if 'distance_km' in plan.columns:
                avg_distance = plan['distance_km'].mean()
                print(f"Average Distance:           {avg_distance:10,.1f} km")
            
            if 'estimated_cost' in plan.columns:
                total_cost = plan['estimated_cost'].sum()
                print(f"Estimated Total Cost:       ₹{total_cost:10,.2f}")
            
            if 'estimated_co2_kg' in plan.columns:
                total_co2 = plan['estimated_co2_kg'].sum()
                print(f"Estimated CO2 Emissions:    {total_co2:10,.1f} kg")
            
            # Top transfers
            print("\n📋 Top 5 Largest Transfers:")
            top_transfers = plan.nlargest(5, 'quantity')
            for _, row in top_transfers.iterrows():
                dist_str = f"{row['distance_km']:6,.0f} km" if 'distance_km' in row else ""
                print(f"  {row['from_warehouse']:12s} → {row['to_warehouse']:12s} | "
                      f"{row['sku']:20s} | {row['quantity']:8,.0f} units | {dist_str}")
        else:
            print("No rebalancing needed - all warehouses are optimally stocked!")
        
        print("\n" + "="*70 + "\n")