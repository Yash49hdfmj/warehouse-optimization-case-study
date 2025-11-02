import pandas as pd
import numpy as np
from typing import Tuple, Any, Optional
from math import radians, sin, cos, asin, sqrt


class OptimizationEngine:
    """Linear programming optimizer for warehouse rebalancing"""
    
    def __init__(self, config, city_locations):
        """
        Initialize optimizer
        
        Args:
            config: OptimizerConfig instance
            city_locations: Dictionary of city locations
        """
        self.config = config
        self.city_locations = city_locations
        self.problem: Optional[Any] = None
        self.status: Optional[str] = None
    
    def calculate_distance(self, city1: str, city2: str) -> float:
        """Calculate haversine distance between two cities"""
        loc1 = self._get_city_location(city1)
        loc2 = self._get_city_location(city2)
        
        R = 6371.0  # Earth radius in km
        
        lat1, lon1 = radians(loc1.lat), radians(loc1.lon)
        lat2, lon2 = radians(loc2.lat), radians(loc2.lon)
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * asin(sqrt(a))
        
        return R * c
    
    def _get_city_location(self, city_name: str):
        """Get location for city with fuzzy matching"""
        city_name_clean = str(city_name).strip()
        
        # Exact match
        if city_name_clean in self.city_locations:
            return self.city_locations[city_name_clean]
        
        # Fuzzy match
        for city, location in self.city_locations.items():
            if city.lower() in city_name_clean.lower():
                return location
        
        # Default to Mumbai
        return self.city_locations["Mumbai"]
    
    def optimize(self, inventory: pd.DataFrame, 
                 demand_forecast: pd.DataFrame) -> Tuple[Any, str]:
        """
        Create and solve warehouse rebalancing optimization problem
        
        Args:
            inventory: Current inventory by warehouse and SKU
            demand_forecast: Forecasted daily demand
            
        Returns:
            Tuple of (optimization problem (or None on error), status)
        """
        print("\n" + "="*60)
        print("STEP 4: OPTIMIZATION")
        print("="*60)
        
        try:
            from pulp import LpProblem, LpVariable, LpMinimize, lpSum, PULP_CBC_CMD, LpStatus
            
            if inventory.empty or demand_forecast.empty:
                print("  ✗ Empty inventory or demand forecast")
                return LpProblem("Empty", LpMinimize), "Empty"
            
            print(f"\n  → Configuration:")
            print(f"      Target days: {self.config.target_days}")
            print(f"      Service level: {self.config.service_level:.0%}")
            print(f"      Optimization: {self.config.optimize_weight}")
            
            # Prepare inventory data
            inv = inventory[['warehouse', 'sku', 'on_hand']].copy()
            inv['on_hand'] = pd.to_numeric(inv['on_hand'], errors='coerce').fillna(0)
            inv = inv[inv['on_hand'] > 0]  # Only consider items in stock
            inv_grouped = inv.groupby(['warehouse', 'sku'])['on_hand'].sum().reset_index()
            
            # Get unique warehouses and SKUs
            warehouses = sorted(inv_grouped['warehouse'].unique().tolist())
            skus = sorted(inv_grouped['sku'].unique().tolist())
            
            print(f"\n  → Problem size:")
            print(f"      Warehouses: {len(warehouses)}")
            print(f"      SKUs: {len(skus)}")
            
            if len(warehouses) < 2:
                print("  ✗ Need at least 2 warehouses for optimization")
                return LpProblem("Empty", LpMinimize), "Empty"
            
            # Calculate target inventory
            demand_avg = demand_forecast.groupby(['warehouse', 'sku'])['orders'].mean().reset_index()
            demand_avg.columns = ['warehouse', 'sku', 'daily_orders']
            
            # Merge with inventory
            data = inv_grouped.merge(demand_avg, on=['warehouse', 'sku'], how='left')
            data['daily_orders'] = data['daily_orders'].fillna(0)
            
            # Calculate target with service level
            sla_multiplier = 1.0 + (self.config.service_level - 0.9)
            data['target'] = self.config.target_days * data['daily_orders'] * sla_multiplier
            
            print(f"      Total current inventory: {data['on_hand'].sum():,.0f} units")
            print(f"      Total target inventory: {data['target'].sum():,.0f} units")
            
            # Calculate distances
            print(f"\n  → Calculating distances...")
            warehouse_pairs = [(i, j) for i in warehouses for j in warehouses if i != j]
            distances = {(i, j): self.calculate_distance(i, j) for i, j in warehouse_pairs}
            
            # Create optimization problem
            print(f"\n  → Building optimization model...")
            prob = LpProblem("WarehouseRebalancing", LpMinimize)
            
            # Decision variables
            X = {}
            for i in warehouses:
                for j in warehouses:
                    if i != j:
                        for s in skus:
                            X[(i, j, s)] = LpVariable(f"x_{i}_{j}_{s}", lowBound=0)
            
            print(f"      Decision variables: {len(X):,}")
            
            # Objective function
            alpha = 0.5 if self.config.optimize_weight == "Balanced" else (
                1.0 if self.config.optimize_weight == "Cost" else 0.0
            )
            
            cost_term = lpSum(
                self.config.cost_per_km * distances[(i, j)] * X[(i, j, s)]
                for (i, j, s) in X
            )
            
            co2_term = lpSum(
                (self.config.co2_per_ton_km / 1000.0) * distances[(i, j)] * X[(i, j, s)]
                for (i, j, s) in X
            )
            
            prob += alpha * cost_term + (1 - alpha) * co2_term
            
            # Constraints
            target_dict = {(r.warehouse, r.sku): r.target for _, r in data.iterrows()}
            onhand_dict = {(r.warehouse, r.sku): r.on_hand for _, r in data.iterrows()}
            
            print(f"  → Adding constraints...")
            for w in warehouses:
                for s in skus:
                    outbound = [X[(i, j, s)] for (i, j, ss) in X if i == w and ss == s]
                    inbound  = [X[(i, j, s)] for (i, j, ss) in X if j == w and ss == s]
                    
                    # Inventory balance
                    prob += (
                        onhand_dict.get((w, s), 0) - lpSum(outbound) + lpSum(inbound)
                        >= target_dict.get((w, s), 0)
                    )
                    
                    # Cannot ship more than available
                    prob += lpSum(outbound) <= onhand_dict.get((w, s), 0)
            
            # Solve
            print(f"\n  → Solving optimization problem...")
            solver = PULP_CBC_CMD(msg=False, timeLimit=300)
            prob.solve(solver)
            
            self.problem = prob
            self.status = LpStatus[prob.status]
            
            print(f"  ✓ Optimization status: {self.status}")
            if self.status == "Optimal":
                print(f"  ✓ Optimal solution found")
            
            return prob, self.status
        
        except ImportError as e:
            print(f"  ✗ Missing package: {str(e)}")
            print("  Install with: pip install pulp")
            raise
        except Exception as e:
            print(f"  ✗ Optimization error: {str(e)}")
            import traceback
            traceback.print_exc()
            # Don't reference possibly-unbound names here
            return None, "Error"
    
    def extract_plan(self, problem: Any) -> pd.DataFrame:
        """Extract rebalancing plan from solved problem"""
        # Guard for None or placeholder problems
        if problem is None or not hasattr(problem, "variables") or getattr(problem, "name", "") in ["Empty", "Error"]:
            return pd.DataFrame(columns=['from_warehouse', 'to_warehouse', 'sku', 'quantity'])
        
        rows = []
        for v in problem.variables():
            # Only consider positive flows (avoid tiny numerical noise)
            if getattr(v, "varValue", 0) and v.varValue > 0.1:
                # Expect variable name format: x_<src>_<dst>_<sku>
                parts = v.name.split("_", 3)
                if len(parts) == 4 and parts[0] == "x":
                    _, src, dst, sku = parts
                    rows.append({
                        'from_warehouse': src,
                        'to_warehouse': dst,
                        'sku': sku,
                        'quantity': round(float(v.varValue), 2)
                    })
        
        return pd.DataFrame(rows)
