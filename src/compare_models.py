"""
MODEL PERFORMANCE COMPARISON UTILITY
====================================
Compares LightGBM, XGBoost, and CatBoost models using DemandForecaster.

- Loads and validates data
- Builds features via FeatureEngineer
- Trains each model individually
- Computes detailed metrics and consistency scores
- Outputs ranked comparison and exports results
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


def compare_all_models():
    """Run comprehensive model comparison"""
    print("\n" + "=" * 80)
    print("COMPREHENSIVE MODEL COMPARISON ANALYSIS")
    print("=" * 80)

    # -----------------------------------------------------------------------
    # Load data
    # -----------------------------------------------------------------------
    print("\n📂 Loading data...")
    from data_loader import DataLoader
    loader = DataLoader(".")
    if not loader.load_all():
        print("❌ Failed to load data")
        return None, None

    # -----------------------------------------------------------------------
    # Feature engineering
    # -----------------------------------------------------------------------
    print("🔧 Engineering features...")
    from feature_engineer import FeatureEngineer
    engineer = FeatureEngineer()
    features = engineer.build_features(loader.orders)

    if features.empty:
        print("❌ Feature engineering failed")
        return None, None

    # -----------------------------------------------------------------------
    # Model testing and comparison
    # -----------------------------------------------------------------------
    results = {}
    models = ["LightGBM", "XGBoost", "CatBoost"]

    print("\n" + "=" * 80)
    print("TRAINING ALL MODELS")
    print("=" * 80)

    from demand_forecaster import DemandForecaster

    for model_name in models:
        print(f"\n{'─' * 80}")
        print(f"Training: {model_name}")
        print(f"{'─' * 80}")

        try:
            forecaster = DemandForecaster(model_type=model_name)
            predictions, metrics = forecaster.train_and_predict(features, train_ratio=0.8)

            # Skip if metrics missing
            if predictions.empty or np.isnan(metrics.get("mae", np.nan)):
                print(f"✗ {model_name} training failed or returned no metrics")
                continue

            # Prepare aligned comparison dataframe
            y_true = features["orders"].astype(float)
            y_pred = predictions.astype(float)

            comparison = pd.DataFrame({
                "actual": y_true,
                "predicted": y_pred
            })
            comparison["error"] = comparison["predicted"] - comparison["actual"]
            comparison["abs_error"] = np.abs(comparison["error"])
            comparison["pct_error"] = np.where(
                comparison["actual"] != 0,
                np.abs(comparison["error"] / comparison["actual"]) * 100,
                np.nan
            )

            comparison = comparison.replace([np.inf, -np.inf], np.nan).dropna()

            results[model_name] = {
                "mae": float(metrics["mae"]),
                "rmse": float(metrics["rmse"]),
                "mape": float(metrics.get("mape", np.nan)),
                "median_error": float(comparison["abs_error"].median()),
                "p90_error": float(comparison["abs_error"].quantile(0.90)),
                "max_error": float(comparison["abs_error"].max()),
                "comparison": comparison
            }

            print(f"✓ {model_name} training completed")

        except Exception as e:
            print(f"✗ {model_name} error: {str(e)}")

    # -----------------------------------------------------------------------
    # Summary comparison
    # -----------------------------------------------------------------------
    if not results:
        print("\n❌ No models completed successfully")
        return None, None

    print("\n" + "=" * 80)
    print("DETAILED MODEL COMPARISON")
    print("=" * 80)

    comparison_data = []
    for model_name, res in results.items():
        comparison_data.append({
            "Model": model_name,
            "MAE": res["mae"],
            "RMSE": res["rmse"],
            "MAPE": res["mape"],
            "Median Error": res["median_error"],
            "90th Percentile": res["p90_error"],
            "Max Error": res["max_error"]
        })

    comparison_df = pd.DataFrame(comparison_data)
    comparison_df = comparison_df.sort_values("MAE").reset_index(drop=True)

    print("\n" + comparison_df.to_string(index=False))

    # -----------------------------------------------------------------------
    # Highlight best model
    # -----------------------------------------------------------------------
    best_model = comparison_df.iloc[0]["Model"]
    best_mae = comparison_df.iloc[0]["MAE"]

    print(f"\n{'═' * 80}")
    print(f"🏆 WINNER: {best_model}")
    print(f"{'═' * 80}")
    print(f"  Best MAE: {best_mae:.3f} orders/day")

    if len(comparison_df) > 1:
        worst_mae = comparison_df.iloc[-1]["MAE"]
        improvement = ((worst_mae - best_mae) / worst_mae) * 100
        print(f"  Improvement over worst: {improvement:.1f}%")

    # -----------------------------------------------------------------------
    # Performance insights
    # -----------------------------------------------------------------------
    print(f"\n{'─' * 80}")
    print("PERFORMANCE ANALYSIS")
    print(f"{'─' * 80}")

    for model_name, res in results.items():
        marker = "→" if model_name == best_model else " "
        print(f"\n{marker} {model_name}:")
        print(f"    Mean Error:        {res['mae']:.3f} ± {res['rmse'] - res['mae']:.3f}")
        print(f"    Median Error:      {res['median_error']:.3f}")
        print(f"    90th Percentile:   {res['p90_error']:.3f}")
        print(f"    Worst Case:        {res['max_error']:.3f}")

        comp = res["comparison"]
        within_10pct = (comp["pct_error"] <= 10).sum() / len(comp) * 100
        within_25pct = (comp["pct_error"] <= 25).sum() / len(comp) * 100

        print(f"    Within 10% error:  {within_10pct:.1f}% of predictions")
        print(f"    Within 25% error:  {within_25pct:.1f}% of predictions")

    # -----------------------------------------------------------------------
    # Recommendations
    # -----------------------------------------------------------------------
    print(f"\n{'═' * 80}")
    print("RECOMMENDATIONS")
    print(f"{'═' * 80}")

    recommendations = []
    best_res = results[best_model]

    # Based on MAE
    if best_mae < 3.0:
        recommendations.append("✓ EXCELLENT accuracy - Deploy with confidence")
    elif best_mae < 5.0:
        recommendations.append("✓ GOOD accuracy - Suitable for production use")
    else:
        recommendations.append("⚠ Consider more training data or feature engineering")

    # Based on variance
    if best_res["p90_error"] < best_res["mae"] * 2:
        recommendations.append("✓ Consistent predictions - Low variance")
    else:
        recommendations.append("⚠ High variance - Review outliers")

    # Based on MAPE
    if best_res["mape"] < 20:
        recommendations.append("✓ Business-ready forecasting accuracy")
    elif best_res["mape"] < 30:
        recommendations.append("⚠ Acceptable for planning, monitor closely")
    else:
        recommendations.append("⚠ High relative error - Add domain-specific features")

    for i, rec in enumerate(recommendations, 1):
        print(f"\n  {i}. {rec}")

    # -----------------------------------------------------------------------
    # Export results
    # -----------------------------------------------------------------------
    print(f"\n{'─' * 80}")
    print("EXPORTING RESULTS")
    print(f"{'─' * 80}")

    output_file = f"optimization_results/model_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    comparison_df.to_csv(output_file, index=False)
    print(f"\n  ✓ Saved to: {output_file}")

    # Detailed report
    report_file = f"optimization_results/comparison_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("MODEL COMPARISON REPORT\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(comparison_df.to_string(index=False))
        f.write("\n\n")
        f.write(f"Winner: {best_model}\n")
        f.write(f"Best MAE: {best_mae:.3f}\n\n")
        f.write("Recommendations:\n")
        for rec in recommendations:
            f.write(f"  {rec}\n")

    print(f"  ✓ Report saved to: {report_file}")

    print("\n" + "=" * 80)
    print("✓ COMPARISON COMPLETED SUCCESSFULLY")
    print("=" * 80 + "\n")

    return results, best_model


# ---------------------------------------------------------------------------
# QUICK MODE
# ---------------------------------------------------------------------------
def quick_comparison():
    """Lightweight quick model comparison for testing/debugging"""
    print("\n🚀 QUICK MODEL COMPARISON (Sampled data)\n")

    from data_loader import DataLoader
    from feature_engineer import FeatureEngineer
    from demand_forecaster import DemandForecaster

    loader = DataLoader(".")
    loader.load_all()

    engineer = FeatureEngineer()
    features = engineer.build_features(loader.orders)

    if len(features) > 100:
        features_sample = features.head(100)
    else:
        features_sample = features

    print(f"Testing with {len(features_sample)} samples\n")

    models = ["LightGBM", "XGBoost", "CatBoost"]
    results = {}

    for model_name in models:
        print(f"Testing {model_name}...", end=" ")
        try:
            forecaster = DemandForecaster(model_type=model_name)
            _, metrics = forecaster.train_and_predict(features_sample, train_ratio=0.8)
            if not np.isnan(metrics.get("mae", np.nan)):
                results[model_name] = metrics["mae"]
                print(f"MAE: {metrics['mae']:.3f}")
            else:
                print("Failed")
        except Exception as e:
            print(f"Error: {str(e)}")

    if results:
        best = min(results.items(), key=lambda x: x[1])
        print(f"\n🏆 Best: {best[0]} (MAE: {best[1]:.3f})")

    return results


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    import os

    os.makedirs("optimization_results", exist_ok=True)

    if len(sys.argv) > 1 and sys.argv[1] == "--quick":
        quick_comparison()
    else:
        compare_all_models()
