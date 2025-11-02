# Warehouse Optimization and Forecasting Dashboard

**Developed by:** Yash Gadhave  
**Language:** Python  
**Libraries:** Streamlit, Plotly, Pandas, NumPy, LightGBM, XGBoost, CatBoost, PuLP

---

## Overview

This project implements a warehouse optimization and demand forecasting system designed to improve operational efficiency, balance inventory, and support data-driven logistics decisions.

It combines predictive modeling with optimization and interactive visualization through a Streamlit dashboard. The system processes operational data, forecasts demand using multiple machine-learning models, and recommends stock redistribution plans between warehouses.

---

## Features

- **Automated Data Processing:** Cleans, merges, and validates multiple operational datasets.  
- **Demand Forecasting:** Uses LightGBM, XGBoost, and CatBoost models for order prediction.  
- **Inventory Optimization:** Suggests optimal inter-warehouse transfers to minimize cost and CO₂ impact.  
- **Interactive Dashboard:** Built with Streamlit and Plotly for real-time data exploration.  
- **Model Comparison:** Evaluates multiple gradient boosting models to identify the most accurate forecaster.  
- **Exportable Outputs:** Forecasts, metrics, and transfer plans saved as CSV and JSON reports.

---

## Directory Structure


```

src/
│
├── app.py                        # Streamlit dashboard
├── main.py                       # Main pipeline entry point
├── compare_models.py             # Model training and comparison script
├── data_loader.py                # Data import and validation
├── feature_engineer.py           # Feature creation and transformation
├── demand_forecaster.py          # Forecast model training and evaluation
├── optimizer.py                  # Inventory optimization engine
├── warehouse_optimizer.py        # Pipeline orchestration
├── config.py                     # Configuration parameters
│
├── cleaned_cost_breakdown.csv
├── cleaned_customer_feedback.csv
├── cleaned_delivery_performance.csv
├── cleaned_orders.csv
├── cleaned_routes.csv
├── cleaned_vehicles.csv
├── cleaned_warehouse_inventory.csv
│
└── optimization_results/
├── demand_forecast.csv
├── inventory_analysis.csv
├── rebalancing_plan.csv
├── metrics.json
├── model_comparison_*.csv
└── comparison_report_*.txt

````

---

## Model Comparison Results

Results obtained from `compare_models.py`:

| Model     | MAE   | RMSE  | MAPE  |
|------------|-------|-------|-------|
| LightGBM   | 0.123 | 0.393 | 6.29% |
| XGBoost    | 0.117 | 0.389 | 5.77% |
| CatBoost   | 0.103 | 0.392 | 4.27% |

**Selected Model:** CatBoost  
**Accuracy:** 95.8% of predictions within 10% error margin  
**Recommendation:** Excellent forecasting performance suitable for deployment

---

## Dashboard Overview

The Streamlit dashboard provides the following tabs:

1. **Demand Forecast** – Actual vs predicted orders with interactive trend lines.  
2. **Model Performance** – Comparison of MAE, RMSE, and MAPE across models.  
3. **Inventory Analysis** – Warehouse-level stock visualization and detailed tables.  
4. **Rebalancing Plan** – Transfer recommendations with cost and CO₂ estimation.

---

## Running the Application

### Step 1. Train and Run test file and main file and Compare Models
```bash
cd src
python test.py
python main.py
python compare_models.py
````

This script trains LightGBM, XGBoost, and CatBoost models and exports their metrics to the `optimization_results/` directory.

### Step 2. Launch the Dashboard

```bash
streamlit run app.py
```

Access the interface at:

```
http://localhost:8501
```

---

## Output Files

| File                    | Description                      |
| ----------------------- | -------------------------------- |
| demand_forecast.csv     | Predicted daily order quantities |
| inventory_analysis.csv  | Stock vs demand summary          |
| rebalancing_plan.csv    | Transfer optimization output     |
| metrics.json            | Evaluation metrics summary       |
| model_comparison_*.csv  | Model comparison table           |
| comparison_report_*.txt | Detailed textual report          |

---

## Dependencies

Install required packages before running:

```bash
pip install -r requirements.txt
```

**Primary Libraries**

* streamlit
* plotly
* pandas
* numpy
* lightgbm
* xgboost
* catboost
* scikit-learn
* pulp

---

## Results Summary

* CatBoost achieved the lowest mean absolute error (0.103).
* All models produced less than 15% relative error.
* Forecasts and optimization outputs are reliable for practical use.

---

## License and Attribution

This repository is intended for research and educational purposes.
All rights reserved © 2025, **Yash Gadhave**.



## Dashboard Preview

<p align="center">
  <img src="https://github.com/Yash49hdfmj/warehouse-optimization-case-study/blob/main/screenshots/demand%20forcast.png" width="80%" alt="Demand Forecast">
</p>

<p align="center">
  <img src="https://github.com/Yash49hdfmj/warehouse-optimization-case-study/blob/main/screenshots/inventoryanalysis.png" width="80%" alt="Inventory Analysis">
</p>

<p align="center">
  <img src="https://github.com/Yash49hdfmj/warehouse-optimization-case-study/blob/main/screenshots/model%20performance.png" width="80%" alt="Model Performance">
</p>

<p align="center">
  <img src="https://github.com/Yash49hdfmj/warehouse-optimization-case-study/blob/main/screenshots/rebalancing%20plan.png" width="80%" alt="Rebalancing Plan">
</p>



```
♥ yash



