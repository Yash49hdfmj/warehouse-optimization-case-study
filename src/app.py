"""
WAREHOUSE OPTIMIZATION SYSTEM - STREAMLIT DASHBOARD
====================================================
Professional logistics dashboard with dynamic visualizations

Developed by: Yash Gadhave
Date: November 2025
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import custom modules
from config import OptimizerConfig, CITY_LOCATIONS
from warehouse_optimizer import WarehouseOptimizer

# Page configuration
st.set_page_config(
    page_title="Warehouse Optimization Dashboard",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional design
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 50%, #0f172a 100%);
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e293b 0%, #0f172a 100%);
        border-right: 1px solid rgba(59, 130, 246, 0.2);
    }
    
    /* Metric cards */
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 700;
        color: #ffffff !important;
    }
    
    [data-testid="stMetricLabel"] {
        color: #e2e8f0 !important;
    }
    
    [data-testid="stMetricDelta"] {
        color: #94a3b8 !important;
    }
    
    /* Headers */
    h1 {
        color: #ffffff;
        font-weight: 800;
        text-shadow: 0 0 20px rgba(59, 130, 246, 0.3);
    }
    
    h2, h3 {
        color: #e2e8f0;
        font-weight: 600;
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #3b82f6 0%, #06b6d4 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(59, 130, 246, 0.3);
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(59, 130, 246, 0.5);
    }
    
    /* Select boxes */
    .stSelectbox label {
        color: #e2e8f0 !important;
    }
    
    .stSelectbox div[data-baseweb="select"] {
        background-color: rgba(30, 41, 59, 0.8);
    }
    
    .stSelectbox div[data-baseweb="select"] > div {
        color: #ffffff !important;
    }
    
    /* Slider labels */
    .stSlider label {
        color: #e2e8f0 !important;
    }
    
    .stSlider div[data-baseweb="slider"] {
        color: #ffffff !important;
    }
    
    /* Cards */
    div[data-testid="stVerticalBlock"] > div[data-testid="stVerticalBlock"] {
        background: rgba(30, 41, 59, 0.6);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(59, 130, 246, 0.2);
        border-radius: 12px;
        padding: 1rem;
    }
    
    /* Tables */
    .dataframe {
        background: rgba(15, 23, 42, 0.8);
        border-radius: 8px;
        color: #ffffff !important;
    }
    
    .dataframe tbody tr {
        color: #ffffff !important;
    }
    
    .dataframe thead tr th {
        color: #3b82f6 !important;
        background-color: rgba(30, 41, 59, 0.9) !important;
    }
    
    /* Tab labels */
    .stTabs [data-baseweb="tab-list"] button {
        color: #94a3b8;
    }
    
    .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
        color: #3b82f6;
    }
    
    /* Text elements */
    p, span, div {
        color: inherit;
    }
    
    /* Info/success/warning boxes */
    .stAlert p {
        color: #1e293b !important;
    }
    
    /* Info boxes */
    .stAlert {
        background: rgba(59, 130, 246, 0.1);
        border: 1px solid rgba(59, 130, 246, 0.3);
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'optimizer' not in st.session_state:
    st.session_state.optimizer = None
if 'results' not in st.session_state:
    st.session_state.results = None
if 'last_run' not in st.session_state:
    st.session_state.last_run = None

# Sidebar - Configuration & Filters
with st.sidebar:
    # Logo and title
    st.markdown("""
        <div style='text-align: center; padding: 1rem 0; margin-bottom: 2rem;'>
            <h1 style='font-size: 1.8rem; margin: 0; color: #3b82f6;'>📦 Warehouse</h1>
            <h1 style='font-size: 1.8rem; margin: 0; color: #06b6d4;'>Optimization</h1>
            <p style='color: #64748b; margin-top: 0.5rem; font-size: 0.9rem;'>AI-Powered Supply Chain Intelligence</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("### 🎯 Optimization Settings")
    
    target_days = st.slider(
        "Target Days of Supply",
        min_value=3,
        max_value=30,
        value=14,
        help="Target inventory level in days"
    )
    
    service_level = st.slider(
        "Service Level (%)",
        min_value=85,
        max_value=99,
        value=95,
        help="Target service level percentage"
    ) / 100
    
    optimize_weight = st.selectbox(
        "Optimization Focus",
        ["Balanced", "Cost", "CO2"],
        help="Balance between cost and emissions"
    )
    
    st.markdown("---")
    
    st.markdown("### 🔍 Filters")
    
    # Load available options from data
    try:
        orders = pd.read_csv('cleaned_orders.csv')
        inventory = pd.read_csv('cleaned_warehouse_inventory.csv')
        
        warehouses = ['All'] + sorted(inventory['Location'].unique().tolist())
        products = ['All'] + sorted(inventory['Product_Category'].unique().tolist())
    except:
        warehouses = ['All', 'Mumbai', 'Delhi', 'Bangalore', 'Chennai', 'Kolkata']
        products = ['All', 'Electronics', 'Fashion', 'Food & Beverage', 'Healthcare', 'Industrial', 'Books', 'Home Goods']
    
    selected_warehouse = st.selectbox(
        "Warehouse Location",
        warehouses,
        help="Filter by warehouse"
    )
    
    selected_product = st.selectbox(
        "Product Category",
        products,
        help="Filter by product category"
    )
    
    date_range = st.selectbox(
        "Date Range",
        ["Last 7 Days", "Last 30 Days", "Last 90 Days"],
        index=1
    )
    
    inventory_status = st.selectbox(
        "Inventory Status",
        ["All", "Critical", "Optimal", "Excess"],
        help="Filter by inventory health"
    )
    
    st.markdown("---")
    
    # Run optimization button
    if st.button("🚀 Run Optimization", use_container_width=True):
        with st.spinner("Running optimization pipeline..."):
            config = OptimizerConfig(
                target_days=target_days,
                service_level=service_level,
                cost_per_km=15.0,
                co2_per_ton_km=62.0,
                optimize_weight=optimize_weight
            )
            
            try:
                optimizer = WarehouseOptimizer(".", config)
                results = optimizer.run()
                
                st.session_state.optimizer = optimizer
                st.session_state.results = results
                st.session_state.last_run = datetime.now()
                
                st.success("✅ Optimization completed!")
                st.rerun()
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

# Main content
st.markdown("""
    <h1 style='text-align: center; font-size: 3rem; margin-bottom: 0; color: #ffffff;'>
        Warehouse Optimization Dashboard
    </h1>
    <p style='text-align: center; color: #94a3b8; font-size: 1.1rem; margin-top: 0.5rem;'>
        Real-time Supply Chain Analytics & AI-Powered Forecasting
    </p>
    <p style='text-align: center; color: #64748b; font-size: 0.95rem; margin-top: 0.3rem; font-style: italic;'>
        Developed by Yash Gadhave
    </p>
""", unsafe_allow_html=True)

st.markdown("---")

# Check if optimization has been run
if st.session_state.optimizer is None:
    st.info("👈 Configure settings and click 'Run Optimization' to get started")
    
    # Show placeholder metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Inventory", "---", help="Total units in stock")
    with col2:
        st.metric("Avg Days of Supply", "---", help="Average inventory coverage")
    with col3:
        st.metric("Critical Items", "---", help="Items needing attention")
    with col4:
        st.metric("Recommended Transfers", "---", help="Units to rebalance")
    
else:
    # Get data
    optimizer = st.session_state.optimizer
    results = st.session_state.results
    
    # Apply filters to inventory analysis
    inv_analysis = optimizer.get_inventory_analysis()
    
    if not inv_analysis.empty:
        filtered_inv = inv_analysis.copy()
        
        if selected_warehouse != 'All':
            filtered_inv = filtered_inv[filtered_inv['warehouse'] == selected_warehouse]
        
        if selected_product != 'All':
            filtered_inv = filtered_inv[filtered_inv['sku'] == selected_product]
        
        if inventory_status != 'All':
            filtered_inv = filtered_inv[filtered_inv['status'] == inventory_status]
    else:
        filtered_inv = pd.DataFrame()
    
    # Calculate KPIs
    if not filtered_inv.empty:
        total_units = filtered_inv['on_hand'].sum()
        avg_days = filtered_inv['days_of_supply'].replace([np.inf], 999).mean()
        critical_items = len(filtered_inv[filtered_inv['status'] == 'Critical'])
    else:
        total_units = 0
        avg_days = 0
        critical_items = 0
    
    plan = optimizer.rebalancing_plan
    total_transfers = plan['quantity'].sum() if not plan.empty else 0
    
    # KPI Cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "📦 Total Inventory",
            f"{total_units:,.0f}",
            delta=None,
            help="Total units across selected filters"
        )
    
    with col2:
        delta_days = avg_days - target_days
        st.metric(
            "⏰ Avg Days of Supply",
            f"{avg_days:.1f}",
            delta=f"{delta_days:+.1f} vs target",
            delta_color="inverse",
            help=f"Target: {target_days} days"
        )
    
    with col3:
        st.metric(
            "⚠️ Critical Items",
            f"{critical_items}",
            delta=None,
            help="Items requiring immediate attention"
        )
    
    with col4:
        st.metric(
            "🔄 Recommended Transfers",
            f"{total_transfers:,.0f}",
            delta=None,
            help="Units to rebalance across network"
        )
    
    st.markdown("---")
    
    # Main visualizations
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Demand Forecast", 
        "🎯 Model Performance", 
        "📊 Inventory Analysis", 
        "🚚 Rebalancing Plan"
    ])
    
    with tab1:
        st.markdown("### 📈 Demand Forecast vs Actual Orders")
        
        daily_features = optimizer.daily_features
        
        if not daily_features.empty and 'predicted_orders' in daily_features.columns:
            # Apply filters
            filtered_forecast = daily_features.copy()
            if selected_warehouse != 'All':
                filtered_forecast = filtered_forecast[filtered_forecast['warehouse'] == selected_warehouse]
            if selected_product != 'All':
                filtered_forecast = filtered_forecast[filtered_forecast['sku'] == selected_product]
            
            # Aggregate by date
            daily_agg = filtered_forecast.groupby('date').agg({
                'orders': 'sum',
                'predicted_orders': 'sum'
            }).reset_index()
            
            # Create line chart
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=daily_agg['date'],
                y=daily_agg['orders'],
                mode='lines+markers',
                name='Actual Orders',
                line=dict(color='#3b82f6', width=3),
                marker=dict(size=8, symbol='circle')
            ))
            
            fig.add_trace(go.Scatter(
                x=daily_agg['date'],
                y=daily_agg['predicted_orders'],
                mode='lines+markers',
                name='Predicted Orders',
                line=dict(color='#06b6d4', width=3, dash='dash'),
                marker=dict(size=8, symbol='diamond')
            ))
            
            fig.update_layout(
                template='plotly_dark',
                paper_bgcolor='rgba(15, 23, 42, 0.8)',
                plot_bgcolor='rgba(30, 41, 59, 0.5)',
                font=dict(color='#e2e8f0'),
                height=400,
                hovermode='x unified',
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                avg_actual = daily_agg['orders'].mean()
                st.metric("Avg Daily Orders", f"{avg_actual:.1f}")
            with col2:
                avg_predicted = daily_agg['predicted_orders'].mean()
                st.metric("Avg Predicted", f"{avg_predicted:.1f}")
            with col3:
                error_pct = abs(avg_actual - avg_predicted) / avg_actual * 100 if avg_actual > 0 else 0
                st.metric("Error %", f"{error_pct:.1f}%")
    
    with tab2:
        st.markdown("### 🎯 ML Model Performance Comparison")
        
        metrics = optimizer.metrics
        daily_features = optimizer.daily_features
        
        # Create model comparison data - recalculate for current filters
        model_data = []
        
        # Get filtered forecast data for accuracy calculation
        filtered_forecast = daily_features.copy()
        if selected_warehouse != 'All':
            filtered_forecast = filtered_forecast[filtered_forecast['warehouse'] == selected_warehouse]
        if selected_product != 'All':
            filtered_forecast = filtered_forecast[filtered_forecast['sku'] == selected_product]
        
        # Calculate mean for accuracy baseline
        mean_orders = filtered_forecast['orders'].mean() if not filtered_forecast.empty and filtered_forecast['orders'].mean() > 0 else 1
        
        # Check if multi-model results are available
        if hasattr(optimizer.forecaster, 'all_model_results') and optimizer.forecaster.all_model_results:
            for model_name, model_metrics in optimizer.forecaster.all_model_results.items():
                if model_metrics.get('mae') is not None:
                    # Recalculate accuracy based on filtered data
                    relative_error = (model_metrics['mae'] / mean_orders) * 100
                    accuracy = max(0, min(100, 100 - relative_error))
                    model_data.append({
                        'Model': model_name,
                        'MAE': model_metrics['mae'],
                        'RMSE': model_metrics['rmse'],
                        'Accuracy': accuracy
                    })
        else:
            # Single model result
            if metrics and metrics.get('mae') is not None:
                relative_error = (metrics['mae'] / mean_orders) * 100
                accuracy = max(0, min(100, 100 - relative_error))
                model_data.append({
                    'Model': 'LightGBM',
                    'MAE': metrics['mae'],
                    'RMSE': metrics['rmse'],
                    'Accuracy': accuracy
                })
        
        if model_data:
            model_df = pd.DataFrame(model_data)
            
            # Bar chart
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                x=model_df['Model'],
                y=model_df['Accuracy'],
                marker_color=['#10b981', '#3b82f6', '#f59e0b'][:len(model_df)],
                text=model_df['Accuracy'].round(1),
                textposition='outside',
                textfont=dict(color='#ffffff'),
                name='Accuracy %'
            ))
            
            fig.update_layout(
                template='plotly_dark',
                paper_bgcolor='rgba(15, 23, 42, 0.8)',
                plot_bgcolor='rgba(30, 41, 59, 0.5)',
                font=dict(color='#e2e8f0'),
                height=350,
                yaxis_title="Accuracy (%)",
                xaxis_title="",
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Model details
            st.markdown("#### Model Details")
            cols = st.columns(len(model_df))
            for idx, (col, row) in enumerate(zip(cols, model_df.to_dict('records'))):
                with col:
                    st.markdown(f"""
                        <div style='background: rgba(30, 41, 59, 0.8); padding: 1rem; border-radius: 8px; border: 1px solid rgba(59, 130, 246, 0.3);'>
                            <h4 style='color: #ffffff; margin: 0;'>{row['Model']}</h4>
                            <p style='font-size: 1.5rem; font-weight: bold; margin: 0.5rem 0; color: #10b981;'>{row['Accuracy']:.1f}%</p>
                            <p style='color: #e2e8f0; margin: 0; font-size: 0.9rem;'>MAE: {row['MAE']:.3f}</p>
                            <p style='color: #e2e8f0; margin: 0; font-size: 0.9rem;'>RMSE: {row['RMSE']:.3f}</p>
                        </div>
                    """, unsafe_allow_html=True)
            
            # Add filter context info

    
    with tab3:
        st.markdown("### 📊 Inventory Distribution & Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Status distribution pie chart
            if not filtered_inv.empty and 'status' in filtered_inv.columns:
                status_counts = filtered_inv['status'].value_counts()
                
                colors = {'Critical': '#ef4444', 'Optimal': '#10b981', 'Excess': '#f59e0b'}
                
                fig = go.Figure(data=[go.Pie(
                    labels=status_counts.index,
                    values=status_counts.values,
                    hole=0.4,
                    marker=dict(colors=[colors.get(x, '#64748b') for x in status_counts.index]),
                    textinfo='label+percent',
                    textfont=dict(size=14)
                )])
                
                fig.update_layout(
                    template='plotly_dark',
                    paper_bgcolor='rgba(15, 23, 42, 0.8)',
                    font=dict(color='#e2e8f0'),
                    height=350,
                    showlegend=True
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Warehouse stock levels
            if not filtered_inv.empty:
                warehouse_summary = filtered_inv.groupby('warehouse').agg({
                    'on_hand': 'sum'
                }).reset_index().sort_values('on_hand', ascending=True)
                
                fig = go.Figure(go.Bar(
                    x=warehouse_summary['on_hand'],
                    y=warehouse_summary['warehouse'],
                    orientation='h',
                    marker_color='#3b82f6',
                    text=warehouse_summary['on_hand'].apply(lambda x: f'{x:,.0f}'),
                    textposition='outside'
                ))
                
                fig.update_layout(
                    template='plotly_dark',
                    paper_bgcolor='rgba(15, 23, 42, 0.8)',
                    plot_bgcolor='rgba(30, 41, 59, 0.5)',
                    font=dict(color='#e2e8f0'),
                    height=350,
                    xaxis_title="Stock Units",
                    yaxis_title="",
                    showlegend=False
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        # Detailed table
        # Detailed table
        st.markdown("#### Detailed Inventory Report")

        # --- CSS once (safe to leave here if you prefer) ---
        st.markdown("""
        <style>
        [data-testid="stDataFrame"] table,
        [data-testid="stDataFrame"] thead th,
        [data-testid="stDataFrame"] tbody td {
        color: #000 !important;
        }
        [data-testid="stDataFrame"] .st-ag-theme, 
        [data-testid="stDataFrame"] table {
        background-color: #ffffff !important;
        }
        </style>
        """, unsafe_allow_html=True)

        if not filtered_inv.empty:
            display_df = filtered_inv[['warehouse', 'sku', 'on_hand', 'avg_daily_orders', 'days_of_supply', 'status']].copy()
            display_df.columns = ['Warehouse', 'Product', 'Stock', 'Daily Orders', 'Days Supply', 'Status']

            # If you want striped rows but still black text:
            # (This is optional; st.dataframe ignores the style colors but keeps the numbers formatting if you prefer plain df)
            st.dataframe(
                display_df.style.format({
                    'Stock': '{:,.0f}',
                    'Daily Orders': '{:.2f}',
                    'Days Supply': '{:.1f}'
                }),
                use_container_width=True,
                height=400
            )

    
    with tab4:
        st.markdown("### 🚚 Recommended Rebalancing Plan")
    

        
        if not plan.empty:
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Transfers", len(plan))
            with col2:
                st.metric("Units to Move", f"{plan['quantity'].sum():,.0f}")
            with col3:
                if 'estimated_cost' in plan.columns:
                    st.metric("Total Cost", f"₹{plan['estimated_cost'].sum():,.2f}")
            with col4:
                if 'estimated_co2_kg' in plan.columns:
                    st.metric("CO₂ Emissions", f"{plan['estimated_co2_kg'].sum():,.1f} kg")
            
            st.markdown("---")
            
            # Detailed plan table
            display_plan = plan.copy()
            
            # Format for display
            display_cols = ['from_warehouse', 'to_warehouse', 'sku', 'quantity']
            if 'distance_km' in display_plan.columns:
                display_cols.append('distance_km')
            if 'estimated_cost' in display_plan.columns:
                display_cols.append('estimated_cost')
            if 'estimated_co2_kg' in display_plan.columns:
                display_cols.append('estimated_co2_kg')
            
            display_plan = display_plan[display_cols].copy()
            display_plan.columns = [col.replace('_', ' ').title() for col in display_plan.columns]
            
            # Style table
            st.dataframe(
                display_plan.style.format({
                    'Quantity': '{:,.0f}',
                    'Distance Km': '{:,.1f}',
                    'Estimated Cost': '₹{:,.2f}',
                    'Estimated Co2 Kg': '{:,.1f}'
                }),
                use_container_width=True,
                height=400
            )
            
            # Download button
            csv = plan.to_csv(index=False)
            st.download_button(
                label="📥 Download Rebalancing Plan",
                data=csv,
                file_name=f"rebalancing_plan_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
        else:
            st.success("✅ No rebalancing needed - inventory is optimally distributed!")
            st.info("All warehouses have adequate stock levels based on current demand forecasts.")

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #94a3b8; padding: 2rem 0;'>
        <p style='font-size: 0.9rem; margin: 0; color: #e2e8f0;'>
            <strong>Powered by AI</strong> • LightGBM, XGBoost, CatBoost • Real-time Optimization Engine
        </p>
        <p style='font-size: 0.8rem; margin-top: 0.5rem; color: #94a3b8;'>
            Last Updated: {}</p>
        <p style='font-size: 0.85rem; margin-top: 0.5rem; color: #64748b; font-style: italic;'>
            © 2025 Developed by Yash Gadhave
        </p>
    </div>
""".format(st.session_state.last_run.strftime('%Y-%m-%d %H:%M:%S') if st.session_state.last_run else 'Not yet run'), unsafe_allow_html=True)