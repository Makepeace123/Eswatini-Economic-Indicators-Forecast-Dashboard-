# -*- coding: utf-8 -*-
"""Eswatini Economic Forecast Dashboard - Streamlit Version"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import io
import base64

# Page configuration
st.set_page_config(
    page_title="Eswatini Economic Forecast",
    page_icon="🇸🇿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #3498db;
        margin-bottom: 1rem;
    }
    .forecast-card {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    .positive-change {
        color: #27ae60;
        font-weight: bold;
    }
    .negative-change {
        color: #e74c3c;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'current_variable' not in st.session_state:
    st.session_state.current_variable = 'Maize meal SZL/1kg'

# Sample data generation
@st.cache_data
def generate_sample_data():
    """Generate comprehensive sample data for all 10 variables"""
    dates = pd.date_range(start='2020-01-01', end='2024-12-31', freq='D')
    n_days = len(dates)
    
    np.random.seed(42)
    
    # Base trends for each variable
    data = {
        'date': dates,
        'Maize meal SZL/1kg': np.linspace(8, 25, n_days) + np.random.normal(0, 1, n_days),
        'All Items CPI': np.linspace(100, 145, n_days) + np.random.normal(0, 1, n_days),
        #'Inflation rate': np.random.normal(0.2, 0.1, n_days).cumsum() + 5,
        #'Diesel SZL/1 liter': np.linspace(12, 28, n_days) + np.random.normal(0, 1.5, n_days),
        'Cabbage SZL/Head': np.linspace(5, 15, n_days) + np.random.normal(0, 2, n_days),
        'Tomato (Round) SZL/1kg': np.linspace(10, 30, n_days) + np.random.normal(0, 3, n_days),
        'Rice SZL/1kg': np.linspace(15, 25, n_days) + np.random.normal(0, 1, n_days),
        'Beans SZL/1kg': np.linspace(12, 22, n_days) + np.random.normal(0, 1.2, n_days),
        'Sugar SZL/1kg': np.linspace(8, 18, n_days) + np.random.normal(0, 1, n_days),
        'Interest Rate (Prime lending rate)': np.linspace(8, 12, n_days) + np.random.normal(0, 0.5, n_days)
    }
    
    # Add seasonality and trends
    for i in range(1, len(data)):
        key = list(data.keys())[i]
        seasonal = 3 * np.sin(2 * np.pi * np.arange(n_days) / 365)
        data[key] += seasonal
    
    return pd.DataFrame(data)

@st.cache_data
def generate_forecasts():
    """Generate forecast data for all variables"""
    forecasts = {}
    variables = [
        'Maize meal SZL/1kg', 'All Items CPI', 'Inflation rate', 'Diesel SZL/1 liter',
        'Cabbage SZL/Head', 'Tomato (Round) SZL/1kg', 'Rice SZL/1kg', 'Beans SZL/1kg',
        'Sugar SZL/1kg', 'Interest Rate (Prime lending rate)'
    ]
    
    future_dates = [datetime.now().date() + timedelta(days=i) for i in range(1, 31)]
    
    for variable in variables:
        current_value = np.random.uniform(15, 25)
        trend = np.random.uniform(-0.3, 0.4, 30).cumsum()
        forecasts[variable] = pd.DataFrame({
            'date': future_dates,
            'forecast': current_value + trend,
            'upper_ci': current_value + trend * 1.15,
            'lower_ci': current_value + trend * 0.85
        })
    
    return forecasts

@st.cache_data
def generate_metrics():
    """Generate performance metrics for all models"""
    metrics = {}
    variables = [
        'Maize meal SZL/1kg', 'All Items CPI', 'Inflation rate', 'Diesel SZL/1 liter',
        'Cabbage SZL/Head', 'Tomato (Round) SZL/1kg', 'Rice SZL/1kg', 'Beans SZL/1kg',
        'Sugar SZL/1kg', 'Interest Rate (Prime lending rate)'
    ]
    
    for variable in variables:
        metrics[variable] = {
            'xgb': {
                'MAE': round(np.random.uniform(0.8, 2.5), 3),
                'RMSE': round(np.random.uniform(1.2, 3.8), 3),
                'R2': round(np.random.uniform(0.82, 0.96), 3)
            },
            'mlp': {
                'MAE': round(np.random.uniform(1.0, 3.0), 3),
                'RMSE': round(np.random.uniform(1.5, 4.2), 3),
                'R2': round(np.random.uniform(0.78, 0.92), 3)
            },
            'gru': {
                'MAE': round(np.random.uniform(0.9, 2.8), 3),
                'RMSE': round(np.random.uniform(1.4, 4.0), 3),
                'R2': round(np.random.uniform(0.80, 0.94), 3)
            }
        }
    
    return metrics

@st.cache_data
def generate_feature_importance():
    """Generate feature importance data"""
    features = [
        'Diesel Price', 'Exchange Rate', 'Rainfall', 'GDP Growth', 'Import Costs',
        'Fuel Prices', 'Transport Costs', 'Seasonal Factor', 'Global Prices', 'Local Production'
    ]
    
    importance = {}
    for variable in [
        'Maize meal SZL/1kg', 'All Items CPI', 'Inflation rate', 'Diesel SZL/1 liter',
        'Cabbage SZL/Head', 'Tomato (Round) SZL/1kg', 'Rice SZL/1kg', 'Beans SZL/1kg',
        'Sugar SZL/1kg', 'Interest Rate (Prime lending rate)'
    ]:
        np.random.seed(hash(variable) % 1000)
        imp_values = {feature: round(np.random.uniform(0.1, 1.0), 3) for feature in features}
        # Sort by importance
        importance[variable] = dict(sorted(imp_values.items(), key=lambda x: x[1], reverse=True))
    
    return importance

# Load data
df = generate_sample_data()
forecasts = generate_forecasts()
metrics = generate_metrics()
feature_importance = generate_feature_importance()

# Sidebar
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/1/1e/Flag_of_Eswatini.svg/1200px-Flag_of_Eswatini.svg.png", 
             width=100)
    st.title("🇸🇿 Eswatini Economic Forecast")
    st.markdown("---")
    
    # Variable selection
    selected_variable = st.selectbox(
        "**Select Economic Indicator:**",
        options=[
            'Maize meal SZL/1kg', 'All Items CPI', 'Inflation rate', 'Diesel SZL/1 liter',
            'Cabbage SZL/Head', 'Tomato (Round) SZL/1kg', 'Rice SZL/1kg', 'Beans SZL/1kg',
            'Sugar SZL/1kg', 'Interest Rate (Prime lending rate)'
        ],
        index=0
    )
    
    st.session_state.current_variable = selected_variable
    
    # Forecast parameters
    st.markdown("---")
    st.subheader("Forecast Settings")
    forecast_days = st.slider("Forecast Horizon (days):", 7, 90, 30, 7)
    confidence_level = st.slider("Confidence Level:", 80, 95, 90, 5)
    
    st.markdown("---")
    st.info("""
    **Dashboard Features:**
    - Real-time economic forecasts
    - Multi-model performance comparison
    - Feature importance analysis
    - Interactive visualizations
    - Downloadable reports
    """)

# Main content
st.markdown(f'<h1 class="main-header">📈 {selected_variable} Forecast</h1>', unsafe_allow_html=True)

# Key metrics row
col1, col2, col3, col4 = st.columns(4)

with col1:
    current_value = df[selected_variable].iloc[-1]
    st.metric("Current Value", f"{current_value:.2f}")

with col2:
    forecast_value = forecasts[selected_variable]['forecast'].iloc[0]
    change_pct = ((forecast_value - current_value) / current_value) * 100
    change_color = "positive-change" if change_pct >= 0 else "negative-change"
    st.metric("Next Forecast", f"{forecast_value:.2f}", f"{change_pct:+.1f}%")

with col3:
    best_model = 'xgb'  # Simplified - would compare metrics in real app
    st.metric("Best Model", "XGBoost", f"MAE: {metrics[selected_variable][best_model]['MAE']:.2f}")

with col4:
    volatility = df[selected_variable].pct_change().std() * 100
    st.metric("Volatility", f"{volatility:.1f}%", "30-day average")

# Main forecast plot
#st.markdown("### 📊 Forecast Visualization")
#fig = go.Figure()

# Historical data
#fig.add_trace(go.Scatter(
#    x=df['date'],
#    y=df[selected_variable],
#    mode='lines',
#    name='Historical',
#    line=dict(color='#3498db', width=3),
#    hovertemplate='<b>Date:</b> %{x}<br><b>Value:</b> %{y:.2f}<extra></extra>'
#))

# Forecast data
#forecast_data = forecasts[selected_variable]
#fig.add_trace(go.Scatter(
#    x=forecast_data['date'],
#    y=forecast_data['forecast'],
#    mode='lines',
#    name='Forecast',
#    line=dict(color='#e74c3c', width=3, dash='dash'),
#    hovertemplate='<b>Date:</b> %{x}<br><b>Forecast:</b> %{y:.2f}<extra></extra>'
#))

# Confidence interval
#fig.add_trace(go.Scatter(
#    x=forecast_data['date'].tolist() + forecast_data['date'].tolist()[::-1],
#    y=forecast_data['upper_ci'].tolist() + forecast_data['lower_ci'].tolist()[::-1],
#    fill='toself',
#    fillcolor='rgba(231, 76, 60, 0.2)',
#    line=dict(color='rgba(255,255,255,0)'),
#    name=f'{confidence_level}% Confidence',
#    hoverinfo='skip'
#))

#fig.update_layout(
#    height=500,
#    showlegend=True,
#    plot_bgcolor='white',
#    paper_bgcolor='white',
#    xaxis_title='Date',
#    yaxis_title='Value',
#    hovermode='x unified'
#)

#st.plotly_chart(fig, use_container_width=True)

# Tabs for additional information
tab1, tab2, tab3, tab4 = st.tabs(["📋 Model Performance", "🔍 Feature Analysis", "📈 Comparative View", "💡 Insights"])

with tab1:
    st.markdown("### Model Performance Comparison")
    
    # Metrics table
    metrics_df = pd.DataFrame(metrics[selected_variable]).T
    st.dataframe(metrics_df.style.format("{:.3f}"))#.highlight_max(axis=0, color='#90EE90').highlight_min(axis=0, color='#FFCCCB'))
    
    # Metrics visualization
    fig_metrics = go.Figure()
    models = ['xgb', 'mlp', 'gru']
    
    for metric in ['MAE', 'RMSE']:
        fig_metrics.add_trace(go.Bar(
            name=metric,
            x=models,
            y=[metrics[selected_variable][model][metric] for model in models],
            text=[f'{metrics[selected_variable][model][metric]:.3f}' for model in models],
            textposition='auto',
        ))
    
    fig_metrics.update_layout(barmode='group', title='Model Error Metrics (Lower is Better)')
    st.plotly_chart(fig_metrics, use_container_width=True)

with tab2:
    st.markdown("### Feature Importance Analysis")
    
    # Feature importance bar chart
    features = list(feature_importance[selected_variable].keys())[:10]
    importance_values = list(feature_importance[selected_variable].values())[:10]
    
    fig_features = px.bar(
        x=importance_values,
        y=features,
        orientation='h',
        title='Top Influencing Factors',
        labels={'x': 'Importance Score', 'y': 'Features'}
    )
    fig_features.update_layout(showlegend=False)
    st.plotly_chart(fig_features, use_container_width=True)
    
    # SHAP explanation
    st.markdown("#### How Features Influence Forecast")
    top_features = list(feature_importance[selected_variable].items())[:5]
    
    for feature, importance in top_features:
        effect = "increases" if np.random.random() > 0.3 else "decreases"
        magnitude = f"{importance * 10:.1f}%"
        st.write(f"• **{feature}**: {effect} forecast by ~{magnitude}")

with tab3:
    st.markdown("### Comparative Economic Indicators")
    
    # Correlation heatmap
    corr_matrix = df.drop(columns=['date']).corr()
    fig_corr = px.imshow(
        corr_matrix,
        text_auto=True,
        aspect="auto",
        title='Correlation Between Economic Indicators'
    )
    st.plotly_chart(fig_corr, use_container_width=True)
    
    # Multi-variable trend comparison
    compare_vars = st.multiselect(
        "Compare with:",
        options=[v for v in df.columns if v != selected_variable and v != 'date'],
        default=['Diesel SZL/1 liter', 'Inflation rate']
    )
    
    if compare_vars:
        fig_compare = go.Figure()
        fig_compare.add_trace(go.Scatter(
            x=df['date'], y=df[selected_variable], name=selected_variable, line=dict(width=3)
        ))
        
        for var in compare_vars:
            # Normalize for comparison
            normalized = (df[var] - df[var].mean()) / df[var].std()
            fig_compare.add_trace(go.Scatter(
                x=df['date'], y=normalized, name=var, line=dict(dash='dot')
            ))
        
        fig_compare.update_layout(title='Normalized Comparison with Other Indicators')
        st.plotly_chart(fig_compare, use_container_width=True)

with tab4:
    st.markdown("### Strategic Insights & Recommendations")
    
    # Generate insights based on forecast
    current = df[selected_variable].iloc[-1]
    forecast = forecasts[selected_variable]['forecast'].iloc[0]
    change = ((forecast - current) / current) * 100
    
    col_insight1, col_insight2 = st.columns(2)
    
    with col_insight1:
        st.markdown("#### 📊 Trend Analysis")
        if change > 5:
            st.error("**Warning**: Significant price increase expected")
            st.write("Consider strategic reserves or import planning")
        elif change < -5:
            st.success("**Opportunity**: Price decrease expected")
            st.write("Good time for procurement or inventory building")
        else:
            st.info("**Stable**: Moderate changes expected")
            st.write("Maintain current strategy with close monitoring")
    
    with col_insight2:
        st.markdown("#### ⚡ Key Drivers")
        top_driver = list(feature_importance[selected_variable].keys())[0]
        st.write(f"**Primary driver**: {top_driver}")
        st.write(f"**Volatility**: {volatility:.1f}% (30-day average)")
        st.write(f"**Forecast confidence**: {confidence_level}%")
    
    # Actionable recommendations
    st.markdown("#### 🎯 Recommended Actions")
    
    if "Maize" in selected_variable or "Rice" in selected_variable:
        st.write("""
        - Monitor grain reserves levels
        - Coordinate with agricultural ministry
        - Consider import/export adjustments
        - Update social welfare program calculations
        """)
    elif "Diesel" in selected_variable or "Fuel" in selected_variable:
        st.write("""
        - Review transportation cost projections
        - Assess impact on supply chain
        - Consider fuel subsidy adjustments
        - Monitor global oil price trends
        """)
    elif "Inflation" in selected_variable or "CPI" in selected_variable:
        st.write("""
        - Prepare monetary policy review
        - Update inflation targeting framework
        - Coordinate with central bank
        - Assess impact on interest rates
        """)
    else:
        st.write("""
        - Monitor market trends closely
        - Coordinate with relevant ministry
        - Prepare contingency plans
        - Update budget projections
        """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>🇸🇿 <b>Eswatini Economic Forecasting System</b> | Powered by Machine Learning | INDABA X 2025</p>
    <p><small>Data updated: {}</small></p>
</div>
""".format(datetime.now().strftime("%Y-%m-%d %H:%M")), unsafe_allow_html=True)

# Download section
st.sidebar.markdown("---")
st.sidebar.markdown("### 📥 Export Data")

if st.sidebar.button("Download Current Forecast Report"):
    # Create a simple report
    report_data = {
        'Variable': [selected_variable],
        'Current Value': [current_value],
        '30-Day Forecast': [forecast_value],
        'Change %': [change_pct],
        'Best Model': ['XGBoost'],
        'Model MAE': [metrics[selected_variable]['xgb']['MAE']]
    }
    
    report_df = pd.DataFrame(report_data)
    csv = report_df.to_csv(index=False)
    
    st.sidebar.download_button(
        label="Download CSV Report",
        data=csv,
        file_name=f"eswatini_forecast_{selected_variable.replace(' ', '_')}.csv",
        mime="text/csv"
    )
