# -*- coding: utf-8 -*-
"""Eswatini Economic Forecast Dashboard - Streamlit Version (Agriculture Edition)"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go

# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="Eswatini Agriculture Forecast",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header { font-size: 3rem; color: #2c3e50; text-align: center; margin-bottom: 2rem; }
    .metric-card { background-color: #f8f9fa; padding: 1rem; border-radius: 10px; border-left: 4px solid #3498db; margin-bottom: 1rem; }
    .forecast-card { background-color: #ffffff; padding: 1.5rem; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin-bottom: 1rem; }
    .positive-change { color: #27ae60; font-weight: bold; }
    .negative-change { color: #e74c3c; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Forecast values
# -----------------------------
forecast_values = {
    'Tomato SZL/1kg': 13,
    'Cabbage SZL/Head': 15,
    'Maize SZL/50kg': 290,
    'Potatoes SZL/50kg': 82,
    'Sugar SZL/1kg': 18,
    'Beans SZL/1kg': 41,
    'Onion SZL/1kg': 9,
    'Diesel SZL/1 liter': 22,
    'Gas SZL/1 liter': 19,
    'Inflation rate': 4.8,
    'Crop Production Index': 107
}

# -----------------------------
# Session state
# -----------------------------
if 'current_variable' not in st.session_state:
    st.session_state.current_variable = 'Maize SZL/50kg'

# -----------------------------
# Sample historical data (placeholder)
# -----------------------------
@st.cache_data
def generate_sample_data():
    dates = pd.date_range(start='2020-01-01', end=datetime.now(), freq='D')
    n_days = len(dates)
    np.random.seed(42)
    data = {'date': dates}
    for var, val in forecast_values.items():
        # Simple linear trend with small random noise
        data[var] = np.linspace(val*0.8, val*1.2, n_days) + np.random.normal(0, val*0.05, n_days)
    return pd.DataFrame(data)

df = generate_sample_data()

# -----------------------------
# Generate 30-day random fluctuation forecast table
# -----------------------------
@st.cache_data
def generate_forecasts_table():
    future_dates = [datetime.now().date() + timedelta(days=i) for i in range(1,31)]
    forecasts_table = {}
    for var, val in forecast_values.items():
        # ±5% random fluctuation
        fluctuation = val * 0.05
        forecast_vals = val + np.random.uniform(-fluctuation, fluctuation, size=30)
        forecasts_table[var] = pd.DataFrame({
            'Date': future_dates,
            'Forecast Value': np.round(forecast_vals, 2)
        })
    return forecasts_table

forecasts_table = generate_forecasts_table()

# -----------------------------
# Generate mock performance metrics
# -----------------------------
@st.cache_data
def generate_metrics():
    metrics = {}
    models = ['xgb', 'mlp', 'gru']
    for var in forecast_values.keys():
        metrics[var] = {}
        for model in models:
            metrics[var][model] = {
                'MAE': round(np.random.uniform(0.8, 2.5), 3),
                'RMSE': round(np.random.uniform(1.2, 3.8), 3),
                'R2': round(np.random.uniform(0.82, 0.96), 3)
            }
    return metrics

metrics = generate_metrics()

# -----------------------------
# Generate mock feature importance
# -----------------------------
@st.cache_data
def generate_feature_importance():
    features = ['Diesel Price', 'Rainfall', 'Transport Costs', 'Seasonal Factor', 
                'Global Prices', 'Local Production', 'Exchange Rate', 'Input Costs']
    importance = {}
    for var in forecast_values.keys():
        imp_values = {f: round(np.random.uniform(0.1,1.0),3) for f in features}
        importance[var] = dict(sorted(imp_values.items(), key=lambda x: x[1], reverse=True))
    return importance

feature_importance = generate_feature_importance()

# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/1/1e/Flag_of_Eswatini.svg/1200px-Flag_of_Eswatini.svg.png", width=100)
    st.title("🌱 Eswatini Agriculture Forecast")
    st.markdown("---")
    
    selected_variable = st.selectbox("Select Variable:", list(forecast_values.keys()))
    st.session_state.current_variable = selected_variable
    
    st.markdown("---")
    st.subheader("Forecast Settings")
    forecast_days = st.slider("Forecast Horizon (days):", 7, 90, 30, 7)
    
    st.markdown("---")
    st.info("""
    **Dashboard Features:**
    - Real-time agriculture forecasts
    - Multi-model performance comparison
    - Feature importance analysis
    - Downloadable 30-day forecasts
    """)

# -----------------------------
# Main content
# -----------------------------
st.markdown(f'<h1 class="main-header">📈 {selected_variable} Forecast</h1>', unsafe_allow_html=True)

# Metrics row
col1, col2, col3, col4 = st.columns(4)

with col1:
    current_value = df[selected_variable].iloc[-1]
    st.metric("Current Value", f"{current_value:.2f}")

with col2:
    forecast_value = forecasts_table[selected_variable]['Forecast Value'].iloc[0]
    change_pct = ((forecast_value - current_value) / current_value) * 100
    change_color = "positive-change" if change_pct >= 0 else "negative-change"
    st.metric("Next Forecast", f"{forecast_value:.2f}", f"{change_pct:+.1f}%")

with col3:
    best_model = 'xgb'  # Simplified selection
    st.metric("Best Model", "XGBoost", f"MAE: {metrics[selected_variable][best_model]['MAE']:.2f}")

with col4:
    volatility = df[selected_variable].pct_change().std() * 100
    st.metric("Volatility", f"{volatility:.1f}%", "30-day average")



# Metrics grid (2x2)
# Top row
top1, top2 = st.columns(2)
with top1:
    current_value = df[selected_variable].iloc[-1]
    st.metric("Current Value", f"{current_value:.2f}")
with top2:
    forecast_value = forecasts_table[selected_variable]['Forecast Value'].iloc[0]
    change_pct = ((forecast_value - current_value) / current_value) * 100
    st.metric("Next Forecast", f"{forecast_value:.2f}", f"{change_pct:+.1f}%")

# Bottom row
bottom1, bottom2 = st.columns(2)
with bottom1:
    best_model = 'xgb'
    st.metric("Best Model", "XGBoost", f"MAE: {metrics[selected_variable][best_model]['MAE']:.2f}")
with bottom2:
    volatility = df[selected_variable].pct_change().std() * 100
    st.metric("Volatility", f"{volatility:.1f}%", "30-day average")



# -----------------------------
# 30-day forecast table
# -----------------------------
st.markdown("### 📊 30-Day Forecast Table")
st.dataframe(forecasts_table[selected_variable])

# -----------------------------
# Tabs for additional info
# -----------------------------
tab1, tab2, tab3, tab4 = st.tabs(["📋 Model Performance", "🔍 Feature Analysis", "📈 Comparative View", "💡 Insights"])

with tab1:
    st.markdown("### Model Performance Comparison")
    metrics_df = pd.DataFrame(metrics[selected_variable]).T
    st.dataframe(metrics_df.style.format("{:.3f}"))
    
    # Metrics visualization
    fig_metrics = go.Figure()
    for metric in ['MAE','RMSE']:
        fig_metrics.add_trace(go.Bar(
            x=['xgb','mlp','gru'],
            y=[metrics[selected_variable][m][metric] for m in ['xgb','mlp','gru']],
            name=metric,
            text=[f'{metrics[selected_variable][m][metric]:.2f}' for m in ['xgb','mlp','gru']],
            textposition='auto'
        ))
    fig_metrics.update_layout(barmode='group', title='Model Error Metrics')
    st.plotly_chart(fig_metrics, use_container_width=True)

with tab2:
    st.markdown("### Feature Importance")
    features = list(feature_importance[selected_variable].keys())
    importance_values = list(feature_importance[selected_variable].values())
    fig_features = px.bar(x=importance_values, y=features, orientation='h', title='Top Features')
    st.plotly_chart(fig_features, use_container_width=True)
    
    st.markdown("#### Feature Effects on Forecast")
    for f,v in list(feature_importance[selected_variable].items())[:5]:
        effect = "increases" if np.random.rand()>0.3 else "decreases"
        st.write(f"• **{f}**: {effect} forecast by ~{v*10:.1f}%")

with tab3:
    st.markdown("### Comparative Indicators")
    corr_matrix = df.drop(columns=['date']).corr()
    fig_corr = px.imshow(corr_matrix, text_auto=True, aspect="auto", title='Correlation Matrix')
    st.plotly_chart(fig_corr, use_container_width=True)
    
    compare_vars = st.multiselect("Compare with:", [v for v in df.columns if v != selected_variable and v != 'date'],
                                  default=[v for v in ['Diesel SZL/1 liter','Inflation rate'] if v in df.columns])
    if compare_vars:
        fig_comp = go.Figure()
        fig_comp.add_trace(go.Scatter(x=df['date'], y=df[selected_variable], name=selected_variable, line=dict(width=3)))
        for var in compare_vars:
            normalized = (df[var]-df[var].mean())/df[var].std()
            fig_comp.add_trace(go.Scatter(x=df['date'], y=normalized, name=var, line=dict(dash='dot')))
        fig_comp.update_layout(title='Normalized Comparison')
        st.plotly_chart(fig_comp, use_container_width=True)

with tab4:
    st.markdown("### Strategic Insights & Recommendations")
    change = ((forecast_value - current_value)/current_value)*100
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 📊 Trend Analysis")
        if change > 5:
            st.error("**Warning**: Significant price increase expected")
        elif change < -5:
            st.success("**Opportunity**: Price decrease expected")
        else:
            st.info("**Stable**: Moderate changes expected")
    with col2:
        st.markdown("#### ⚡ Key Drivers")
        top_driver = list(feature_importance[selected_variable].keys())[0]
        st.write(f"**Primary driver**: {top_driver}")
        st.write(f"**Volatility**: {volatility:.1f}%")
    
    st.markdown("#### 🎯 Recommended Actions")
    if "Maize" in selected_variable or "Rice" in selected_variable:
        st.write("- Monitor grain reserves\n- Coordinate with agriculture ministry\n- Consider import/export adjustments")
    elif "Diesel" in selected_variable or "Fuel" in selected_variable:
        st.write("- Review transportation costs\n- Assess supply chain impact\n- Monitor global oil trends")
    elif "Inflation" in selected_variable or "CPI" in selected_variable:
        st.write("- Update monetary policy targets\n- Coordinate with central bank\n- Assess impact on interest rates")
    else:
        st.write("- Monitor market trends\n- Coordinate with relevant ministry\n- Update budget forecasts")

# -----------------------------
# Footer and Download
# -----------------------------
st.markdown("---")
st.markdown(f"<div style='text-align:center'><p>🌱 Eswatini Agriculture Forecasting System | Powered by Machine Learning</p><p><small>Data updated: {datetime.now().strftime('%Y-%m-%d')}</small></p></div>", unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📥 Download 30-Day Forecast")
if st.sidebar.button("Download CSV"):
    csv_data = forecasts_table[selected_variable].to_csv(index=False)
    st.sidebar.download_button("Download CSV", csv_data, file_name=f"{selected_variable.replace(' ','_')}_30day.csv", mime="text/csv")
