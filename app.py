# -*- coding: utf-8 -*-
"""Eswatini Economic Forecast Dashboard - Streamlit Version (Agriculture Edition) with AI Chatbot"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
import time

# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="Eswatini Agriculture Forecast HUB",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------
# Chatbot Functions
# -----------------------------
def get_ai_response(user_message, context_data=None):
    """
    Simulate AI response based on user message and context
    In production, this would connect to OpenAI API, Claude API, etc.
    """
    user_message = user_message.lower()
    
    # Context-aware responses based on current page data
    if context_data:
        current_var = context_data.get('current_variable', 'Unknown')
        forecast_val = context_data.get('forecast_value', 0)
        current_val = context_data.get('current_value', 0)
    
    # Agriculture-specific responses
    if any(word in user_message for word in ['maize', 'corn', 'crop']):
        return f"🌽 Based on current data, maize prices show {'upward' if forecast_val > current_val else 'downward'} trends. Our models suggest monitoring weather patterns and seasonal factors for better accuracy."
    
    elif any(word in user_message for word in ['price', 'forecast', 'prediction']):
        if context_data:
            change = ((forecast_val - current_val) / current_val * 100) if current_val != 0 else 0
            return f"📈 For {current_var}, I'm forecasting a {change:+.1f}% change. This is based on our XGBoost model with 92% accuracy. Key factors include seasonal trends and market dynamics."
        return "📊 I can help you understand price forecasts! Please select a specific variable from the sidebar to get detailed predictions."
    
    elif any(word in user_message for word in ['model', 'accuracy', 'performance']):
        return "🤖 Our forecasting uses three models: XGBoost (best overall), MLP Neural Networks, and GRU. XGBoost typically achieves 90-96% accuracy across different agricultural commodities."
    
    elif any(word in user_message for word in ['weather', 'climate', 'season']):
        return "🌦️ Weather is a crucial factor! Our models incorporate seasonal patterns, and I recommend monitoring rainfall and temperature data for crops like maize and vegetables."
    
    elif any(word in user_message for word in ['export', 'import', 'trade']):
        return "🚢 Trade flows significantly impact local prices. Monitor SACU trade data and South African market trends, as they strongly influence Eswatini's agricultural markets."
    
    elif any(word in user_message for word in ['help', 'how', 'explain']):
        return """🎯 I can help you with:
• Price forecasts and trends
• Model explanations and accuracy
• Agricultural insights for Eswatini
• Feature importance analysis
• Market recommendations
        
Just ask me about any specific crop, price, or analysis!"""
    
    else:
        return "🤖 I'm your AI agriculture analyst for Eswatini! Ask me about price forecasts, crop trends, model performance, or market insights. How can I help you today?"

# Initialize chatbot session state
if 'chat_messages' not in st.session_state:
    st.session_state.chat_messages = [
        {"role": "assistant", "content": "👋 Hi! I'm your AI Agriculture Assistant. I can help you understand forecasts, explain trends, and provide insights about Eswatini's agricultural markets. What would you like to know?"}
    ]
if 'chat_open' not in st.session_state:
    st.session_state.chat_open = False

# Custom CSS for floating chatbot
st.markdown("""
<style>
    .main-header { font-size: 3rem; color: #2c3e50; text-align: center; margin-bottom: 2rem; }
    .metric-card { background-color: #f8f9fa; padding: 1rem; border-radius: 10px; border-left: 4px solid #3498db; margin-bottom: 1rem; }
    .forecast-card { background-color: #ffffff; padding: 1.5rem; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin-bottom: 1rem; }
    .positive-change { color: #27ae60; font-weight: bold; }
    .negative-change { color: #e74c3c; font-weight: bold; }
    
    /* Floating Chatbot Button */
    .chatbot-button {
        position: fixed;
        bottom: 20px;
        right: 20px;
        z-index: 1000;
        background: linear-gradient(45deg, #3498db, #2ecc71);
        color: white;
        border: none;
        border-radius: 50%;
        width: 60px;
        height: 60px;
        font-size: 24px;
        cursor: pointer;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        transition: all 0.3s ease;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    
    .chatbot-button:hover {
        transform: scale(1.1);
        box-shadow: 0 6px 16px rgba(0,0,0,0.4);
    }
    
    .chat-container {
        position: fixed;
        bottom: 90px;
        right: 20px;
        width: 350px;
        height: 400px;
        background: white;
        border-radius: 15px;
        box-shadow: 0 8px 25px rgba(0,0,0,0.2);
        z-index: 999;
        display: flex;
        flex-direction: column;
        overflow: hidden;
    }
    
    .chat-header {
        background: linear-gradient(45deg, #3498db, #2ecc71);
        color: white;
        padding: 12px 16px;
        font-weight: bold;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    
    .chat-messages {
        flex: 1;
        overflow-y: auto;
        padding: 10px;
        background: #f8f9fa;
    }
    
    .message {
        margin-bottom: 10px;
        padding: 8px 12px;
        border-radius: 15px;
        max-width: 80%;
        word-wrap: break-word;
    }
    
    .user-message {
        background: #3498db;
        color: white;
        margin-left: auto;
        text-align: right;
    }
    
    .assistant-message {
        background: white;
        color: #333;
        border: 1px solid #e0e0e0;
    }
    
    .chat-input-container {
        padding: 10px;
        background: white;
        border-top: 1px solid #e0e0e0;
    }
    
    .kpi-container {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 20px;
    }
    .kpi-card {
        background: #f2f2f2;
        padding: 25px;
        border-radius: 15px;
        text-align: center;
        box-shadow: 2px 2px 8px rgba(0,0,0,0.15);
    }
    .kpi-value {
        font-size: 28px;
        font-weight: bold;
        color: #222222;
    }
    .kpi-label {
        color: #555555;
        font-size: 16px;
        font-weight: 500;
        margin-bottom: 8px;
    }
    
    .sticky-header {
        position: -webkit-sticky;
        position: sticky;
        top: 0;
        background-color: #f8f9fa;
        color: #2c3e50;
        font-size: 32px;
        font-weight: bold;
        text-align: center;
        padding: 10px 0;
        border-bottom: 2px solid #3498db;
        z-index: 9999;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Forecast values
# -----------------------------
forecast_values = {
    'Potatoes SZL/50kg': 82,
    'Beans SZL/1kg': 41,
    'Onion SZL/1kg': 9,
    'Diesel SZL/1 liter': 22,
    'Tomato SZL/1kg': 13,
    'Maize SZL/50kg': 290,
    'Sugar SZL/1kg': 18,
    'Cabbage SZL/Head': 15,
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
        data[var] = np.linspace(val*0.8, val*1.2, n_days) + np.random.normal(0, val*0.05, n_days)
    return pd.DataFrame(data)

df = generate_sample_data()

# -----------------------------
# Generate forecasts, metrics, and feature importance (same as before)
# -----------------------------
@st.cache_data
def generate_forecasts_table():
    future_dates = [datetime.now().date() + timedelta(days=i) for i in range(1,31)]
    forecasts_table = {}
    for var, val in forecast_values.items():
        fluctuation = val * 0.05
        forecast_vals = val + np.random.uniform(-fluctuation, fluctuation, size=30)
        forecasts_table[var] = pd.DataFrame({
            'Date': future_dates,
            'Forecast Value': np.round(forecast_vals, 2)
        })
    return forecasts_table

forecasts_table = generate_forecasts_table()

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

@st.cache_data
def generate_feature_importance():
    features = ['Onion SZL/1kg', 'Tomato (Round) SZL/1kg', 'Rice SZL/1kg', 'Gas SZL/1 liter',
                'Beans SZL/1kg', 'Cabbage SZL/Head', 'Diesel SZL/1 liter', 'Maize SZL/50kg',
                'Brown Bread SZL', 'Sugar SZL/1kg', 'Potatoes SZL/50kg', 'Maize meal SZL/1kg',
                'Money supply SZL', 'GDP by economic activity (Current Prices)', 
                'Interest Rate (Prime lending rate)', 'All Items CPI', 'Inflation rate']
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
    st.title("🌱 Eswatini Agriculture Forecast Hub")
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
    - **AI Assistant** (click chat button!)
    """)

# -----------------------------
# Chatbot Interface in Sidebar
# -----------------------------
st.sidebar.markdown("---")
st.sidebar.markdown("### 🤖 AI Assistant")

# Chat toggle
if st.sidebar.button("💬 Open Chat Assistant"):
    st.session_state.chat_open = not st.session_state.chat_open

if st.session_state.chat_open:
    st.sidebar.markdown("**Chat with AI Assistant:**")
    
    # Display chat messages
    chat_container = st.sidebar.container()
    with chat_container:
        for message in st.session_state.chat_messages[-5:]:  # Show last 5 messages
            if message["role"] == "user":
                st.markdown(f"**You:** {message['content']}")
            else:
                st.markdown(f"**AI:** {message['content']}")
    
    # Chat input
    user_input = st.sidebar.text_input("Ask me anything about the forecasts:", key="chat_input")
    
    if st.sidebar.button("Send") and user_input:
        # Add user message
        st.session_state.chat_messages.append({"role": "user", "content": user_input})
        
        # Prepare context for AI
        current_value = df[selected_variable].iloc[-1]
        forecast_value = forecasts_table[selected_variable]['Forecast Value'].iloc[0]
        context = {
            'current_variable': selected_variable,
            'current_value': current_value,
            'forecast_value': forecast_value
        }
        
        # Get AI response
        ai_response = get_ai_response(user_input, context)
        st.session_state.chat_messages.append({"role": "assistant", "content": ai_response})
        
        # Rerun to update display
        st.rerun()

# -----------------------------
# Main content (same as before)
# -----------------------------
st.markdown(
    """
    <div class="sticky-header">🇸🇿Eswatini AgriForecast Hub🌾</div>
    """,
    unsafe_allow_html=True
)

st.markdown(f'<h2 class="main-header">📈 {selected_variable} Forecast</h2>', unsafe_allow_html=True)

# Get values for display
current_value = df[selected_variable].iloc[-1]
forecast_value = forecasts_table[selected_variable]['Forecast Value'].iloc[0]
change_pct = ((forecast_value - current_value) / current_value) * 100
volatility = df[selected_variable].pct_change().std() * 100

st.markdown(
    f"""
    <div class="kpi-container">
        <div class="kpi-card">
            <div class="kpi-label">Current Value</div>
            <div class="kpi-value">{current_value:.2f}</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Next Forecast</div>
            <div class="kpi-value">{forecast_value:.2f} ({change_pct:+.1f}%)</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Best Model</div>
            <div class="kpi-value">XGBoost<br><span style="font-size:18px;">MAE: {metrics[selected_variable]['xgb']['MAE']:.2f}</span></div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Volatility (30-day avg)</div>
            <div class="kpi-value">{volatility:.1f}%</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# 30-day forecast table
st.markdown("### 📊 30-Day Forecast Table")
st.dataframe(forecasts_table[selected_variable])

# Tabs for additional info
tab1, tab2, tab3 = st.tabs(["📋 Model Performance", "🔍 Feature Analysis", "💡 Insights"])

with tab1:
    st.markdown("### Model Performance Comparison")
    metrics_df = pd.DataFrame(metrics[selected_variable]).T
    st.dataframe(metrics_df.style.format("{:.3f}"))
    
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
    relevant_features = [
        'Tomato SZL/1kg', 'Onion SZL/1kg', 'Rice SZL/1kg', 'Gas SZL/1 liter', 
        'Beans SZL/1kg', 'Cabbage SZL/Head', 'Diesel SZL/1 liter', 'Maize SZL/50kg', 
        'Brown Bread SZL', 'Sugar SZL/1kg', 'Potatoes SZL/50kg', 'Inflation rate'
    ]
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
        top_driver_candidates = [f for f in feature_importance[selected_variable].keys() if f in relevant_features]
        top_drivers = top_driver_candidates[:3] if top_driver_candidates else list(feature_importance[selected_variable].keys())[:3]
        for i, driver in enumerate(top_drivers, 1):
            st.write(f"**Primary driver {i}**: {driver}")
        st.write(f"**Volatility**: {volatility:.1f}%")
    
    st.markdown("#### 🎯 AI Assistant Recommendations")
    st.info("💡 **Tip**: Use the AI Assistant in the sidebar to get personalized insights about this forecast!")

# Footer
st.markdown("---")
st.markdown(
    f"""
    <div style='text-align:center; font-size: 16px;'>
        🌾 <b>Eswatini Agriculture Forecasting System</b> | Powered by Machine Learning & AI<br>
        <small>Data updated: {datetime.now().strftime('%Y-%m-%d')} | AI Assistant available 24/7</small>
    </div>
    """,
    unsafe_allow_html=True
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📥 Download 30-Day Forecast")
if st.sidebar.button("Download CSV"):
    csv_data = forecasts_table[selected_variable].to_csv(index=False)
    st.sidebar.download_button("Download CSV", csv_data, file_name=f"{selected_variable.replace(' ','_')}_30day.csv", mime="text/csv")

# Add floating chatbot button using HTML/JS (alternative approach)
st.markdown("""
<div style="position: fixed; bottom: 20px; right: 20px; z-index: 1000;">
    <div style="background: linear-gradient(45deg, #3498db, #2ecc71); color: white; border-radius: 50%; width: 60px; height: 60px; display: flex; align-items: center; justify-content: center; font-size: 24px; box-shadow: 0 4px 12px rgba(0,0,0,0.3); cursor: pointer;" 
         title="Click to open AI Assistant in sidebar!">
        🤖
    </div>
</div>
""", unsafe_allow_html=True)
