# -*- coding: utf-8 -*-
"""Eswatini Economic Forecast Dashboard - Streamlit Version (Agriculture Edition) with AI Chatbot"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
import os
import openai

# -----------------------------
# OpenAI API Key setup (safe)
# -----------------------------
openai.api_key = os.environ.get("OPENAI_API_KEY") or st.secrets.get("openai", {}).get("api_key")
if not openai.api_key:
    st.warning("⚠️ OpenAI API key not found. AI assistant will not work until key is provided.")

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
# Session state initialization
# -----------------------------
if 'current_variable' not in st.session_state:
    st.session_state.current_variable = 'Maize SZL/50kg'

if 'chat_messages' not in st.session_state:
    st.session_state.chat_messages = [
        {"role": "assistant", "content": "👋 Hi! I'm your AI Agriculture Assistant. Ask me about forecasts, trends, or market insights."}
    ]

if 'chat_open' not in st.session_state:
    st.session_state.chat_open = False

# -----------------------------
# Utility Functions
# -----------------------------
def get_ai_response(user_message, context_data=None):
    """
    Generate a simple AI response based on user message.
    Replace with OpenAI API call for real responses.
    """
    user_message = user_message.lower()
    current_var = context_data.get('current_variable', 'Unknown') if context_data else 'Unknown'
    forecast_val = context_data.get('forecast_value', 0) if context_data else 0
    current_val = context_data.get('current_value', 0) if context_data else 0

    if any(word in user_message for word in ['maize', 'corn', 'crop']):
        trend = "upward" if forecast_val > current_val else "downward"
        return f"🌽 Maize prices are showing a {trend} trend. Monitor seasonal and weather factors for better accuracy."

    elif any(word in user_message for word in ['price', 'forecast', 'prediction']):
        change = ((forecast_val - current_val) / current_val * 100) if current_val != 0 else 0
        return f"📈 For {current_var}, expected change: {change:+.1f}%."

    elif any(word in user_message for word in ['model', 'accuracy', 'performance']):
        return "🤖 Forecasting models used: XGBoost, MLP, GRU. XGBoost is generally most accurate."

    elif any(word in user_message for word in ['weather', 'climate', 'season']):
        return "🌦️ Weather is crucial. Track rainfall and temperature for crops like maize and vegetables."

    elif any(word in user_message for word in ['export', 'import', 'trade']):
        return "🚢 Trade impacts local prices. SACU and South African market trends are important."

    elif any(word in user_message for word in ['help', 'how', 'explain']):
        return """🎯 I can help you with:
• Price forecasts
• Model explanations
• Agricultural insights
• Feature importance analysis
• Market recommendations"""

    else:
        return "🤖 I'm your AI agriculture analyst! Ask about price forecasts, trends, or market insights."

def create_sample_data():
    dates = pd.date_range(start='2020-01-01', end=datetime.now(), freq='D')
    n_days = len(dates)
    np.random.seed(42)
    data = {'date': dates}
    for var, val in forecast_values.items():
        data[var] = np.linspace(val*0.8, val*1.2, n_days) + np.random.normal(0, val*0.05, n_days)
    return pd.DataFrame(data)

def create_forecasts_table():
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

def create_metrics():
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

def create_feature_importance():
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

# -----------------------------
# Custom CSS
# -----------------------------
st.markdown("""
<style>
.main-header { font-size: 3rem; color: #2c3e50; text-align: center; margin-bottom: 2rem; }
.kpi-container { display: grid; grid-template-columns: 1fr 1fr 1fr 1fr; gap: 15px; margin-bottom: 20px; }
.kpi-card { background: #f2f2f2; padding: 20px; border-radius: 12px; text-align:center; box-shadow:2px 2px 6px rgba(0,0,0,0.1); }
.kpi-value { font-size: 24px; font-weight: bold; }
.sticky-header { position: sticky; top:0; background:#f8f9fa; font-size:32px; font-weight:bold; text-align:center; padding:10px; border-bottom:2px solid #3498db; z-index:9999; }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Generate data
# -----------------------------
df = create_sample_data()
forecasts_table = create_forecasts_table()
metrics = create_metrics()
feature_importance = create_feature_importance()

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
    st.info("**Dashboard Features:**\n- Real-time agriculture forecasts\n- Multi-model performance comparison\n- Feature importance analysis\n- Downloadable 30-day forecasts\n- AI Assistant")

# -----------------------------
# Chatbot Interface in Sidebar
# -----------------------------
st.sidebar.markdown("---")
st.sidebar.markdown("### 🤖 AI Assistant")

col1, col2 = st.sidebar.columns([3, 1])
with col1:
    if st.button("💬 Toggle Chat", key="chat_toggle"):
        st.session_state.chat_open = not st.session_state.chat_open
with col2:
    if st.button("🗑️ Clear", key="clear_chat"):
        st.session_state.chat_messages = [{"role": "assistant", "content": "👋 Chat cleared! How can I help you?"}]

if st.session_state.chat_open:
    st.sidebar.markdown("**Chat with AI Assistant:**")
    
    chat_container = st.sidebar.container()
    with chat_container:
        for message in st.session_state.chat_messages[-5:]:
            if message["role"] == "user":
                st.sidebar.markdown(f"**You:** {message['content']}")
            else:
                st.sidebar.markdown(f"**🤖 AI:** {message['content']}")
    
    user_input = st.sidebar.text_input("Ask me anything:", key="chat_input", placeholder="e.g., What will maize prices be next week?")
    send_clicked = st.sidebar.button("Send 📤", key="send_message")
    
    if send_clicked and user_input:
        st.session_state.chat_messages.append({"role": "user", "content": user_input})

        current_value = df[selected_variable].iloc[-1]
        forecast_value = forecasts_table[selected_variable]['Forecast Value'].iloc[0]
        context = {
            'current_variable': selected_variable,
            'current_value': current_value,
            'forecast_value': forecast_value
        }

        ai_response = get_ai_response(user_input, context)
        st.session_state.chat_messages.append({"role": "assistant", "content": ai_response})
        st.session_state.chat_input = ""  # clear input

# -----------------------------
# Main content
# -----------------------------

st.markdown(
    f'<div class="sticky-header">🇸🇿 Eswatini AgriForecast Hub 🌾</div>',
    unsafe_allow_html=True
)

st.markdown(f'<h2 class="main-header">📈 {selected_variable} Forecast</h2>', unsafe_allow_html=True)

current_value = df[selected_variable].iloc[-1]
forecast_value = forecasts_table[selected_variable]['Forecast Value'].iloc[0]
change_pct = ((forecast_value - current_value) / current_value) * 100
volatility = df[selected_variable].pct_change().std() * 100

# KPI Cards
st.markdown(
    f"""
    <div class="kpi-container">
        <div class="kpi-card">
            <div class="kpi-value">{current_value:.2f}</div>
            <div>Current Value</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-value">{forecast_value:.2f} ({change_pct:+.1f}%)</div>
            <div>Next Forecast</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-value">XGBoost<br><span style="font-size:16px;">MAE: {metrics[selected_variable]['xgb']['MAE']:.2f}</span></div>
            <div>Best Model</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-value">{volatility:.1f}%</div>
            <div>Volatility (30-day avg)</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

st.info("💡 Use the AI Assistant in the sidebar to get personalized insights about this forecast!")

# 30-Day Forecast Table
st.markdown("### 📊 30-Day Forecast Table")
st.dataframe(forecasts_table[selected_variable])

# Tabs: Model, Feature, Insights
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
        relevant_features = [
            'Tomato SZL/1kg', 'Onion SZL/1kg', 'Rice SZL/1kg', 'Gas SZL/1 liter', 
            'Beans SZL/1kg', 'Cabbage SZL/Head', 'Diesel SZL/1 liter', 'Maize SZL/50kg', 
            'Brown Bread SZL', 'Sugar SZL/1kg', 'Potatoes SZL/50kg', 'Inflation rate'
        ]
        top_driver_candidates = [f for f in feature_importance[selected_variable].keys() if f in relevant_features]
        top_drivers = top_driver_candidates[:3] if top_driver_candidates else list(feature_importance[selected_variable].keys())[:3]
        for i, driver in enumerate(top_drivers, 1):
            st.write(f"**Primary driver {i}**: {driver}")
        st.write(f"**Volatility**: {volatility:.1f}%")

# Recommendations
st.markdown("#### 🎯 Recommended Actions")
recommendations = []
if "Maize" in selected_variable or "Rice" in selected_variable:
    recommendations = [
        "Monitor grain reserves",
        "Coordinate with agriculture ministry",
        "Consider import/export adjustments",
        "Assess storage and distribution capacity",
        "Review price stabilization policies"
    ]
elif "Diesel" in selected_variable or "Fuel" in selected_variable:
    recommendations = [
        "Review transportation costs",
        "Assess supply chain impact",
        "Monitor global oil trends",
        "Optimize fuel consumption",
        "Plan for potential price shocks"
    ]
elif "Inflation" in selected_variable or "CPI" in selected_variable:
    recommendations = [
        "Update monetary policy targets",
        "Coordinate with central bank",
        "Assess impact on interest rates",
        "Monitor consumer price trends",
        "Adjust budget forecasts"
    ]
else:
    recommendations = [
        "Monitor market trends",
        "Coordinate with relevant ministry",
        "Update budget forecasts",
        "Track production levels",
        "Review seasonal planning strategies"
    ]
for rec in recommendations[:5]:
    st.write(f"- {rec}")

st.markdown("#### 🎯 AI Assistant Recommendations")
st.info("💡 Ask the AI: 'What factors are driving these changes?' or 'What should I watch for next week?'")

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

# Download CSV
st.sidebar.markdown("---")
st.sidebar.markdown("### 📥 Download 30-Day Forecast")
csv_data = forecasts_table[selected_variable].to_csv(index=False)
st.sidebar.download_button(
    "📄 Download CSV", 
    csv_data, 
    file_name=f"{selected_variable.replace(' ','_').replace('/','_')}_30day.csv", 
    mime="text/csv"
)
