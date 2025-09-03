# -*- coding: utf-8 -*-
"""Eswatini Economic Forecast Dashboard"""

import dash
from dash import dcc, html, Input, Output, callback
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import json
import base64
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Initialize the Dash App
app = dash.Dash(__name__)
app.title = "Eswatini Economic Forecast Dashboard"

# Sample data for demonstration (will be replaced with your actual data)
def create_demo_data():
    """Create demo time series data"""
    dates = pd.date_range(start='2020-01-01', end='2024-12-31', freq='D')
    n_days = len(dates)
    
    # Create synthetic trends for demonstration
    np.random.seed(42)
    base_trend = np.linspace(10, 50, n_days)
    seasonal = 5 * np.sin(2 * np.pi * np.arange(n_days) / 365)
    noise = np.random.normal(0, 2, n_days)
    
    demo_data = {
        'date': dates,
        'Maize_meal': base_trend + seasonal + noise,
        'CPI': np.linspace(100, 145, n_days) + np.random.normal(0, 1, n_days),
        'Diesel': np.linspace(15, 35, n_days) + np.random.normal(0, 1.5, n_days),
        'Inflation': np.random.normal(5, 1, n_days).cumsum() / 10 + 2,
    }
    
    return pd.DataFrame(demo_data)

# Create demo artifacts for each variable
def create_demo_artifacts():
    """Create demo model artifacts"""
    variables = ['Maize_meal', 'CPI', 'Diesel', 'Inflation']
    features = ['Fuel_Price', 'Exchange_Rate', 'Rainfall', 'GDP_Growth', 'Import_Costs']
    
    for variable in variables:
        os.makedirs(f'model_artifacts/{variable}', exist_ok=True)
        
        # Demo metrics
        metrics = {
            'xgb': {'MAE': np.random.uniform(0.5, 2.0), 'RMSE': np.random.uniform(0.8, 3.0), 'R2': np.random.uniform(0.85, 0.95)},
            'mlp': {'MAE': np.random.uniform(0.6, 2.2), 'RMSE': np.random.uniform(1.0, 3.5), 'R2': np.random.uniform(0.8, 0.92)},
            'gru': {'MAE': np.random.uniform(0.7, 2.5), 'RMSE': np.random.uniform(1.2, 4.0), 'R2': np.random.uniform(0.75, 0.9)}
        }
        
        with open(f'model_artifacts/{variable}/metrics.json', 'w') as f:
            json.dump(metrics, f, indent=4)
        
        # Demo forecast
        future_dates = [datetime.now().date() + timedelta(days=i) for i in range(1, 31)]
        current_value = np.random.uniform(20, 40)
        trend = np.random.uniform(-0.2, 0.3, 30).cumsum()
        forecast_values = current_value + trend
        
        forecast_df = pd.DataFrame({
            'date': future_dates,
            'forecast': forecast_values
        })
        forecast_df.to_csv(f'model_artifacts/{variable}/forecast.csv', index=False)
        
        # Demo feature importance
        feature_importance = {feature: np.random.uniform(0.1, 1.0) for feature in features}
        with open(f'model_artifacts/{variable}/selected_features.json', 'w') as f:
            json.dump(feature_importance, f, indent=4)

# Create demo data and artifacts
df = create_demo_data()
create_demo_artifacts()

# Available variables for dropdown
VARIABLE_OPTIONS = [
    {'label': 'Maize Meal (SZL/kg)', 'value': 'Maize_meal'},
    {'label': 'Consumer Price Index', 'value': 'CPI'},
    {'label': 'Diesel Price (SZL/liter)', 'value': 'Diesel'},
    {'label': 'Inflation Rate (%)', 'value': 'Inflation'}
]

# App Layout
app.layout = html.Div([
    # Header
    html.Div([
        html.H1("🇸🇿 Eswatini Economic Forecast Dashboard", 
                style={'textAlign': 'center', 'color': 'white', 'marginBottom': '20px'}),
        html.P("Real-time forecasting of key economic indicators for strategic decision-making",
              style={'textAlign': 'center', 'color': 'white', 'fontSize': '16px'})
    ], style={'backgroundColor': '#2c3e50', 'padding': '20px', 'borderRadius': '10px'}),
    
    html.Br(),
    
    # Main content
    html.Div([
        # Left panel - Controls
        html.Div([
            html.H3("📊 Control Panel", style={'color': '#2c3e50'}),
            html.Hr(),
            
            html.Label("Select Economic Indicator:"),
            dcc.Dropdown(
                id='variable-selector',
                options=VARIABLE_OPTIONS,
                value='Maize_meal',
                clearable=False,
                style={'marginBottom': '20px'}
            ),
            
            html.Label("Forecast Horizon:"),
            dcc.Slider(
                id='forecast-horizon',
                min=7,
                max=90,
                step=7,
                value=30,
                marks={7: '1W', 30: '1M', 60: '2M', 90: '3M'},
                style={'marginBottom': '20px'}
            ),
            
            html.Label("Confidence Interval:"),
            dcc.Slider(
                id='confidence-level',
                min=80,
                max=95,
                step=5,
                value=90,
                marks={80: '80%', 85: '85%', 90: '90%', 95: '95%'},
                style={'marginBottom': '20px'}
            ),
            
            html.Button('🔄 Update Forecast', id='update-button', n_clicks=0,
                       style={'width': '100%', 'padding': '10px', 'backgroundColor': '#27ae60', 
                              'color': 'white', 'border': 'none', 'borderRadius': '5px',
                              'cursor': 'pointer'}),
            
            html.Hr(),
            
            # Key Metrics Display
            html.Div(id='key-metrics', style={'padding': '10px', 'backgroundColor': '#f8f9fa', 
                                            'borderRadius': '5px', 'marginTop': '20px'})
            
        ], style={'width': '25%', 'padding': '20px', 'backgroundColor': 'white', 
                 'borderRadius': '10px', 'boxShadow': '0 4px 6px rgba(0,0,0,0.1)'}),
        
        # Right panel - Visualizations
        html.Div([
            # Forecast Plot
            dcc.Graph(id='forecast-plot', style={'height': '400px', 'marginBottom': '20px'}),
            
            # Tabs for additional information
            dcc.Tabs([
                # Model Performance Tab
                dcc.Tab(label='📈 Model Performance', children=[
                    html.Div([
                        html.H4("Model Evaluation Metrics"),
                        html.Div(id='metrics-table', style={'marginBottom': '20px'}),
                        
                        html.H4("Actual vs Predicted Values"),
                        dcc.Graph(id='actual-vs-predicted-plot')
                    ], style={'padding': '20px'})
                ]),
                
                # Feature Importance Tab
                dcc.Tab(label='🔍 Feature Importance', children=[
                    html.Div([
                        html.H4("Top Influencing Factors"),
                        dcc.Graph(id='feature-importance-plot'),
                        
                        html.H4("How Features Affect the Forecast"),
                        html.Div(id='shap-explanation')
                    ], style={'padding': '20px'})
                ]),
                
                # Data Insights Tab
                dcc.Tab(label='📋 Data Insights', children=[
                    html.Div([
                        html.H4("Historical Trends"),
                        dcc.Graph(id='historical-trends-plot'),
                        
                        html.H4("Correlation Heatmap"),
                        dcc.Graph(id='correlation-heatmap')
                    ], style={'padding': '20px'})
                ])
            ])
        ], style={'width': '75%', 'padding': '0 20px'})
    ], style={'display': 'flex', 'gap': '20px'})
], style={'padding': '20px', 'backgroundColor': '#ecf0f1', 'minHeight': '100vh'})

# Callbacks for interactive components
@callback(
    [Output('forecast-plot', 'figure'),
     Output('key-metrics', 'children'),
     Output('metrics-table', 'children'),
     Output('feature-importance-plot', 'figure'),
     Output('actual-vs-predicted-plot', 'figure'),
     Output('historical-trends-plot', 'figure'),
     Output('correlation-heatmap', 'figure'),
     Output('shap-explanation', 'children')],
    [Input('variable-selector', 'value'),
     Input('update-button', 'n_clicks')],
    prevent_initial_call=False
)
def update_dashboard(selected_variable, n_clicks):
    """Update all dashboard components based on selected variable"""
    
    # Filter data for selected variable
    variable_data = df[['date', selected_variable]].copy()
    variable_data = variable_data.rename(columns={selected_variable: 'value'})
    
    # Load forecast data (in real app, this would come from your model artifacts)
    try:
        forecast_df = pd.read_csv(f'model_artifacts/{selected_variable}/forecast.csv', parse_dates=['date'])
        with open(f'model_artifacts/{selected_variable}/metrics.json', 'r') as f:
            metrics = json.load(f)
        with open(f'model_artifacts/{selected_variable}/selected_features.json', 'r') as f:
            feature_importance = json.load(f)
    except:
        # Fallback to demo data if files don't exist
        forecast_df = pd.DataFrame({
            'date': [datetime.now().date() + timedelta(days=i) for i in range(1, 31)],
            'forecast': variable_data['value'].iloc[-1] + np.random.normal(0, 2, 30).cumsum()
        })
        metrics = {'xgb': {'MAE': 1.2, 'RMSE': 1.8, 'R2': 0.89}}
        feature_importance = {'Feature1': 0.8, 'Feature2': 0.6, 'Feature3': 0.4, 'Feature4': 0.3}
    
    # 1. Forecast Plot
    fig_forecast = go.Figure()
    
    # Historical data
    fig_forecast.add_trace(go.Scatter(
        x=variable_data['date'],
        y=variable_data['value'],
        mode='lines',
        name='Historical',
        line=dict(color='#3498db', width=2),
        hovertemplate='Date: %{x}<br>Value: %{y:.2f}<extra></extra>'
    ))
    
    # Forecast data
    fig_forecast.add_trace(go.Scatter(
        x=forecast_df['date'],
        y=forecast_df['forecast'],
        mode='lines',
        name='Forecast',
        line=dict(color='#e74c3c', width=2, dash='dash'),
        hovertemplate='Date: %{x}<br>Forecast: %{y:.2f}<extra></extra>'
    ))
    
    # Confidence interval (simulated)
    upper_bound = forecast_df['forecast'] * 1.1
    lower_bound = forecast_df['forecast'] * 0.9
    
    fig_forecast.add_trace(go.Scatter(
        x=forecast_df['date'].tolist() + forecast_df['date'].tolist()[::-1],
        y=upper_bound.tolist() + lower_bound.tolist()[::-1],
        fill='toself',
        fillcolor='rgba(231, 76, 60, 0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        name='90% Confidence',
        hoverinfo='skip'
    ))
    
    fig_forecast.update_layout(
        title=f'{selected_variable.replace("_", " ").title()} - 30-Day Forecast',
        xaxis_title='Date',
        yaxis_title='Value',
        hovermode='x unified',
        showlegend=True,
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    # 2. Key Metrics
    current_value = variable_data['value'].iloc[-1]
    forecast_value = forecast_df['forecast'].iloc[0]
    change_pct = ((forecast_value - current_value) / current_value) * 100
    
    metrics_display = html.Div([
        html.H4("📊 Current Metrics"),
        html.P(f"Current Value: {current_value:.2f}"),
        html.P(f"Next Forecast: {forecast_value:.2f}"),
        html.P(f"Change: {change_pct:+.1f}%", 
               style={'color': 'green' if change_pct >= 0 else 'red'}),
        html.P(f"Best Model: XGBoost (MAE: {metrics.get('xgb', {}).get('MAE', 'N/A'):.2f})")
    ])
    
    # 3. Metrics Table
    metrics_table = html.Table([
        html.Thead(html.Tr([html.Th('Model'), html.Th('MAE'), html.Th('RMSE'), html.Th('R²')])),
        html.Tbody([
            html.Tr([html.Td('XGBoost'), html.Td(f"{metrics.get('xgb', {}).get('MAE', 'N/A'):.3f}"), 
                    html.Td(f"{metrics.get('xgb', {}).get('RMSE', 'N/A'):.3f}"), 
                    html.Td(f"{metrics.get('xgb', {}).get('R2', 'N/A'):.3f}")]),
            html.Tr([html.Td('MLP'), html.Td(f"{metrics.get('mlp', {}).get('MAE', 'N/A'):.3f}"), 
                    html.Td(f"{metrics.get('mlp', {}).get('RMSE', 'N/A'):.3f}"), 
                    html.Td(f"{metrics.get('mlp', {}).get('R2', 'N/A'):.3f}")]),
            html.Tr([html.Td('GRU'), html.Td(f"{metrics.get('gru', {}).get('MAE', 'N/A'):.3f}"), 
                    html.Td(f"{metrics.get('gru', {}).get('RMSE', 'N/A'):.3f}"), 
                    html.Td(f"{metrics.get('gru', {}).get('R2', 'N/A'):.3f}")])
        ])
    ], style={'width': '100%', 'borderCollapse': 'collapse'})
    
    # 4. Feature Importance Plot
    features_df = pd.DataFrame({
        'feature': list(feature_importance.keys()),
        'importance': list(feature_importance.values())
    }).sort_values('importance', ascending=True)
    
    fig_features = px.bar(features_df, y='feature', x='importance', orientation='h',
                         title='Feature Importance Ranking')
    fig_features.update_layout(showlegend=False, plot_bgcolor='white', paper_bgcolor='white')
    
    # 5. Actual vs Predicted Plot (simulated)
    actual_vs_predicted = pd.DataFrame({
        'date': variable_data['date'].iloc[-100:],
        'actual': variable_data['value'].iloc[-100:],
        'predicted': variable_data['value'].iloc[-100:] + np.random.normal(0, 1, 100)
    })
    
    fig_actual_pred = go.Figure()
    fig_actual_pred.add_trace(go.Scatter(x=actual_vs_predicted['date'], y=actual_vs_predicted['actual'],
                                        name='Actual', line=dict(color='blue')))
    fig_actual_pred.add_trace(go.Scatter(x=actual_vs_predicted['date'], y=actual_vs_predicted['predicted'],
                                        name='Predicted', line=dict(color='red', dash='dash')))
    fig_actual_pred.update_layout(title='Actual vs Predicted Values', plot_bgcolor='white', paper_bgcolor='white')
    
    # 6. Historical Trends
    fig_trends = px.line(variable_data, x='date', y='value', 
                        title=f'{selected_variable} Historical Trend')
    fig_trends.update_layout(plot_bgcolor='white', paper_bgcolor='white')
    
    # 7. Correlation Heatmap (simulated with multiple variables)
    corr_data = df[['Maize_meal', 'CPI', 'Diesel', 'Inflation']].corr()
    fig_heatmap = px.imshow(corr_data, text_auto=True, aspect="auto",
                           title='Correlation Between Economic Indicators')
    fig_heatmap.update_layout(plot_bgcolor='white', paper_bgcolor='white')
    
    # 8. SHAP Explanation
    shap_explanation = html.Div([
        html.H5("How features influence the current forecast:"),
        html.Ul([
            html.Li("📈 Diesel prices: +2.3 units (increasing forecast)"),
            html.Li("📉 Exchange rate: -1.8 units (decreasing forecast)"),
            html.Li("📈 Import costs: +1.5 units (increasing forecast)"),
            html.Li("📉 Rainfall: -0.9 units (decreasing forecast)")
        ]),
        html.P("The forecast is most sensitive to changes in diesel prices and exchange rates.")
    ])
    
    return (fig_forecast, metrics_display, metrics_table, fig_features, 
            fig_actual_pred, fig_trends, fig_heatmap, shap_explanation)

# Run the app
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8050)
