import dash
from dash import dcc, html, Input, Output, callback
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import json
import base64
import os

# Initialize the Dash App
app = dash.Dash(__name__)
app.title = "Eswatini Economic Forecast Dashboard"

# Define the variables you are forecasting
VARIABLE_OPTIONS = [
    {'label': 'Maize meal SZL/1kg', 'value': 'maize_meal'},
    {'label': 'All Items CPI', 'value': 'cpi'},
    {'label': 'Inflation rate', 'value': 'inflation'},
    {'label': 'Diesel SZL/1 liter', 'value': 'diesel'},
    {'label': 'Cabbage SZL/Head', 'value': 'cabbage'},
    {'label': 'Tomato (Round) SZL/1kg', 'value': 'tomato'},
    {'label': 'Rice SZL/1kg', 'value': 'rice'},
    {'label': 'Beans SZL/1kg', 'value': 'beans'},
    {'label': 'Sugar SZL/1kg', 'value': 'sugar'},
    {'label': 'Interest Rate', 'value': 'interest_rate'}
]

# App Layout
app.layout = html.Div([
    html.H1("Eswatini Economic Indicator Forecast", style={'textAlign': 'center'}),
    html.Hr(),

    # Dropdown for variable selection
    html.Label("Select an Economic Indicator to Forecast:"),
    dcc.Dropdown(
        id='variable-selector',
        options=VARIABLE_OPTIONS,
        value='maize_meal',  # Default selection
        clearable=False
    ),

    html.Br(),
    html.Div(id='last-update-div'),
    html.Br(),

    # Forecast Plot
    dcc.Graph(id='forecast-plot'),

    html.Br(),

    # Metrics and Explanations in Tabs
    dcc.Tabs([
        dcc.Tab(label='Model Performance', children=[
            html.Div(id='metrics-display')
        ]),
        dcc.Tab(label='Global Feature Importance', children=[
            html.Img(id='global-shap-plot')
        ]),
        dcc.Tab(label='Explanation for Latest Forecast', children=[
            html.Iframe(id='local-shap-plot', style={'height': '400px', 'width': '100%'})
        ])
    ])
])

# Callback to update all components based on dropdown selection
@callback(
    [Output('forecast-plot', 'figure'),
     Output('metrics-display', 'children'),
     Output('global-shap-plot', 'src'),
     Output('local-shap-plot', 'srcDoc'),
     Output('last-update-div', 'children')],
    [Input('variable-selector', 'value')]
)
def update_dashboard(selected_variable):
    # 1. Construct the path to the precomputed files for the selected variable
    base_path = f'model_artifacts/{selected_variable}'

    # 2. Load the forecast data
    forecast_df = pd.read_csv(f'{base_path}/forecast.csv')

    # 3. Create the Forecast Plot
    fig = make_subplots(specs=[[{"secondary_y": False}]])
    # Add historical data
    fig.add_trace(go.Scatter(x=forecast_df['date'], y=forecast_df['historical'],
                             mode='lines', name='Historical', line=dict(color='blue')))
    # Add forecasted data
    fig.add_trace(go.Scatter(x=forecast_df['date'], y=forecast_df['forecast'],
                             mode='lines', name='Forecast', line=dict(color='red', dash='dash')))
    fig.update_layout(title=f'30-Day Forecast for {selected_variable.replace("_", " ").title()}',
                      xaxis_title='Date', yaxis_title='Price (SZL)')

    # 4. Load and display metrics
    with open(f'{base_path}/metrics.json', 'r') as f:
        metrics = json.load(f)
    metrics_text = [html.P(f"{key}: {value:.4f}") for key, value in metrics.items()]

    # 5. Load and display Global SHAP image
    with open(f'{base_path}/global_shap.png', 'rb') as img_file:
        encoded_global_shap = base64.b64encode(img_file.read()).decode('ascii')
    global_src = 'data:image/png;base64,{}'.format(encoded_global_shap)

    # 6. Load and display Local SHAP plot (HTML file)
    with open(f'{base_path}/local_shap.html', 'r') as f:
        local_shap_html = f.read()

    # 7. Get the last updated time
    last_updated = "Forecasts Last Updated: Monday, October 7, 2024"  # You can make this dynamic

    return fig, metrics_text, global_src, local_shap_html, last_updated

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
