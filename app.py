# -*- coding: utf-8 -*-
"""Eswatini_Agri_Economic_Dashboard"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# -------------------------------------------------------
# CONFIG
# -------------------------------------------------------
st.set_page_config(page_title="🇸🇿 Eswatini Agricultural & Economic Dashboard",
                   layout="wide")

st.title("🇸🇿 Eswatini Agricultural & Economic Forecast Dashboard")

# -------------------------------------------------------
# BASE VALUES PROVIDED
# -------------------------------------------------------
forecast_values = {
    'Tomato (Round) SZL/1kg': 13,
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

# -------------------------------------------------------
# FUNCTION TO GENERATE 30-DAY RANDOM FLUCTUATIONS
# -------------------------------------------------------
@st.cache_data
def generate_random_fluctuations():
    forecasts = {}
    future_dates = [datetime.now().date() + timedelta(days=i) for i in range(1, 31)]

    for variable, value in forecast_values.items():
        fluctuations = [round(random.uniform(value * 0.95, value * 1.05), 2) for _ in range(30)]
        forecasts[variable] = pd.DataFrame({
            'date': future_dates,
            'expected_price': fluctuations
        })
    return forecasts

forecasts = generate_random_fluctuations()

# -------------------------------------------------------
# SIDEBAR
# -------------------------------------------------------
st.sidebar.header("⚙️ Dashboard Settings")
selected_variable = st.sidebar.selectbox("Select Economic Indicator:", list(forecast_values.keys()))

# -------------------------------------------------------
# MAIN METRICS DISPLAY
# -------------------------------------------------------
current_value = forecast_values[selected_variable]
st.metric(label=f"Current Value of {selected_variable}", value=current_value)

# -------------------------------------------------------
# PRICE TABLE (Replacing Forecast Visualization)
# -------------------------------------------------------
st.subheader(f"📊 30-Day Price Fluctuations: {selected_variable}")
forecast_data = forecasts[selected_variable]
st.dataframe(forecast_data, use_container_width=True)

# -------------------------------------------------------
# INSIGHTS SECTION (Expanded for all variables)
# -------------------------------------------------------
st.subheader("🔍 Insights & Strategic Recommendations")

insights = {
    'Tomato (Round) SZL/1kg': {
        "primary_drivers": [
            "Seasonal rainfall variation",
            "Pest infestations (e.g., Tuta absoluta)",
            "Transport and logistics costs",
            "Market demand fluctuations in urban centers"
        ],
        "actions": [
            "Promote greenhouse adoption to stabilize yields",
            "Strengthen pest and disease monitoring systems",
            "Support cooperative transport schemes",
            "Encourage local tomato processing industries to absorb surplus"
        ]
    },
    'Cabbage SZL/Head': {
        "primary_drivers": [
            "Temperature extremes affecting head formation",
            "Fuel and irrigation costs",
            "Soil fertility levels",
            "Cross-border demand from neighboring countries"
        ],
        "actions": [
            "Introduce drought- and heat-tolerant varieties",
            "Provide subsidized irrigation equipment",
            "Promote soil testing and organic composting",
            "Strengthen export linkages with Mozambique and South Africa"
        ]
    },
    'Maize SZL/50kg': {
        "primary_drivers": [
            "Rainfall variability",
            "Fertilizer and input costs",
            "Regional import-export restrictions",
            "Storage and post-harvest losses"
        ],
        "actions": [
            "Promote climate-smart maize hybrids",
            "Subsidize fertilizers and improve input access",
            "Invest in community grain storage facilities",
            "Negotiate regional grain trade agreements"
        ]
    },
    'Potatoes SZL/50kg': {
        "primary_drivers": [
            "Seed potato availability and quality",
            "Irrigation water access",
            "Transportation costs from highland farms",
            "Disease pressure (late blight)"
        ],
        "actions": [
            "Promote certified seed potato production",
            "Introduce water harvesting technologies",
            "Support farmer transport cooperatives",
            "Strengthen extension services for integrated pest management"
        ]
    },
    'Sugar SZL/1kg': {
        "primary_drivers": [
            "Global sugar market prices",
            "Fuel and irrigation costs",
            "Labor productivity and strikes",
            "Policy reforms on sugarcane pricing"
        ],
        "actions": [
            "Diversify sugarcane products (bioethanol, byproducts)",
            "Invest in energy-efficient irrigation systems",
            "Promote labor skills training",
            "Strengthen negotiation platforms between millers and growers"
        ]
    },
    'Beans SZL/1kg': {
        "primary_drivers": [
            "Seasonal rainfall",
            "Seed variety performance",
            "Storage losses from pests",
            "Local and regional demand"
        ],
        "actions": [
            "Promote drought-resistant bean varieties",
            "Introduce hermetic storage bags",
            "Encourage farmer cooperatives for bulk sales",
            "Strengthen regional trade networks"
        ]
    },
    'Onion SZL/1kg': {
        "primary_drivers": [
            "Post-harvest handling practices",
            "Storage infrastructure availability",
            "Irrigation costs",
            "Market access and middlemen influence"
        ],
        "actions": [
            "Invest in onion cold storage facilities",
            "Provide training on proper curing and packaging",
            "Encourage direct farmer-to-market linkages",
            "Promote irrigation efficiency technologies"
        ]
    },
    'Diesel SZL/1 liter': {
        "primary_drivers": [
            "Global crude oil price fluctuations",
            "Exchange rate volatility",
            "Domestic taxation policies",
            "Transportation bottlenecks"
        ],
        "actions": [
            "Encourage adoption of solar-powered irrigation",
            "Provide targeted fuel subsidies for farmers",
            "Diversify energy import sources",
            "Promote biofuel production locally"
        ]
    },
    'Gas SZL/1 liter': {
        "primary_drivers": [
            "International liquefied petroleum gas (LPG) prices",
            "Import dependency and port costs",
            "Currency exchange rate movements",
            "Distribution inefficiencies"
        ],
        "actions": [
            "Invest in biogas production from agricultural waste",
            "Negotiate regional LPG bulk purchasing agreements",
            "Improve domestic distribution networks",
            "Promote household-level renewable energy adoption"
        ]
    },
    'Inflation rate': {
        "primary_drivers": [
            "Food and fuel price hikes",
            "Currency exchange rate fluctuations",
            "Supply chain disruptions",
            "Monetary policy and interest rate changes"
        ],
        "actions": [
            "Stabilize staple food prices via strategic reserves",
            "Negotiate favorable fuel import contracts",
            "Support local production to reduce imports",
            "Enhance central bank monetary policy tools"
        ]
    },
    'Crop Production Index': {
        "primary_drivers": [
            "Adoption of modern farming technologies",
            "Climate variability",
            "Access to agricultural inputs",
            "Government policies and subsidies"
        ],
        "actions": [
            "Scale up mechanization programs",
            "Strengthen farmer extension services",
            "Improve access to affordable inputs",
            "Invest in agricultural research and innovation"
        ]
    }
}

# Display expanded insights
if selected_variable in insights:
    st.markdown("**Primary Drivers:**")
    for driver in insights[selected_variable]["primary_drivers"]:
        st.write(f"- {driver}")
    st.markdown("**Recommended Actions:**")
    for action in insights[selected_variable]["actions"]:
        st.write(f"- {action}")

# -------------------------------------------------------
# DOWNLOAD OPTION
# -------------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.markdown("### 📥 Download 30-Day Forecast Data")
csv_data = forecast_data.to_csv(index=False)
st.sidebar.download_button(
    label="Download CSV",
    data=csv_data,
    file_name=f"{selected_variable.replace(' ', '_')}_30day_forecast.csv",
    mime="text/csv"
)
