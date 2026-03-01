import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

st.set_page_config(page_title="Outbreak Tracker", layout="wide")
# Add this right after st.set_page_config
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stMetric { background-color: #000000; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    h1 { color: #2c3e50; }
    </style>
    """, unsafe_allow_html=True)

# Styling
st.title("🏥 Disease Surveillance Dashboard")
st.sidebar.header("Controls")

@st.cache_data
def load_results():
    if os.path.exists("model.pkl"):
        return joblib.load("model.pkl")
    return None

results = load_results()

if results is not None:
    # sorting
    results = results.sort_values(
        by=['three_interval_anomaly', 'trend_prediction'], 
        ascending=[False, True]
    )

    total_diseases = len(results)
    high_risk_count = int(results['three_interval_anomaly'].sum())
    
    c1, c2 = st.columns(2)
    # Adding 'label' and 'value' clearly
    c1.metric(label="Total Diseases Monitored", value=total_diseases)
    c2.metric(label="High Risk Alerts 🚨", value=high_risk_count)

    # Layout Tabs
    tab1, tab2 = st.tabs(["Global Risk Overview", "Detailed Trends"])

    with tab1:
        st.subheader("Anomaly Risk by Disease")
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.barplot(data=results, x='three_interval_anomaly', y='Disease', palette='Reds_r', ax=ax)
        st.pyplot(fig)

    with tab2:
        st.write("### Data Table")
        def color_trend(val):
            color = '#ff4b4b' if val == 'increase' else '#28a745' if val == 'stable' else 'white'
            return f'color: {color}; font-weight: bold'

        styled_df = results.style.applymap(color_trend, subset=['trend_prediction'])
        st.dataframe(styled_df, use_container_width=True)
else:
    st.error("Please run the disease_model.py script first to generate results.")
