import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

st.set_page_config(page_title="Outbreak Tracker", layout="wide")

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
    # Metric Row
    high_risk_count = results[results['three_interval_anomaly'] == 1].shape[0]
    c1, c2 = st.columns(2)
    c1.metric("Total Diseases", len(results))
    c2.metric("High Risk Alerts", high_risk_count)

    # Layout Tabs
    tab1, tab2 = st.tabs(["Global Risk Overview", "Detailed Trends"])

    with tab1:
        st.subheader("Anomaly Risk by Disease")
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.barplot(data=results, x='three_interval_anomaly', y='Disease', palette='Reds_r', ax=ax)
        st.pyplot(fig)

    with tab2:
        st.write("### Data Table")
        st.dataframe(results, use_container_width=True)
else:
    st.error("Please run the disease_model.py script first to generate results.")