import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression


# =====================================================
# LOAD DATA
# =====================================================
def load_data(path):
    df = pd.read_csv(path)

    # Combine date and time into one proper timestamp
    df['Admission_DateTime'] = pd.to_datetime(
        df['Admission_Date'] + ' ' + df['Admission_Time']
    )

    df.dropna(inplace=True)
    return df


# =====================================================
# MAIN ANALYSIS FUNCTION
# =====================================================

def predict_next_week(df):

    df = df.copy()

    # ---------- Create 12-hour bins ----------
    df['interval_12h'] = df['Admission_DateTime'].dt.floor('12h')

    agg = (
        df.groupby(['Disease', 'interval_12h'])
        .size()
        .reset_index(name='cases')
        .sort_values('interval_12h')
    )

    results = []
    plot_data = []  # store only diseases that qualify

    diseases = agg['Disease'].unique()

    # =================================================
    # FIRST PASS — analysis only
    # =================================================
    for disease in diseases:

        ddf = agg[agg['Disease'] == disease].copy()
        ddf = ddf.sort_values('interval_12h').reset_index(drop=True)

        if len(ddf) < 10:
            results.append([disease, 0, "insufficient_data"])
            continue

        # ---------- Isolation Forest ----------
        iso = IsolationForest(contamination=0.1, random_state=42)
        ddf['anomaly'] = iso.fit_predict(ddf[['cases']])
        ddf['anomaly_flag'] = (ddf['anomaly'] == -1).astype(int)

        anomaly_indices = np.where(ddf['anomaly_flag'] == 1)[0]

        consecutive_alert = 0
        trend_direction = "stable"
        slope = 0

        # ---------- Check 3 consecutive ----------
        for i in range(len(anomaly_indices) - 2):
            if (anomaly_indices[i + 2] - anomaly_indices[i]) == 2:
                consecutive_alert = 1

                # ---------- Regression ----------
                X = np.arange(len(ddf)).reshape(-1, 1)
                y = ddf['cases'].values

                model = LinearRegression()
                model.fit(X, y)

                slope = model.coef_[0]

                if slope > 0:
                    trend_direction = "increase"
                elif slope < 0:
                    trend_direction = "decrease"
                else:
                    trend_direction = "stable"

                break

        results.append([disease, consecutive_alert, trend_direction])

        # =================================================
        # STORE ONLY IF RISING
        # =================================================
        if consecutive_alert == 1 and trend_direction == "increase":
            plot_data.append((disease, ddf, slope))

    # =================================================
    # SECOND PASS — plotting only rising diseases
    # =================================================
    if len(plot_data) > 0:

        fig, axes = plt.subplots(len(plot_data), 1,
                                 figsize=(12, 4 * len(plot_data)),
                                 squeeze=False)

        for idx, (disease, ddf, slope) in enumerate(plot_data):

            ax = axes[idx, 0]

            # base line
            ax.plot(ddf['interval_12h'], ddf['cases'],
                    marker='o', label='Cases')

            # anomalies
            anomalies = ddf[ddf['anomaly_flag'] == 1]
            ax.scatter(anomalies['interval_12h'],
                       anomalies['cases'],
                       color='red', label='Anomaly', zorder=5)

            # regression line
            X = np.arange(len(ddf)).reshape(-1, 1)
            model = LinearRegression().fit(X, ddf['cases'].values)
            trend_line = model.predict(X)

            ax.plot(ddf['interval_12h'], trend_line,
                    linestyle='--', label='Trend')

            ax.set_title(f"Disease (Rising): {disease}")
            ax.set_xlabel("Time (12-hour intervals)")
            ax.set_ylabel("Case Count")
            ax.grid(alpha=0.3)
            ax.legend()

        plt.tight_layout()
        plt.show()

    else:
        print("\nNo diseases with rising trend after anomaly detection.")

    result_df = pd.DataFrame(
        results,
        columns=[
            'Disease',
            'three_interval_anomaly',
            'trend_prediction'
        ]
    )

    return result_df

# =====================================================
# MAIN DRIVER
# =====================================================
if __name__ == "__main__":

    df = load_data("data/clean_hospital_dataset_with_spikes.csv")

    final_output = predict_next_week(df)

    print("\nFinal Output:")
    print(final_output)

    # ---------- Optional: Top risk visualization ----------
    if not final_output.empty:
        final_output['risk_flag'] = (
            final_output['three_interval_anomaly'] == 1
        ).astype(int)

        top_risk = final_output.sort_values(
            by='risk_flag',
            ascending=False
        ).head(10)

        plt.figure(figsize=(10, 6))
        sns.barplot(
            x='risk_flag',
            y='Disease',
            data=top_risk,
            palette='viridis'
        )
        plt.title('Diseases with Consecutive Anomaly Risk')
        plt.xlabel('Risk Flag')
        plt.ylabel('Disease')
        plt.tight_layout()
        plt.show()

        # Save the plot_data list so the dashboard can access the raw time-series
        joblib.dump(final_output, "plot_data.pkl")
        print("\nmodel saved successfully")
        # ... (at the very bottom of your script)
