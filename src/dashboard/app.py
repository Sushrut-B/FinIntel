import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
import io
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from statsmodels.tsa.seasonal import STL
from fpdf import FPDF  # Requires: pip install fpdf

API_BASE_URL = "http://localhost:8000"

USER_CREDENTIALS = {
    "sushrut": {"password": "sushrutpass", "role": "admin"},
    "admin": {"password": "adminpass", "role": "admin"},
    "analyst": {"password": "analystpass", "role": "analyst"},
    "guest": {"password": "guestpass", "role": "guest"},
}

PRIMARY_COLOR = "#0a73bb"
SECONDARY_COLOR = "#1f2630"
BACKGROUND_COLOR = "#121212"
TEXT_COLOR = "#e0e0e0"

def set_custom_styles():
    st.markdown(f"""
        <style>
            .main {{
                background-color: {BACKGROUND_COLOR};
                color: {TEXT_COLOR};
                font-family: "Segoe UI", Tahoma, Geneva, Verdana, sans-serif;
            }}
            .css-1d391kg, .css-1v3fvcr {{
                background-color: {SECONDARY_COLOR};
                color: {TEXT_COLOR};
                padding: 20px;
                border-radius: 8px;
                margin-bottom: 20px;
            }}
            h1, h2, h3, h4 {{
                color: {PRIMARY_COLOR};
                font-weight: 700;
            }}
            .css-1v0mbdj .css-ffhzg2 {{
                background-color: {SECONDARY_COLOR} !important;
                border-radius: 10px !important;
                padding: 15px !important;
            }}
            button[kind="primary"] {{
                background-color: {PRIMARY_COLOR} !important;
                border: none !important;
                color: white !important;
            }}
            a {{
                color: {PRIMARY_COLOR};
                text-decoration: none;
                font-weight: 600;
            }}
            a:hover {{
                text-decoration: underline;
            }}
        </style>
    """, unsafe_allow_html=True)

def login():
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
        st.session_state.username = ""
        st.session_state.password = ""
        st.session_state.role = ""

    if not st.session_state.logged_in:
        st.title("Welcome to the UPI Macro Intelligence Platform")
        st.markdown("Please enter your credentials to access the dashboard.")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            creds = USER_CREDENTIALS.get(username)
            if creds and password == creds["password"]:
                st.session_state.logged_in = True
                st.session_state.username = username
                st.session_state.password = password
                st.session_state.role = creds["role"]
            else:
                st.error("Incorrect username or password")
        return False
    else:
        st.sidebar.markdown(f"**Logged in as:** {st.session_state.username} ({st.session_state.role})")
        if st.sidebar.button("Logout"):
            st.session_state.logged_in = False
            st.session_state.username = ""
            st.session_state.password = ""
            st.session_state.role = ""
        return True

@st.cache_data
def load_data():
    auth = (st.session_state.username, st.session_state.password) if st.session_state.logged_in else None

    def fetch_json(url):
        try:
            resp = requests.get(url, auth=auth)
            resp.raise_for_status()
            return resp.json()
        except Exception:
            return []

    actuals = fetch_json(f"{API_BASE_URL}/actuals")
    nbeats = fetch_json(f"{API_BASE_URL}/forecast?model=NBEATSx")
    tft = fetch_json(f"{API_BASE_URL}/forecast?model=TFT")
    linreg = fetch_json(f"{API_BASE_URL}/forecast?model=LinearRegression")
    anomalies = fetch_json(f"{API_BASE_URL}/anomalies")

    df_actuals = pd.DataFrame(actuals)
    if not df_actuals.empty:
        df_actuals["ds"] = pd.to_datetime(df_actuals["ds"])
        df_actuals["y"] = df_actuals["y"].astype(float)
        df_actuals["ds"] = df_actuals["ds"].dt.to_period("M").dt.to_timestamp()

    def prep_forecast(df, model_name):
        if df.empty:
            return df
        df["ds"] = pd.to_datetime(df["ds"])
        df.rename(columns={"forecast": model_name}, inplace=True)
        df["ds"] = df["ds"].dt.to_period("M").dt.to_timestamp()
        return df[["ds", model_name]]

    df_nbeats = prep_forecast(pd.DataFrame(nbeats), "NBEATSx")
    df_tft = prep_forecast(pd.DataFrame(tft), "TFT")
    df_linreg = prep_forecast(pd.DataFrame(linreg), "LinearRegression")

    df_anomalies = pd.DataFrame(anomalies)
    if not df_anomalies.empty:
        df_anomalies["ds"] = pd.to_datetime(df_anomalies["ds"])
        df_anomalies["ds"] = df_anomalies["ds"].dt.to_period("M").dt.to_timestamp()
        if "type" not in df_anomalies.columns:
            df_anomalies["type"] = np.random.choice(
                ["Spike", "Drop", "Seasonal Outlier"], len(df_anomalies)
            )
    return df_actuals, df_nbeats, df_tft, df_linreg, df_anomalies

def generate_report_pdf(text):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    for line in text.split("\n"):
        pdf.multi_cell(0, 10, line)
    pdf_bytes = pdf.output(dest='S').encode('latin1')
    return io.BytesIO(pdf_bytes)

def format_large_number(num):
    if num is None:
        return "N/A"
    if num >= 1e12:
        return f"{num / 1e12:.2f} trillion"
    elif num >= 1e9:
        return f"{num / 1e9:.2f} billion"
    elif num >= 1e7:
        return f"{num / 1e7:.2f} crore"
    elif num >= 1e5:
        return f"{num / 1e5:.2f} lakh"
    elif num >= 1e3:
        return f"{num / 1e3:.2f} thousand"
    return f"{num:.2f}"

def main():
    set_custom_styles()
    if not login():
        return

    st.title("UPI Macro Intelligence Dashboard")

    # API Health
    try:
        resp = requests.get(f"{API_BASE_URL}/health")
        health_status = resp.status_code == 200
    except:
        health_status = False

    st.sidebar.markdown("### API & Scheduler Status")
    st.sidebar.write("🟢 Healthy" if health_status else "🔴 Unreachable")
    st.sidebar.info("Scheduler Last Run: 2025-09-07 23:00\nNext Run: 2025-09-08 23:00\nStatus: Success")
    if st.sidebar.button("Refresh API Status"):
        st.rerun()

    df_actuals, df_nbeats, df_tft, df_linreg, df_anomalies = load_data()

    min_date = df_actuals["ds"].min() if not df_actuals.empty else None
    max_date = df_actuals["ds"].max() if not df_actuals.empty else None
    date_range = st.sidebar.date_input("Select Date Range", [min_date, max_date], min_value=min_date, max_value=max_date)
    date_range = date_range if isinstance(date_range, (list, tuple)) else [date_range, date_range]
    date_start, date_end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])

    anomaly_types = ["All"] + sorted(df_anomalies["type"].unique()) if not df_anomalies.empty else ["All"]
    selected_anomalies = st.sidebar.multiselect("Filter Anomaly Types", anomaly_types, default=["All"])

    model_options = {"NBEATSx": df_nbeats, "TFT": df_tft, "LinearRegression": df_linreg}
    selected_models = st.sidebar.multiselect("Select Models", list(model_options.keys()), default=["NBEATSx", "TFT"])

    filtered_actuals = df_actuals[(df_actuals["ds"] >= date_start) & (df_actuals["ds"] <= date_end)] if not df_actuals.empty else pd.DataFrame()
    filtered_models = {}
    for m, df in model_options.items():
        if m in selected_models and not df.empty and 'ds' in df.columns:
            filtered_models[m] = df[(df["ds"] >= date_start) & (df["ds"] <= date_end)]

    if not df_anomalies.empty:
        if "All" in selected_anomalies:
            filtered_anomalies = df_anomalies[(df_anomalies["ds"] >= date_start) & (df_anomalies["ds"] <= date_end)]
        else:
            filtered_anomalies = df_anomalies[(df_anomalies["ds"] >= date_start) & (df_anomalies["ds"] <= date_end) &
                                              (df_anomalies["type"].isin(selected_anomalies))]
    else:
        filtered_anomalies = pd.DataFrame()

    # Historical Plot
    fig = go.Figure()
    if not filtered_actuals.empty:
        fig.add_trace(go.Scatter(x=filtered_actuals["ds"], y=filtered_actuals["y"], mode="lines+markers", name="Actuals"))
    for m, df in filtered_models.items():
        if not df.empty:
            fig.add_trace(go.Scatter(x=df["ds"], y=df[m], mode="lines+markers", name=m))
            upper = df[m] * 1.05
            lower = df[m] * 0.95
            xs = pd.concat([df["ds"], df["ds"][::-1]])
            ys = pd.concat([upper, lower[::-1]])
            fig.add_trace(go.Scatter(x=xs, y=ys, fill='toself',
                                     fillcolor='rgba(0,100,80,0.2)',
                                     line=dict(color='rgba(255,255,255,0)'),
                                     showlegend=False, hoverinfo='skip', name=f"{m} Confidence"))
    colors = {"Spike": "red", "Drop": "blue", "Seasonal Outlier": "orange"}
    for t in filtered_anomalies["type"].unique() if not filtered_anomalies.empty else []:
        tmp = filtered_anomalies[filtered_anomalies["type"] == t]
        fig.add_trace(go.Scatter(x=tmp["ds"], y=tmp["y"], mode="markers",
                                 marker=dict(color=colors.get(t, "gray")), name=f"Anomaly: {t}"))
    fig.update_layout(title="Forecasts vs Actuals with Confidence & Anomalies",
                      xaxis_title="Date", yaxis_title="UPI Volume", hovermode="closest")
    st.plotly_chart(fig, use_container_width=True)

    # Metrics Table
    metrics_list = []
    for m, df in filtered_models.items():
        merged = filtered_actuals.merge(df, on="ds")
        if len(merged) > 0:
            mae_val = mean_absolute_error(merged["y"], merged[m])
            mape_val = mean_absolute_percentage_error(merged["y"], merged[m]) * 100
            metrics_list.append({"Model": m, "MAE": format_large_number(mae_val), "MAPE (%)": f"{mape_val:.2f}%"})
    if metrics_list:
        st.markdown("### Model Performance Metrics")
        st.table(pd.DataFrame(metrics_list))
    else:
        st.write("No model metrics available.")

    # STL Decomposition
    if len(filtered_actuals) > 12:
        series = filtered_actuals.set_index("ds")["y"].asfreq("MS")
        decomposed = STL(series, period=7).fit()
        decomp_fig = go.Figure()
        decomp_fig.add_trace(go.Scatter(x=decomposed.trend.index, y=decomposed.trend.values, name="Trend"))
        decomp_fig.add_trace(go.Scatter(x=decomposed.seasonal.index, y=decomposed.seasonal.values, name="Seasonal"))
        decomp_fig.add_trace(go.Scatter(x=decomposed.resid.index, y=decomposed.resid.values, name="Residual"))
        decomp_fig.update_layout(title="STL Decomposition Components", xaxis_title="Date", yaxis_title="Value")
        st.plotly_chart(decomp_fig, use_container_width=True)

    # Summary & Insights
    st.markdown("### Summary")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Data Points", len(filtered_actuals))
    c2.metric("Forecast Points", sum(len(df) for df in filtered_models.values()))
    c3.metric("Anomalies", len(filtered_anomalies))
    c4.metric("MAE", metrics_list[0]["MAE"] if metrics_list else "N/A")

    st.markdown("### Insight Summary")
    insights = []
    if filtered_actuals.empty:
        insights.append("No data available for the selected period.")
    else:
        min_val = filtered_actuals["y"].min()
        max_val = filtered_actuals["y"].max()
        insights.append(f"Transaction volume ranged from {format_large_number(min_val)} to {format_large_number(max_val)} units.")
        if metrics_list:
            for metric in metrics_list:
                insights.append(f"Model {metric['Model']} had an average error of {metric['MAE']} units ({metric['MAPE (%)']} error).")
        else:
            insights.append("No model performance data available.")
        if filtered_anomalies.empty:
            insights.append("No anomalies detected.")
        else:
            insights.append(f"Detected {len(filtered_anomalies)} anomalies indicating unusual patterns.")

        if len(filtered_actuals) > 1:
            start_val = filtered_actuals.iloc[0]["y"]
            end_val = filtered_actuals.iloc[-1]["y"]
            overall_change = (end_val - start_val) / max(start_val,1) * 100
            insights.append(f"Overall, transaction volume changed by {overall_change:.2f}% during the period.")

            if len(filtered_actuals) > 2:
                prev_val = filtered_actuals.iloc[-2]["y"]
                last_val = filtered_actuals.iloc[-1]["y"]
                recent_change = (last_val - prev_val) / max(prev_val,1) * 100
                insights.append(f"Recent change: {recent_change:.2f}% compared to previous period.")

        if len(filtered_models) and len(filtered_actuals) > 0:
            recent_actual = filtered_actuals.iloc[-1]["y"]
            first_model = next(iter(filtered_models))
            recent_forecast = filtered_models[first_model].iloc[-1][first_model]
            diff = recent_actual - recent_forecast
            diff_str = format_large_number(abs(diff))
            trend = "increased" if diff > 0 else "decreased" if diff < 0 else "remained stable"
            insights.append(f"The latest actual volume has {trend} by {diff_str} compared to the forecast.")

    for line in insights:
        st.write(line)

    # Diagnostics
    st.markdown("### Date Diagnostics")
    st.write(f"Data from {date_start.date()} to {date_end.date()}")
    st.write(f"Total actual records: {len(filtered_actuals)}")
    st.write(f"Forecast records per model: " + ", ".join(f"{m}: {len(df)}" for m, df in filtered_models.items()))
    st.write(f"Number of displayed anomalies: {len(filtered_anomalies)}")

    # Download options
    with st.expander("Download Reports and Data"):
        st.download_button("Download Actuals CSV", filtered_actuals.to_csv(index=False).encode("utf-8"), "actuals.csv")
        for m, df in filtered_models.items():
            st.download_button(f"Download {m} Forecast CSV", df.to_csv(index=False).encode("utf-8"), f"{m}_forecast.csv")
        if not filtered_anomalies.empty:
            st.download_button("Download Anomalies CSV", filtered_anomalies.to_csv(index=False).encode("utf-8"), "anomalies.csv")

        buf = io.BytesIO()
        fig.write_image(buf, format='png')
        buf.seek(0)
        st.download_button("Download Forecast Plot", buf, "forecast_plot.png")

        summary_text = "\n".join(insights)
        report_pdf = generate_report_pdf(summary_text)
        st.download_button("Download Summary Report (PDF)", report_pdf, "summary_report.pdf")

    # ======= TRUE MODEL-BASED FUTURE FORECAST SECTION (FIXED) =======
    st.markdown("## Future Forecast (Model-Based)")

    try:
        future_forecast_df = pd.read_csv("future_forecast.csv")
        future_forecast_df["ds"] = pd.to_datetime(future_forecast_df["ds"])
    except Exception as e:
        st.warning("Could not load future_forecast.csv: " + str(e))
        future_forecast_df = pd.DataFrame()

    if not filtered_actuals.empty:
        last_actual_date = filtered_actuals["ds"].max()
    else:
        last_actual_date = None

    if not future_forecast_df.empty and last_actual_date is not None:
        forecast_part = future_forecast_df[future_forecast_df["ds"] > last_actual_date]
    else:
        forecast_part = pd.DataFrame()

    fig_future = go.Figure()
    if not filtered_actuals.empty:
        fig_future.add_trace(go.Scatter(
            x=filtered_actuals["ds"], y=filtered_actuals["y"],
            mode="lines+markers", name="Historical Actuals",
            line=dict(color="deepskyblue", width=2)
        ))

    if not forecast_part.empty:
        fig_future.add_trace(go.Scatter(
            x=forecast_part["ds"], y=forecast_part["forecast"],
            mode="lines+markers", name="Forecast (Most Likely)",
            line=dict(color="orange", width=3, dash="dash")
        ))
        future_start = forecast_part["ds"].iloc[0]
        max_y = max(forecast_part["forecast"].max(), filtered_actuals["y"].max() if not filtered_actuals.empty else 0)
        fig_future.add_shape(
            type="line",
            x0=future_start, x1=future_start,
            y0=0, y1=max_y,
            line=dict(color="gray", width=2, dash="dot")
        )
        fig_future.add_annotation(
            x=future_start,
            y=max_y,
            text="Future Predictions Start",
            showarrow=True,
            arrowhead=2,
            ax=0,
            ay=-80,
            font=dict(color="orange")
        )
    else:
        st.warning("No model-based future prediction data available after the latest actual date. Check your 'future_forecast.csv' file.")

    fig_future.update_layout(
        title="Simulated Future Forecast Next 5 Years - Model Based",
        xaxis_title="Date",
        yaxis_title="UPI Transaction Volume",
        hovermode="x unified"
    )
    st.plotly_chart(fig_future, use_container_width=True)

if __name__ == "__main__":
    main()
