import streamlit as st
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import plotly.graph_objects as go
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet

st.set_page_config(page_title="FX Rate Forecasting", layout="wide")
st.title("💱 FX Forward & Futures Rate Forecasting")

# Sidebar options
pairs = ["USD/MXN", "EUR/USD", "GBP/USD", "USD/JPY", "AUD/USD"]
pair = st.sidebar.selectbox("Currency Pair", pairs)
data_type = st.sidebar.radio("Data Type", ["Forward Rates", "Futures Contracts"])
model_type = st.sidebar.selectbox("Forecast Model", ["ETS", "ARIMA", "Prophet"])
forecast_steps = st.sidebar.slider("Forecast Steps", 1, 12, 3)

headers = {"User-Agent": "Mozilla/5.0"}

@st.cache_data
def fetch_forward_rates(pair_slug):
    url = f"https://www.investing.com/currencies/{pair_slug}/forward-rates"
    resp = requests.get(url, headers=headers, timeout=10)
    if resp.status_code != 200:
        return pd.DataFrame()
    soup = BeautifulSoup(resp.text, "html.parser")
    table = soup.find("table", id="curr_table")
    if not table:
        return pd.DataFrame()
    data = []
    for row in table.find("tbody").find_all("tr"):
        cols = row.find_all("td")
        if len(cols) >= 4:
            tenor = cols[1].get_text(strip=True)
            bid = float(cols[2].get_text(strip=True).replace(",", ""))
            ask = float(cols[3].get_text(strip=True).replace(",", ""))
            mid = (bid + ask) / 2
            data.append({"Tenor": tenor, "Mid": mid})
    return pd.DataFrame(data)

@st.cache_data
def fetch_futures(pair_slug):
    # Investing.com's Futures contracts table is JavaScript-rendered—would need Selenium or API.
    return pd.DataFrame()

# Prepare slug for URL
slug = pair.lower().replace("/", "-")
df = fetch_forward_rates(slug) if data_type == "Forward Rates" else fetch_futures(slug)

if df.empty:
    st.error("No data available for this selection.")
else:
    st.subheader(f"{pair} {data_type}")
    st.dataframe(df)

    ts = df["Mid"]
    df_ts = pd.DataFrame({"ds": np.arange(len(ts)), "y": ts.values})

    if model_type == "ETS":
        model = ExponentialSmoothing(ts, trend="add", seasonal=None, initialization_method="estimated")
        fit = model.fit()
        forecast = fit.forecast(steps=forecast_steps)
        labels = list(df["Tenor"]) + [f"F+{i+1}" for i in range(forecast_steps)]
        values = np.concatenate([ts.values, forecast])

    elif model_type == "ARIMA":
        ar_model = ARIMA(ts, order=(1,1,1)).fit()
        forecast = ar_model.forecast(steps=forecast_steps)
        labels = list(df["Tenor"]) + [f"F+{i+1}" for i in range(forecast_steps)]
        values = np.concatenate([ts.values, forecast.values])

    elif model_type == "Prophet":
        m = Prophet(yearly_seasonality=False, weekly_seasonality=False, daily_seasonality=False)
        m.fit(df_ts)
        future = m.make_future_dataframe(periods=forecast_steps, freq="D")
        forecast_df = m.predict(future)
        labels = [f"t={d}" for d in future["ds"].astype(int)]
        values = forecast_df["yhat"].values

    fig = go.Figure(go.Scatter(x=labels, y=values, mode="lines+markers", name="Forecast"))
    fig.update_layout(
        title=f"{pair} Rate Forecast ({model_type})",
        xaxis_title="Tenor / Forecast Step",
        yaxis_title="Mid Rate",
        plot_bgcolor="#222",
        paper_bgcolor="#222",
        font=dict(color="white")
    )
    st.plotly_chart(fig, use_container_width=True)
