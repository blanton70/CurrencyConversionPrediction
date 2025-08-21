import streamlit as st
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import plotly.graph_objects as go
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Page setup
st.set_page_config(page_title="USD/MXN Forward Rate Forecast", layout="wide")
st.title("💱 USD/MXN Forward Rate Forecast — Investing.com")

headers = {"User-Agent": "Mozilla/5.0"}

@st.cache_data
def fetch_forward_data():
    url = "https://www.investing.com/currencies/usd-mxn-forward-rates"
    resp = requests.get(url, headers=headers, timeout=10)
    if resp.status_code != 200:
        return pd.DataFrame()

    soup = BeautifulSoup(resp.text, "html.parser")
    table = soup.find("table", {"id": "curr_table"})
    if not table:
        return pd.DataFrame()

    rows = table.find("tbody").find_all("tr")
    data = []
    for tr in rows:
        cells = tr.find_all("td")
        if len(cells) < 3:
            continue

        tenor = cells[1].get_text(strip=True)
        bid_text = cells[2].get_text(strip=True).replace(",", "")
        ask_text = cells[3].get_text(strip=True).replace(",", "")

        try:
            bid = float(bid_text)
            ask = float(ask_text)
        except ValueError:
            continue

        mid = (bid + ask) / 2
        data.append({"Tenor": tenor, "Bid": bid, "Ask": ask, "Mid": mid})

    return pd.DataFrame(data)

# Fetch and show data
df = fetch_forward_data()

if df.empty:
    st.error("Could not fetch forward rate data from Investing.com.")
else:
    st.write("### USD/MXN Forward Rates")
    st.dataframe(df)

    df = df.reset_index(drop=True)
    df["Index"] = np.arange(len(df))

    # Exponential Smoothing model
    model = ExponentialSmoothing(
        df["Mid"], trend="add", seasonal=None, initialization_method="estimated"
    )
    fit = model.fit()
    forecast_steps = 3
    forecast = fit.forecast(steps=forecast_steps)
    forecast_tenors = [f"Forecast +{i+1}" for i in range(forecast_steps)]

    # Plot the actual rates and forecast
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df["Tenor"], y=df["Mid"], mode="lines+markers", name="Observed",
        line=dict(color="lightblue")
    ))

    fig.add_trace(go.Scatter(
        x=forecast_tenors,
        y=forecast,
        mode="lines+markers+text",
        text=[f"{v:.4f}" for v in forecast],
        textposition="top center",
        marker=dict(color="red", size=10),
        name="Forecast"
    ))

    fig.update_layout(
        title="USD/MXN Forward Mid Rate with ETS Forecast",
        xaxis_title="Tenor",
        yaxis_title="Mid Rate",
        plot_bgcolor="#222222",
        paper_bgcolor="#222222",
        font=dict(color="white")
    )

    st.plotly_chart(fig, use_container_width=True)
