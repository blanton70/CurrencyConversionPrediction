import streamlit as st
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import plotly.graph_objects as go
from statsmodels.tsa.holtwinters import ExponentialSmoothing

st.set_page_config(page_title="USD/MXN Forward Rate Forecast (Investing.com)", layout="wide")
st.title("💱 USD/MXN Forward Rate Forecast (ETS Model) — Investing.com")

headers = {"User-Agent": "Mozilla/5.0"}

@st.cache_data
def fetch_forward_data():
    url = "https://www.investing.com/currencies/usd-mxn-forward-rates"
    resp = requests.get(url, headers=headers, timeout=10)
    if resp.status_code != 200:
        return pd.DataFrame()

    soup = BeautifulSoup(resp.text, "html.parser")
    tables = soup.find_all("table")
    forward_table = None

    for table in tables:
        th_texts = [th.get_text(strip=True).lower() for th in table.find_all("th")]
        if {"bid", "ask", "chg."}.issubset(th_texts):
            forward_table = table
            break

    if not forward_table:
        return pd.DataFrame()

    rows = forward_table.find("tbody").find_all("tr")
    data = []
    for tr in rows:
        cols = [td.get_text(strip=True) for td in tr.find_all("td")]
        if len(cols) >= 3:
            tenor, bid, ask = cols[0], cols[1], cols[2]
            try:
                bid, ask = float(bid.replace(",", "")), float(ask.replace(",", ""))
            except:
                continue
            mid = (bid + ask) / 2
            data.append({"Tenor": tenor, "Bid": bid, "Ask": ask, "Mid": mid})

    return pd.DataFrame(data)

df = fetch_forward_data()

if df.empty:
    st.error("Could not fetch forward rate data from Investing.com.")
else:
    st.write("### USD/MXN Forward Rates")
    st.dataframe(df)

    df = df.reset_index(drop=True)
    df["Index"] = np.arange(len(df))

    try:
        model = ExponentialSmoothing(df["Mid"], trend="add", seasonal=None, initialization_method="estimated")
        fit = model.fit()
        forecast_steps = 3
        forecast = fit.forecast(steps=forecast_steps)
        forecast_tenors = [f"Forecast {i+1}" for i in range(forecast_steps)]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df["Tenor"], y=df["Mid"],
            mode="lines+markers", name="Observed", line=dict(color="lightblue")
        ))
        fig.add_trace(go.Scatter(
            x=forecast_tenors, y=forecast,
            mode="markers+lines+text", name="Forecast",
            marker=dict(color="red", size=10),
            text=[f"{v:.4f}" for v in forecast], textposition="top center"
        ))
        fig.update_layout(
            title="USD/MXN Forward Mid Rate with ETS Forecast",
            xaxis_title="Tenor",
            yaxis_title="Mid Rate",
            plot_bgcolor="#222",
            paper_bgcolor="#222",
            font=dict(color="white"),
        )
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Forecasting failed: {e}")
