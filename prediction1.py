import streamlit as st
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import plotly.graph_objects as go
from statsmodels.tsa.holtwinters import ExponentialSmoothing

st.set_page_config(page_title="FX Forward Rate Forecasting", layout="wide")
st.title("💱 FX Forward Rate Forecasting App")

# -------------------------------
# Currency pairs & Investing.com slugs
# -------------------------------
pairs = {
    "USD/MXN": "usd-mxn",
    "EUR/USD": "eur-usd",
    "GBP/USD": "gbp-usd",
    "USD/JPY": "usd-jpy",
    "AUD/USD": "aud-usd"
}

# --- Sidebar ---
pair_name = st.sidebar.selectbox("Select Currency Pair", list(pairs.keys()))
forecast_steps = st.sidebar.slider("Forecast steps", 1, 12, 3)

pair_slug = pairs[pair_name]
headers = {"User-Agent": "Mozilla/5.0"}

# -------------------------------
# Scraper function
# -------------------------------
@st.cache_data
def fetch_forward_rates(slug):
    url = f"https://www.investing.com/currencies/{slug}-forward-rates"
    try:
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")
        table = soup.find("table", {"id": "curr_table"})

        if not table:
            return pd.DataFrame()

        rows = table.find("tbody").find_all("tr")
        data = []

        for row in rows:
            cols = row.find_all("td")
            if len(cols) >= 4:
                tenor = cols[1].get_text(strip=True)
                bid = cols[2].get_text(strip=True).replace(",", "")
                ask = cols[3].get_text(strip=True).replace(",", "")

                try:
                    bid = float(bid)
                    ask = float(ask)
                    mid = (bid + ask) / 2
                    data.append({"Tenor": tenor, "Bid": bid, "Ask": ask, "Mid": mid})
                except:
                    continue

        return pd.DataFrame(data)

    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return pd.DataFrame()

# -------------------------------
# Fetch and process data
# -------------------------------
df = fetch_forward_rates(pair_slug)

if df.empty:
    st.error("❌ Could not fetch forward rate data.")
else:
    st.subheader(f"{pair_name} Forward Rate Table")
    st.dataframe(df)

    df = df.reset_index(drop=True)
    df["Index"] = np.arange(len(df))

    # -------------------------------
    # Forecasting with ETS
    # -------------------------------
    try:
        model = ExponentialSmoothing(
            df["Mid"],
            trend="add",
            seasonal=None,
            initialization_method="estimated"
        )
        fit = model.fit()
        forecast = fit.forecast(steps=forecast_steps)
        forecast_labels = [f"Forecast +{i+1}" for i in range(forecast_steps)]

        # -------------------------------
        # Plotting
        # -------------------------------
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=df["Tenor"], y=df["Mid"], mode="lines+markers", name="Observed",
            line=dict(color="cyan")
        ))

        fig.add_trace(go.Scatter(
            x=forecast_labels,
            y=forecast,
            mode="lines+markers+text",
            text=[f"{v:.4f}" for v in forecast],
            textposition="top center",
            marker=dict(color="red", size=8),
            name="Forecast"
        ))

        fig.update_layout(
            title=f"{pair_name} Mid Rate Forecast (Holt-Winters)",
            xaxis_title="Tenor",
            yaxis_title="Mid Rate",
            plot_bgcolor="#111",
            paper_bgcolor="#111",
            font=dict(color="white")
        )

        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Forecasting failed: {e}")
