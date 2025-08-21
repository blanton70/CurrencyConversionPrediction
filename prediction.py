import streamlit as st
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import plotly.express as px
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="FX Forward Rate Prediction", layout="wide")
st.title("💱 FX Forward Rate Prediction App")

# --- Currency pair map ---
pairs = {
    "USD/MYR": "usd-myr",
    "EUR/USD": "eur-usd",
    "USD/JPY": "usd-jpy",
    "GBP/USD": "gbp-usd",
    "AUD/USD": "aud-usd",
}

pair_name = st.selectbox("Select Currency Pair", list(pairs.keys()))
pair_slug = pairs[pair_name]

# --- Load headers securely from secrets ---
try:
    headers = st.secrets["headers"]
except Exception:
    headers = {"User-Agent": "Mozilla/5.0"}  # Fallback

# --- Scraping function ---
@st.cache_data
def fetch_forward_data(pair_slug):
    url = f"https://www.fxempire.com/currencies/{pair_slug}/forward-rates"
    resp = requests.get(url, headers=headers, timeout=10)
    soup = BeautifulSoup(resp.text, "html.parser")
    table = soup.find("table", class_="empire-table")

    if not table:
        return pd.DataFrame()

    rows = table.find("tbody").find_all("tr")
    data = []
    for tr in rows:
        cols = [td.get_text(strip=True) for td in tr.find_all("td")]
        if len(cols) >= 5:
            exp, bid, ask, mid, points = cols[:5]
            data.append({
                "Expiration": exp,
                "Bid": pd.to_numeric(bid, errors="coerce"),
                "Ask": pd.to_numeric(ask, errors="coerce"),
                "Mid": pd.to_numeric(mid, errors="coerce"),
                "Points": pd.to_numeric(points, errors="coerce"),
            })
    return pd.DataFrame(data)

# --- Fetch and display data ---
df = fetch_forward_data(pair_slug)

if df.empty:
    st.error("Unable to fetch forward rate data for this currency pair.")
else:
    st.write(f"### Forward Rates for {pair_name}")
    st.dataframe(df)

    # --- Linear prediction ---
    df = df.dropna(subset=["Mid"])
    df["Index"] = np.arange(len(df))
    model = LinearRegression()
    model.fit(df[["Index"]], df["Mid"])
    next_idx = len(df)
    prediction = model.predict([[next_idx]])[0]

    st.markdown(f"### 📈 Predicted next forward MID rate: **{prediction:.5f}**")

    # --- Chart ---
    fig = px.line(df, x="Expiration", y="Mid", title="Forward Mid Rate Curve", markers=True)
    fig.add_scatter(x=[f"Future({next_idx})"], y=[prediction], mode='markers+text',
                    text=["Predicted"], name="Prediction", marker=dict(size=10, color="red"))

    fig.update_layout(
        plot_bgcolor='#222',
        paper_bgcolor='#222',
        font=dict(color='white'),
        xaxis=dict(title='Expiration'),
        yaxis=dict(title='Mid Rate'),
    )
    st.plotly_chart(fig, use_container_width=True)
