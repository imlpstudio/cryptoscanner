import os, io, yaml, pandas as pd, numpy as np, streamlit as st
from scanner import HypeScanner

st.set_page_config(page_title="Hype Hunter", layout="wide")

st.title("🔥 Hype Hunter — Coinbase + News + Trends")
st.caption("Classroom paper-trading scanner. Computes a HypeScore from news, Google Trends, and market confirms.")

@st.cache_resource
def load_assets():
    with open("config/assets.yaml","r") as f:
        return yaml.safe_load(f)

assets = load_assets()

with st.sidebar:
    st.header("Scan Settings")
    lookback_news = st.slider("News lookback (hours)", 6, 72, 24, step=6)
    hype_threshold = st.slider("HypeScore threshold", 0, 100, 70, step=5)
    volz_thr = st.slider("Volume z-score threshold", 0.0, 3.0, 1.5, step=0.1)
    st.divider()
    st.subheader("Universe")
    enabled = st.multiselect("Products", [a["product_id"] for a in assets],
                             default=[a["product_id"] for a in assets])
    filtered_assets = [a for a in assets if a["product_id"] in enabled]
    st.divider()
    run_btn = st.button("🔎 Scan Now", use_container_width=True)

if run_btn:
    scan = HypeScanner(filtered_assets)
    with st.spinner("Scanning news, trends, and Coinbase..."):
        results = scan.scan(lookback_hours_news=lookback_news, hype_threshold=hype_threshold, vol_z_threshold=volz_thr)

    if results.empty:
        st.warning("No data returned. Try changing lookbacks or check your internet connection.")
    else:
        st.success(f"Scan complete — {results['eligible'].sum()} eligible of {len(results)}.")
        st.dataframe(results, use_container_width=True, height=420)

        # Download CSV
        csv = results.to_csv(index=False).encode("utf-8")
        st.download_button("Download results (CSV)", csv, file_name="hype_scan.csv", mime="text/csv")

        # Chart section
        sel = st.selectbox("Chart symbol", results["product_id"].tolist())
        if sel:
            # lightweight chart: ask scanner again for market df (avoid circular import)
            from data_sources import CoinbaseMarketData, ema
            cb = CoinbaseMarketData()
            mdf = cb.candles(sel, granularity=3600)  # 1h
            if not mdf.empty:
                mdf["ema20"] = ema(mdf["close"].astype(float), 20)
                import plotly.graph_objects as go
                fig = go.Figure()
                fig.add_trace(go.Candlestick(x=mdf["time"], open=mdf["open"], high=mdf["high"], low=mdf["low"], close=mdf["close"], name="Price"))
                fig.add_trace(go.Scatter(x=mdf["time"], y=mdf["ema20"], mode="lines", name="EMA20"))
                fig.update_layout(height=420, margin=dict(l=10,r=10,t=30,b=10))
                st.plotly_chart(fig, use_container_width=True)

st.markdown("""
---
**Tip:** If your network blocks the public Exchange endpoint, set `CB_API_KEY` / `CB_API_SECRET` environment variables for Coinbase Advanced Trade and restart the app.
""")
