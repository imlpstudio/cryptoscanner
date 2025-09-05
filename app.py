import os, io, yaml, pandas as pd, numpy as np, streamlit as st
from scanner import HypeScanner
from universe import UniverseBuilder
from data_sources import CoinbaseMarketData, ema

st.set_page_config(page_title="Hype Hunter Pro", layout="wide")
st.title("🔥 Hype Hunter Pro — Coinbase + News + Trends")
st.caption("Find attention-driven swing candidates with regime filters and rich 'why' explanations.")

@st.cache_resource
def load_assets_yaml():
    try:
        with open("config/assets.yaml","r") as f:
            return yaml.safe_load(f)
    except Exception:
        return []

with st.sidebar:
    st.header("Scan Settings")
    dynamic = st.toggle("Use Dynamic Coinbase Universe", value=True)
    max_assets = st.slider("Max assets", 10, 80, 40, step=5)
    min_notional = st.number_input("Min median hourly notional (USD)", 1e5, 2e7, value=5e5, step=1e5, format="%.0f")
    lookback_news = st.slider("News lookback (hours)", 6, 72, 24, step=6)
    hype_threshold = st.slider("HypeScore threshold", 0, 100, 70, step=5)
    volz_thr = st.slider("Volume z threshold", 0.0, 3.0, 1.5, step=0.1)
    st.divider()
    run_btn = st.button("🔎 Scan Now", use_container_width=True)

# Universe
if dynamic:
    with st.spinner("Building dynamic Coinbase universe..."):
        ub = UniverseBuilder()
        assets = ub.build(max_assets=max_assets, min_median_notional_usd=min_notional)
else:
    assets = load_assets_yaml()

if run_btn:
    scan = HypeScanner(assets)
    with st.spinner("Scanning news, trends, and market confirms..."):
        results, details = scan.scan(lookback_hours_news=lookback_news, hype_threshold=hype_threshold, vol_z_threshold=volz_thr)

    if results.empty:
        st.warning("No data returned or all filtered. Try lowering thresholds or disabling 'Dynamic universe'.")
        st.stop()

    # Summary header
    eligible_ct = int(results["eligible"].sum())
    st.success(f"Scan complete — **{eligible_ct}** eligible out of **{len(results)}** (regime OK: **{bool(results['regime_ok'].all())}**)")

    # Recommended picks (cards)
    recs = results[results["eligible"]].head(10)
    if not recs.empty:
        st.subheader("✅ Recommended (meets all triggers)")
        cols = st.columns(min(4, len(recs)))
        for i, (_, r) in enumerate(recs.iterrows()):
            with cols[i % len(cols)]:
                st.markdown(f"### {r['symbol']} — {r['hype_score']}")
                st.metric("HypeScore", r["hype_score"], help="0–100 attention composite")
                st.write(
                    f"- **Price**: {r['price']:.4f}\n"
                    f"- **Vol z**: {r['vol_z']}\n"
                    f"- **Trends ROC**: {r['trends_roc']}\n"
                    f"- **Novelty z**: {r['novelty_z']}\n"
                    f"- **EMA20**: {'✅' if r['ema20_ok'] else '❌'} | **Donchian20**: {'✅' if r['donchian_breakout'] else '❌'}"
                )
                st.caption("Entry idea: HypeScore≥th, price>EMA20, Donchian breakout, vol z≥th. SL 2×ATR; trail 2.5×ATR; time-stop 72h.")

    st.divider()
    st.subheader("All results")
    st.dataframe(results, use_container_width=True, height=420)

    # Download
    csv = results.to_csv(index=False).encode("utf-8")
    st.download_button("Download results (CSV)", csv, file_name="hype_scan.csv", mime="text/csv")

    # Why this pick — select a symbol
    st.divider()
    sel = st.selectbox("Why this coin?", results["product_id"].tolist())
    if sel:
        r = results[results["product_id"]==sel].iloc[0]
        st.markdown(f"### {r['symbol']} — Why it’s recommended")
        b1, b2, b3, b4 = st.columns(4)
        b1.metric("HypeScore", r["hype_score"])
        b2.metric("Vol z", r["vol_z"])
        b3.metric("Trends ROC", r["trends_roc"])
        b4.metric("Novelty z", r["novelty_z"])

        # Chart
        cb = CoinbaseMarketData()
        mdf = cb.candles(sel, granularity=3600)
        if not mdf.empty:
            mdf["ema20"] = ema(mdf["close"].astype(float), 20)
            import plotly.graph_objects as go
            fig = go.Figure()
            fig.add_trace(go.Candlestick(x=mdf["time"], open=mdf["open"], high=mdf["high"], low=mdf["low"], close=mdf["close"], name="Price"))
            fig.add_trace(go.Scatter(x=mdf["time"], y=mdf["ema20"], mode="lines", name="EMA20"))
            fig.update_layout(height=420, margin=dict(l=10,r=10,t=30,b=10))
            st.plotly_chart(fig, use_container_width=True)

        # Top headlines (linked)
        st.markdown("#### Top recent headlines")
        dd = details[details["product_id"]==sel].sort_values("published", ascending=False).head(6)
        if dd.empty:
            st.write("_No matching headlines in lookback._")
        else:
            for _, row in dd.iterrows():
                ts = row["published"].strftime("%Y-%m-%d %H:%M UTC")
                st.markdown(f"- [{row['title']}]({row['url']}) — {ts}  ·  {row['domain']}  ·  sent: {row['sentiment']:.2f}")
