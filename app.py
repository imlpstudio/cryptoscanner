import os, io, yaml, pandas as pd, numpy as np, streamlit as st
from universe import UniverseBuilder
from strategies.hype import HypeStrategy
from strategies.whale import WhaleStrategy
from strategies.quant import QuantStrategy
from data_sources import CoinbaseMarketData, ema

st.set_page_config(page_title="Hype Hunter — Multi Strategy", layout="wide")
st.title("🚀 Crypto Multi-Strategy Scanner (Coinbase)")
st.caption("Hype (attention), Whale (flow), Quant (trend). With regime filters and 'why' explanations.")

@st.cache_resource
def load_assets_yaml():
    try:
        with open("config/assets.yaml","r") as f:
            return yaml.safe_load(f)
    except Exception:
        return []

with st.sidebar:
    st.header("Universe")
    dynamic = st.toggle("Use Dynamic Coinbase Universe", value=True)
    max_assets = st.slider("Max assets", 10, 80, 40, step=5)
    min_notional = st.number_input("Min median hourly notional (USD)", 1e5, 2e7, value=5e5, step=1e5, format="%.0f")

    st.header("Strategy")
    strategy = st.selectbox("Choose strategy", ["Hype", "Whale", "Quant"])
    st.write("---")

    st.header("Parameters")
    lookback_news = st.slider("News lookback (hours) [Hype only]", 6, 72, 24, step=6)
    threshold = st.slider("Score threshold", 0, 100, 70, step=5)
    volz_thr = st.slider("Volume z threshold", 0.0, 3.0, 1.5, step=0.1)
    big_trade_usd = st.number_input("Whale: min 'big trade' notional (USD)", 50_000.0, 5_000_000.0, 250_000.0, step=50_000.0, format="%.0f")

    run_btn = st.button("🔎 Scan Now", use_container_width=True)

# Universe build
if dynamic:
    with st.spinner("Building dynamic Coinbase universe..."):
        ub = UniverseBuilder()
        assets = ub.build(max_assets=max_assets, min_median_notional_usd=min_notional)
else:
    assets = load_assets_yaml()

if not assets:
    st.error("Universe is empty. Lower 'Min median hourly notional' or disable Dynamic.")
    st.stop()

if run_btn:
    if strategy == "Hype":
        strat = HypeStrategy(assets)
        with st.spinner("Scanning (Hype)…"):
            results, details = strat.scan(lookback_hours_news=lookback_news, threshold=threshold, vol_z_threshold=volz_thr)
    elif strategy == "Whale":
        strat = WhaleStrategy(assets, big_trade_usd=big_trade_usd)
        with st.spinner("Scanning (Whale)…"):
            results, details = strat.scan(threshold=threshold, vol_z_threshold=volz_thr)
    else:
        strat = QuantStrategy(assets)
        with st.spinner("Scanning (Quant)…"):
            results, details = strat.scan(threshold=threshold, vol_z_threshold=volz_thr)

    if results.empty:
        st.warning("No candidates met the filters. Try lowering thresholds or expanding the universe.")
        st.stop()

    eligible_ct = int(results["eligible"].sum())
    st.success(f"Scan complete — **{eligible_ct}** eligible of **{len(results)}**. Strategy: **{strategy}**")

    # Recommended cards
    recs = results[results["eligible"]].head(12)
    if not recs.empty:
        st.subheader("✅ Recommended")
        cols = st.columns(min(4, len(recs)))
        for i, (_, r) in enumerate(recs.iterrows()):
            with cols[i % len(cols)]:
                st.markdown(f"### {r['symbol']} — Score {r['score']}")
                st.write(
                    f"- **Price**: {r['price']:.4f}\n"
                    f"- **Vol z**: {r.get('vol_z',0)}\n"
                    f"- **Explain**: {r.get('explain','')}"
                )
                # strategy-specific flags
                flags=[]
                if 'ema20_ok' in r:    flags.append(f"EMA20 {'✅' if r['ema20_ok'] else '❌'}")
                if 'ema50_ok' in r:    flags.append(f"EMA50 {'✅' if r['ema50_ok'] else '❌'}")
                if 'above_200' in r:   flags.append(f"Above EMA200 {'✅' if r['above_200'] else '❌'}")
                if 'donchian_breakout' in r: flags.append(f"Donchian20 {'✅' if r['donchian_breakout'] else '❌'}")
                st.caption(" | ".join(flags))

    st.divider()
    st.subheader("All results")
    st.dataframe(results, use_container_width=True, height=420)

    # Download
    csv = results.to_csv(index=False).encode("utf-8")
    st.download_button("Download results (CSV)", csv, file_name=f"{strategy.lower()}_scan.csv", mime="text/csv")

    # Why this pick — select a symbol
    st.divider()
    sel = st.selectbox("Why this coin?", results["product_id"].tolist())
    if sel:
        r = results[results["product_id"]==sel].iloc[0]
        st.markdown(f"### {r['symbol']} — Why")
        b1, b2, b3, b4 = st.columns(4)
        b1.metric("Score", r["score"])
        b2.metric("Vol z", r.get("vol_z",0))
        if strategy=="Hype":
            b3.metric("Trends ROC", r.get("trends_roc",0))
            b4.metric("Novelty z", r.get("novelty_z",0))
        elif strategy=="Quant":
            b3.metric("Trend Str", r.get("trend_strength",0))
            b4.metric("Above 200", "Yes" if r.get("above_200",False) else "No")
        else:
            b3.metric("EMA50", "Yes" if r.get("ema50_ok",False) else "No")
            b4.metric("Donchian", "Yes" if r.get("donchian_breakout",False) else "No")

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

        # Headlines or Whale trades (details)
        if strategy=="Hype" and 'title' in (details.columns if hasattr(details, "columns") else []):
            st.markdown("#### Top headlines")
            dd = details[details["product_id"]==sel].sort_values("published", ascending=False).head(6)
            if dd.empty:
                st.write("_No matching headlines in lookback._")
            else:
                for _, row in dd.iterrows():
                    ts = row["published"].strftime("%Y-%m-%d %H:%M UTC")
                    st.markdown(f"- [{row['title']}]({row.get('url','')}) — {ts}  ·  {row.get('domain','')}")
        elif strategy=="Whale" and hasattr(details, "empty") and not details.empty:
            st.markdown("#### Biggest recent trades")
            dd = details[details["product_id"]==sel].head(6)
            if dd.empty:
                st.write("_No large trades captured._")
            else:
                st.dataframe(dd[["time","side","price","size","notional"]], use_container_width=True, height=240)
