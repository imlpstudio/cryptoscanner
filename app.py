import os, io, yaml, pandas as pd, numpy as np, streamlit as st
from universe import UniverseBuilder
from strategies.hype import HypeStrategy
from strategies.whale import WhaleStrategy
from strategies.quant import QuantStrategy
from data_sources import CoinbaseMarketData, ema

st.set_page_config(page_title="Crypto Scanner — Resilient Recs + Confidence", layout="wide")
st.title("🚀 Crypto Multi-Strategy Scanner")
st.caption("Hype (attention), Whale (flow), Quant (trend) — with auto-fallback universe and confidence scoring.")

PRESET_PATH = "config/presets.yaml"

@st.cache_resource
def load_presets():
    if not os.path.exists(PRESET_PATH):
        return {"presets": {}}
    with open(PRESET_PATH, "r") as f:
        data = yaml.safe_load(f) or {}
        if "presets" not in data:
            data = {"presets": {}}
        return data

def _build_universe_resilient(max_assets:int, min_notional:float):
    ub = UniverseBuilder()
    attempts = [1.0, 0.5, 0.25, 0.1]
    for m in attempts:
        assets = ub.build(max_assets=max_assets, min_median_notional_usd=min_notional*m)
        if len(assets) >= max(20, int(0.3*max_assets)):
            return assets, (min_notional*m), m
    # last resort: try static YAML
    try:
        with open("config/assets.yaml","r") as f:
            assets = yaml.safe_load(f) or []
            for a in assets:
                a.setdefault("median_notional", 0.0)
        if assets:
            return assets, 0.0, 0.0
    except Exception:
        pass
    return [], min_notional, 1.0

# ------- Sidebar -------
PRESETS = load_presets()["presets"]
with st.sidebar:
    st.header("Universe")
    dynamic = st.toggle("Use Dynamic Coinbase Universe", value=True)
    max_assets = st.slider("Max assets", 10, 200, 120, step=10)
    min_notional = st.number_input("Min median hourly notional (USD)", 1e5, 2e7, value=300000.0, step=100000.0, format="%.0f")

    st.header("Strategy")
    strategy = st.selectbox("Choose strategy", ["Hype", "Whale", "Quant"])

    st.header("Parameters")
    lookback_news = st.slider("News lookback (Hype)", 6, 72, 24, step=6)
    threshold = st.slider("Score threshold", 0, 100, 70, step=5)
    volz_thr = st.slider("Volume z threshold", 0.0, 3.0, 1.5, step=0.1)
    big_trade_usd = st.number_input("Whale: min 'big trade' (USD)", 50_000.0, 5_000_000.0, 250_000.0, step=50_000.0, format="%.0f")

    run_btn = st.button("🔎 Scan Now", use_container_width=True)

# ------- Universe -------
if dynamic:
    with st.spinner("Building dynamic universe..."):
        assets, used_notional, relax_factor = _build_universe_resilient(max_assets, min_notional)
        if relax_factor < 1.0:
            st.info(f"Universe auto-relaxed: min notional lowered to **${used_notional:,.0f}** (x{relax_factor:.2f}).")
else:
    try:
        with open("config/assets.yaml","r") as f:
            assets = yaml.safe_load(f) or []
            for a in assets: a.setdefault("median_notional", 0.0)
    except Exception:
        assets = []

if not assets:
    st.error("Could not build any universe (even after relaxing). Check network or try non-dynamic mode.")
    st.stop()

# Always show full universe considered
with st.expander(f"📃 Universe candidates ({len(assets)}) — click to view", expanded=False):
    dfu = pd.DataFrame(assets)
    cols = [c for c in ["symbol","product_id","median_notional"] if c in dfu.columns]
    st.dataframe(dfu[cols] if cols else dfu, use_container_width=True, height=300)

# ------- Scan -------
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
        st.warning("No data returned. Try again or adjust universe.")
        st.stop()

    eligible_ct = int(results["eligible"].sum())
    st.success(f"Scan complete — **{eligible_ct}** strict candidates of **{len(results)}**. Strategy: **{strategy}**")

    # Recommended (strict)
    recs = results[results["eligible"]].sort_values(["confidence","score"], ascending=False).head(12)
    if not recs.empty:
        st.subheader("✅ Recommended (strict)")
        cols = st.columns(min(4, len(recs)))
        for i, (_, r) in enumerate(recs.iterrows()):
            with cols[i % len(cols)]:
                st.markdown(f"### {r['symbol']} — Score {r['score']}")
                st.metric("Confidence", f"{r.get('confidence',0)}% ({r.get('confidence_label','')})")
                st.write(
                    f"- **Price**: {r.get('price',0):.4f}\n"
                    f"- **Vol z**: {r.get('vol_z',0)}\n"
                    f"- **Explain**: {r.get('explain','')}"
                )
                flags=[]
                for k,label in [
                    ("ema20_ok","EMA20"), ("ema50_ok","EMA50"), ("above_200","Above EMA200"),
                    ("donchian_breakout","Donchian20"), ("regime_ok","Regime")
                ]:
                    if k in r:
                        flags.append(f"{label} {'✅' if r[k] else '❌'}")
                if flags: st.caption(" | ".join(flags))

    # If no strict picks, provide relaxed suggestions (top by score)
    if recs.empty:
        fallback = results.sort_values(["score","vol_z"], ascending=False).head(8)
        st.warning("No coins met *all* strict gates. Showing **best available (relaxed)** suggestions.")
        cols = st.columns(min(4, len(fallback)))
        for i, (_, r) in enumerate(fallback.iterrows()):
            with cols[i % len(cols)]:
                st.markdown(f"### {r['symbol']} — Score {r['score']}")
                st.metric("Confidence", f"{r.get('confidence',0)}% ({r.get('confidence_label','')})")
                st.caption(r.get("explain",""))

    st.divider()
    st.subheader("All results")
    st.dataframe(results, use_container_width=True, height=420)

    csv = results.to_csv(index=False).encode("utf-8")
    st.download_button("Download results (CSV)", csv, file_name=f"{strategy.lower()}_scan.csv", mime="text/csv")

    # Why this coin
    st.divider()
    sel = st.selectbox("Why this coin?", results["product_id"].tolist())
    if sel:
        r = results[results["product_id"]==sel].iloc[0]
        st.markdown(f"### {r['symbol']} — Why")
        b1, b2, b3, b4 = st.columns(4)
        b1.metric("Score", r.get("score",0))
        b2.metric("Confidence", f"{r.get('confidence',0)}%")
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

        # Details panel (headlines or whale trades)
        # Available from previous branches; shown if present
        # (kept simple to focus on confidence feature)
