import os, io, yaml, pandas as pd, numpy as np, streamlit as st
from universe import UniverseBuilder
from strategies.hype import HypeStrategy
from strategies.whale import WhaleStrategy
from strategies.quant import QuantStrategy
from data_sources import CoinbaseMarketData, ema

st.set_page_config(page_title="Crypto Scanner — Presets + Wide Universe", layout="wide")
st.title("🧪 Crypto Multi-Strategy Scanner — Presets & Wide Universe")

# ==== Strategy rules help panel ====
help_md = """
### Hype (attention momentum — not mathy)
**Goal:** Catch coins with sudden *attention*; use price/volume only as confirmations.

- **Pre-conditions:** BTC 1h > 200EMA **and** breadth ≥ 30% (share above 20EMA)
- **Signals (score):** news peer-z, domain-weighted mentions (sentiment-gated), Google Trends 24h ROC z, TF-IDF novelty, volume z
- **Confirms (AND):** price > 20EMA, Donchian(20) breakout, volume z ≥ 1.5
- **Default trigger:** HypeScore ≥ 70

### Whale (follow the money — flow/tape)
**Goal:** Ride unusual **large prints** and **order-book pressure**.

- **Pre-conditions:** BTC regime OK (1h > 200EMA)
- **Signals (score):** big-trade notional z (≥ \$250k prints), L2 book imbalance z, volume z
- **Confirms (AND):** price > 50EMA, Donchian(20) breakout, volume z ≥ 1.2
- **Default trigger:** WhaleScore ≥ 70

### Quant (math/trend — pure price/stat)
**Goal:** Hold robust trends; no attention or flow data.

- **Signals (score):** EMA(20–50) trend strength, Donchian(20) breakout, volume health
- **Confirms (AND):** price > 200EMA, Donchian(20) breakout, volume z ≥ 1.0
- **Default trigger:** QuantScore ≥ 70–75

**Universe (shared):** Coinbase USD/USDC pairs; keep coins with 30-day **median hourly notional** ≥ \$300k–\$500k; cap to 100–150.
"""

try:
    import streamlit as st  # ensure we only add this in the app context
    with st.expander("📘 Strategy rules & filters", expanded=False):
        st.markdown(help_md)
except Exception:
    pass
# ==== end help panel ====
st.caption("Compare saved filter sets; scan many Coinbase pairs; see exactly which coins were considered every run.")

PRESET_PATH = "config/presets.yaml"

# -------- Presets I/O --------
@st.cache_resource
def load_presets():
    if not os.path.exists(PRESET_PATH):
        return {"presets": {}}
    with open(PRESET_PATH, "r") as f:
        data = yaml.safe_load(f) or {}
        if "presets" not in data:
            data = {"presets": {}}
        return data

def save_presets(data):
    os.makedirs(os.path.dirname(PRESET_PATH), exist_ok=True)
    with open(PRESET_PATH, "w") as f:
        yaml.safe_dump(data, f, sort_keys=True)

def apply_preset_to_session(p):
    # set default values into session_state so widgets pick them up
    for k, v in p.items():
        st.session_state[k] = v

# initial preset load
PRESETS = load_presets()["presets"]
if "params" not in st.session_state:
    # choose first preset or a minimal default
    first = next(iter(PRESETS.values()), {
        "strategy": "Hype",
        "dynamic": True,
        "max_assets": 100,
        "min_notional": 300000,
        "lookback_news": 24,
        "threshold": 70,
        "vol_z_threshold": 1.5,
        "big_trade_usd": 250000,
    })
    st.session_state.params = first
    apply_preset_to_session(first)

# -------- Sidebar UI --------
with st.sidebar:
    st.header("Presets")
    preset_names = list(PRESETS.keys())
    selected_preset = st.selectbox("Select preset", options=preset_names or ["(none)"])
    colp = st.columns(2)
    with colp[0]:
        if st.button("Load preset", use_container_width=True) and selected_preset in PRESETS:
            apply_preset_to_session(PRESETS[selected_preset])
            st.session_state.params = PRESETS[selected_preset]
            st.rerun()
    with colp[1]:
        new_name = st.text_input("Preset name", value=selected_preset if selected_preset else "MyPreset")
    cols = st.columns(2)
    with cols[0]:
        if st.button("Save as NEW", use_container_width=True):
            data = load_presets()
            data["presets"][new_name] = dict(st.session_state)
            save_presets(data)
            st.success(f"Saved new preset: {new_name}")
    with cols[1]:
        if st.button("Overwrite selected", use_container_width=True) and selected_preset:
            data = load_presets()
            data["presets"][selected_preset] = dict(st.session_state)
            save_presets(data)
            st.success(f"Overwrote preset: {selected_preset}")

    st.write("---")
    st.header("Universe")
    dynamic = st.toggle("Use Dynamic Coinbase Universe", value=st.session_state.get("dynamic", True), key="dynamic")
    max_assets = st.slider("Max assets", 10, 200, int(st.session_state.get("max_assets", 100)), step=10, key="max_assets")
    min_notional = st.number_input("Min median hourly notional (USD)", 1e5, 2e7,
                                   value=float(st.session_state.get("min_notional", 300000)),
                                   step=100000.0, format="%.0f", key="min_notional")

    st.header("Strategy & Filters")
    strategy = st.selectbox("Strategy", ["Hype","Whale","Quant"], index=["Hype","Whale","Quant"].index(st.session_state.get("strategy","Hype")), key="strategy")
    lookback_news = st.slider("News lookback (Hype)", 6, 72, int(st.session_state.get("lookback_news", 24)), step=6, key="lookback_news")
    threshold = st.slider("Score threshold", 0, 100, int(st.session_state.get("threshold", 70)), step=5, key="threshold")
    volz_thr = st.slider("Volume z threshold", 0.0, 3.0, float(st.session_state.get("vol_z_threshold", 1.5)), step=0.1, key="vol_z_threshold")
    big_trade_usd = st.number_input("Whale: min 'big trade' notional (USD)", 50_000.0, 5_000_000.0,
                                    value=float(st.session_state.get("big_trade_usd", 250000)),
                                    step=50_000.0, format="%.0f", key="big_trade_usd")

    run_btn = st.button("🔎 Scan Now", use_container_width=True)
    st.caption("Tip: Save your current settings as a preset, then re-run later to compare outputs.")

# -------- Build Universe --------
if dynamic:
    with st.spinner("Building dynamic Coinbase universe..."):
        ub = UniverseBuilder()
        assets = ub.build(max_assets=max_assets, min_median_notional_usd=min_notional)
else:
    # static fallback from assets.yaml
    try:
        with open("config/assets.yaml","r") as f:
            assets = yaml.safe_load(f) or []
            # enrich with placeholder notional so we can list
            for a in assets:
                a.setdefault("median_notional", 0.0)
    except Exception:
        assets = []

if not assets:
    st.error("Universe is empty. Lower 'Min median hourly notional' or enable Dynamic.")
    st.stop()

# Show full universe (coins considered)
with st.expander(f"📃 Universe candidates ({len(assets)}) — click to view"):
    dfu = pd.DataFrame(assets)
    cols = [c for c in ["symbol","product_id","median_notional"] if c in dfu.columns]
    st.dataframe(dfu[cols] if cols else dfu, use_container_width=True, height=300)

# -------- Run Strategy --------
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

    # Recommended
    recs = results[results["eligible"]].head(12)
    if not recs.empty:
        st.subheader("✅ Recommended (meets all triggers)")
        cols = st.columns(min(4, len(recs)))
        for i, (_, r) in enumerate(recs.iterrows()):
            with cols[i % len(cols)]:
                st.markdown(f"### {r['symbol']} — Score {r['score']}")
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

    st.divider()
    st.subheader("All results")
    st.dataframe(results, use_container_width=True, height=420)

    # Download
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

        # Strategy-specific details
        if strategy=="Hype" and hasattr(details, "columns") and 'title' in details.columns:
            st.markdown("#### Top headlines")
            dd = details[details["product_id"]==sel].sort_values("published", ascending=False).head(8)
            if dd.empty:
                st.write("_No matching headlines in lookback._")
            else:
                for _, row in dd.iterrows():
                    ts = row["published"].strftime("%Y-%m-%d %H:%M UTC")
                    url = row.get('url','')
                    dom = row.get('domain','')
                    st.markdown(f"- [{row['title']}]({url}) — {ts} · {dom}")
        elif strategy=="Whale" and hasattr(details, "empty") and not details.empty:
            st.markdown("#### Biggest recent trades")
            dd = details[details["product_id"]==sel].head(8)
            if dd.empty:
                st.write("_No large trades captured._")
            else:
                st.dataframe(dd[["time","side","price","size","notional"]], use_container_width=True, height=280)
