# Hype Hunter — Streamlit (Coinbase + News + Google Trends)

Scan **Coinbase-tradable** crypto for **attention momentum** ("hype") and
basic price/volume confirmation. Designed for **classroom paper trading**.

## What it does
- Pulls **OHLCV** from Coinbase public endpoints (unauth) or v3 Advanced Trade (if keys supplied).
- Parses **RSS headlines** from major crypto sites and scores **mentions + sentiment**.
- Fetches **Google Trends** interest for coin keywords (last 7d hourly) and computes **24h ROC + z-scores**.
- Computes a combined **HypeScore** and checks **price > 20EMA** and **volume z-score**.
- UI to **scan now**, filter candidates, and **chart** a selected symbol with EMA and volume.
- Exports a **CSV** of today’s scan.

## Quickstart
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# first run downloads VADER lexicon
streamlit run app.py
```

Open the local URL Streamlit shows (usually http://localhost:8501).

## Optional: Coinbase v3 Advanced Trade (read-only keys)
Public market data should work unauthenticated using the **Exchange** REST endpoints.
If your environment requires the **v3 brokerage** endpoints, create keys in Coinbase and set:

```bash
export CB_API_KEY=<your_key>
export CB_API_SECRET=<your_secret>
export CB_API_PASSPHRASE=<if applicable>
```

The app will try v3 with auth first, then fall back to the public Exchange endpoints.

## Notes & Limits
- **Google Trends** values are *relative indices* (not absolute search counts). We compute deltas/z-scores only.
- **News** scores are based on RSS mention counts + headline sentiment (VADER) — a classroom proxy for “attention.”
- Network/API rate limits and changes can cause intermittent failures. The app will show warnings and continue.
- Nothing here is financial advice. For **paper trading only**.

## HypeScore (default weights)
```text
30% * z(news_mentions) * avg_sentiment
30% * z(google_trends_interest_24h_ROC)
20% * z(volume, vs 10-day 1h median)
20% * z(headline_velocity, 6-24h vs 7d)
```
Entry trigger (configurable): HypeScore ≥ threshold (default 70/100) **AND** price > 20EMA **AND** volume z ≥ +1.5

## Customize Universe
Edit `config/assets.yaml` — it maps product ids to names + keyword hints for news/Trends.

## Export
Use the **Download** button in the UI to export scan results as CSV.


## Strategies & Filters

### Hype (attention momentum — not mathy)
- **Signals (score):** news peer-z, domain-weighted mentions (sentiment-gated), Google Trends 24h ROC z, TF-IDF novelty, volume z  
- **Confirms (AND):** price > 20EMA, Donchian(20) breakout, volume z ≥ 1.5  
- **Default trigger:** HypeScore ≥ 70  
- **Regime:** BTC 1h > 200EMA and market breadth ≥ 30%

### Whale (follow the money — flow/tape)
- **Signals (score):** big-trade notional z (≥ \$250k prints), L2 book imbalance z, volume z  
- **Confirms (AND):** price > 50EMA, Donchian(20) breakout, volume z ≥ 1.2  
- **Default trigger:** WhaleScore ≥ 70  
- **Regime:** BTC 1h > 200EMA

### Quant (math/trend — pure price/stat)
- **Signals (score):** EMA(20–50) trend strength, Donchian(20) breakout, volume health  
- **Confirms (AND):** price > 200EMA, Donchian(20) breakout, volume z ≥ 1.0  
- **Default trigger:** QuantScore ≥ 70–75

### Universe (shared)
- Coinbase USD/USDC pairs.  
- Require **30-day median hourly notional** ≥ \$300k–\$500k.  
- Cap universe (100–150) to scan many coins but avoid illiquid traps.

