from typing import List, Dict, Any, Tuple
import pandas as pd, numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from data_sources import (
    CoinbaseMarketData, NewsScanner, TrendsScanner,
    ema, atr, donchian_high, volume_z
)

DEFAULT_WEIGHTS = {
    "news_peer_z": 0.25,
    "news_weighted": 0.15,
    "trends_z": 0.20,
    "volume_z": 0.20,
    "novelty_z": 0.20,
}

class HypeScanner:
    def __init__(self, assets: List[Dict[str, Any]], weights: Dict[str, float]=None):
        self.assets = assets or []
        self.weights = weights or DEFAULT_WEIGHTS
        self.cb = CoinbaseMarketData()
        self.news = NewsScanner()
        self.trends = TrendsScanner()

    def _market_df(self, pid: str, hours: int=240, granularity: int=3600):
        end = pd.Timestamp.utcnow(); start = end - pd.Timedelta(hours=hours)
        df = self.cb.candles(pid, granularity=granularity, start=start, end=end)
        if df.empty:
            return df
        df["ema20"] = ema(df["close"].astype(float), 20)
        df["ema200"] = ema(df["close"].astype(float), 200)
        df["atr14"] = atr(df, 14)
        df["donchian20"] = donchian_high(df, 20)
        return df

    def _regime_and_breadth(self, market_assets: List[Dict]) -> Tuple[bool, float]:
        # BTC regime
        try:
            btc = self._market_df("BTC-USD")
            btc_ok = (not btc.empty) and (btc["close"].iloc[-1] > btc["ema200"].iloc[-1])
        except Exception:
            btc_ok = True  # fail open; don't block everything on BTC fetch error
        # Breadth: % close > EMA20
        count = 0; above = 0
        for a in market_assets or []:
            try:
                df = self._market_df(a["product_id"])
                if df.empty: 
                    continue
                count += 1
                above += int(df["close"].iloc[-1] > df["ema20"].iloc[-1])
            except Exception:
                continue
        breadth = (above / count) if count else 0.0
        return btc_ok, breadth

    def scan(self, lookback_hours_news: int=24, hype_threshold: float=70.0, vol_z_threshold: float=1.5):
        # If no assets, return empty frames gracefully
        if not self.assets:
            return pd.DataFrame(), pd.DataFrame()

        # --- News & Trends ---
        news_agg, news_details = self.news.scan(self.assets, lookback_hours=lookback_hours_news)
        if news_agg.empty:
            news_agg = pd.DataFrame({
                "product_id":[a["product_id"] for a in self.assets],
                "mentions":[0]*len(self.assets),
                "avg_sentiment":[0.0]*len(self.assets),
                "headline_velocity":[0.0]*len(self.assets),
                "news_w_mentions":[0.0]*len(self.assets),
                "news_w_sentiment":[0.0]*len(self.assets)
            })
        # peer z's with safe denom
        def _zcol(df, col):
            s = df[col]
            std = s.std(ddof=1)
            return (s - s.mean()) / (std if std and std != 0 else 1.0)
        news_agg["news_peer_z"] = _zcol(news_agg, "mentions")
        news_agg["velocity_z"]  = _zcol(news_agg, "headline_velocity")
        news_agg["news_w_z"]    = _zcol(news_agg, "news_w_mentions")

        # Trends (ensure non-empty with zeros)
        trows=[]
        for a in self.assets:
            last, roc = self.trends.interest_24h_metrics(a.get("keywords",[a["name"]]))
            trows.append({"product_id": a["product_id"], "trends_last": last, "trends_roc": roc})
        trends_df = pd.DataFrame(trows)
        if trends_df.empty:
            trends_df = pd.DataFrame({
                "product_id":[a["product_id"] for a in self.assets],
                "trends_last":[0.0]*len(self.assets),
                "trends_roc":[0.0]*len(self.assets)
            })
        std = trends_df["trends_roc"].std(ddof=1)
        trends_df["trends_z"] = (trends_df["trends_roc"] - trends_df["trends_roc"].mean()) / (std if std and std != 0 else 1.0)

        # Novelty (TF-IDF); safe default if no headlines
        if not news_details.empty:
            dd = news_details.copy()
            docs = dd.groupby("product_id")["title"].apply(lambda s: " . ".join(s.tolist()))
            pids = docs.index.tolist()
            if len(pids) > 0:
                vect = TfidfVectorizer(max_features=1000, ngram_range=(1,2))
                X = vect.fit_transform(docs.values)
                nov = np.asarray(X.mean(axis=1)).ravel()
                novelty = pd.DataFrame({"product_id": pids, "novelty": nov})
            else:
                novelty = pd.DataFrame({"product_id":[a["product_id"] for a in self.assets], "novelty":[0.0]*len(self.assets)})
        else:
            novelty = pd.DataFrame({"product_id":[a["product_id"] for a in self.assets], "novelty":[0.0]*len(self.assets)})
        stdn = novelty["novelty"].std(ddof=1)
        novelty["novelty_z"] = (novelty["novelty"] - novelty["novelty"].mean()) / (stdn if stdn and stdn != 0 else 1.0)

        # Merge features
        feats = news_agg.merge(trends_df, on="product_id", how="outer").merge(
            novelty[["product_id","novelty_z"]], on="product_id", how="left"
        ).fillna(0.0)

        # Regime & Breadth filters
        btc_ok, breadth = self._regime_and_breadth(self.assets)
        regime_bad = (not btc_ok) and (breadth < 0.30)

        # --- Market confirms & HypeScore ---
        rows=[]
        for a in self.assets:
            pid=a["product_id"]; sym=a["symbol"]; name=a["name"]
            try:
                mdf = self._market_df(pid)
                if mdf.empty or len(mdf) < 50:
                    rows.append({"product_id": pid, "symbol": sym, "name": name, "eligible": False, "regime_ok": (not regime_bad)})
                    continue
                price=float(mdf["close"].iloc[-1]); v_z=float(volume_z(mdf, 240))
                ema20_ok = price > float(mdf["ema20"].iloc[-1])
                donchian_ok = price > float(mdf["donchian20"].iloc[-2]) if not pd.isna(mdf["donchian20"].iloc[-2]) else False

                n = feats.loc[feats["product_id"]==pid].squeeze()
                news_peer = max(float(n.get("news_peer_z",0.0)),0.0)
                trends_z  = max(float(n.get("trends_z",0.0)),0.0)
                novelty_z = max(float(n.get("novelty_z",0.0)),0.0)
                news_w    = max(float(n.get("news_w_z",0.0)),0.0)
                trends_roc = float(n.get("trends_roc",0.0))

                hype_raw = (
                    self.weights["news_peer_z"] * news_peer +
                    self.weights["news_weighted"]* news_w +
                    self.weights["trends_z"]     * trends_z +
                    self.weights["volume_z"]     * max(v_z,0.0) +
                    self.weights["novelty_z"]    * novelty_z
                )
                hype_score = float(np.clip(50 + 15*hype_raw, 0, 100))

                eligible = (hype_score >= hype_threshold) and ema20_ok and donchian_ok and (v_z >= vol_z_threshold)
                if regime_bad:
                    eligible = False

                rows.append({
                    "product_id": pid, "symbol": sym, "name": name, "price": price,
                    "ema20_ok": bool(ema20_ok), "donchian_breakout": bool(donchian_ok),
                    "vol_z": round(v_z,2), "trends_roc": round(trends_roc,3),
                    "news_mentions": int(n.get("mentions",0.0)),
                    "avg_sentiment": round(float(n.get("avg_sentiment",0.0)),3),
                    "novelty_z": round(novelty_z,2),
                    "hype_score": round(hype_score,2),
                    "eligible": bool(eligible),
                    "regime_ok": (not regime_bad)
                })
            except Exception as ex:
                rows.append({"product_id": pid, "symbol": sym, "name": name, "error": str(ex), "eligible": False, "regime_ok": (not regime_bad)})
        out = pd.DataFrame(rows).sort_values(["eligible","hype_score","vol_z"], ascending=[False,False,False]).reset_index(drop=True)
        return out, (news_details if 'news_details' in locals() else pd.DataFrame())
