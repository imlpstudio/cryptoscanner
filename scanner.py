from typing import List, Dict, Any
import pandas as pd
import numpy as np

from data_sources import CoinbaseMarketData, NewsScanner, TrendsScanner, ema, atr, volume_z

DEFAULT_WEIGHTS = {
    "news_z": 0.30,
    "trends_z": 0.30,
    "volume_z": 0.20,
    "velocity_z": 0.20,
}

class HypeScanner:
    def __init__(self, assets: List[Dict[str, Any]], weights: Dict[str, float]=None):
        self.assets = assets
        self.weights = weights or DEFAULT_WEIGHTS
        self.cb = CoinbaseMarketData()
        self.news = NewsScanner()
        self.trends = TrendsScanner()

    def _get_market_df(self, product_id: str, hours: int=240, granularity: int=3600):
        end = pd.Timestamp.utcnow()
        start = end - pd.Timedelta(hours=hours)
        df = self.cb.candles(product_id, granularity=granularity, start=start, end=end)
        df["ema20"] = ema(df["close"].astype(float), 20)
        df["atr14"] = atr(df, 14)
        return df

    def scan(self, lookback_hours_news: int=24, hype_threshold: float=70.0, vol_z_threshold: float=1.5):
        # News
        news_df = self.news.scan(self.assets, lookback_hours=lookback_hours_news)
        if news_df.empty:
            news_df = pd.DataFrame({"product_id":[a["product_id"] for a in self.assets],
                                    "mentions":[0]*len(self.assets),
                                    "avg_sentiment":[0.0]*len(self.assets),
                                    "headline_velocity":[0.0]*len(self.assets)})
        # z-score for mentions and velocity relative to peers today
        if not news_df.empty:
            news_df["news_z"] = (news_df["mentions"] - news_df["mentions"].mean())/ (news_df["mentions"].std(ddof=1) or 1.0)
            news_df["velocity_z"] = (news_df["headline_velocity"] - news_df["headline_velocity"].mean())/ (news_df["headline_velocity"].std(ddof=1) or 1.0)
        # Trends
        trends_rows = []
        for a in self.assets:
            last, roc = self.trends.interest_24h_metrics(a.get("keywords", [a["name"]]))
            trends_rows.append({"product_id": a["product_id"], "trends_last": last, "trends_roc": roc})
        trends_df = pd.DataFrame(trends_rows)
        if not trends_df.empty:
            trends_df["trends_z"] = (trends_df["trends_roc"] - trends_df["trends_roc"].mean())/ (trends_df["trends_roc"].std(ddof=1) or 1.0)

        # Merge
        df = pd.merge(news_df, trends_df, on="product_id", how="outer").fillna(0.0)

        # Market features + triggers
        rows = []
        for a in self.assets:
            pid = a["product_id"]
            try:
                mdf = self._get_market_df(pid, hours=240, granularity=3600)  # 10 days of 1h
                if mdf.empty or len(mdf) < 50:
                    continue
                volz = volume_z(mdf, lookback=240)
                price = float(mdf["close"].iloc[-1])
                ema20_ok = price > float(mdf["ema20"].iloc[-1])
                # HypeScore
                n = df.loc[df["product_id"]==pid].squeeze()
                news_z = float(n.get("news_z", 0.0))
                vel_z = float(n.get("velocity_z", 0.0))
                trends_z = float(n.get("trends_z", 0.0))
                avg_sent = float(n.get("avg_sentiment", 0.0))
                # sentiment gating
                news_term = max(news_z, 0.0) * max(avg_sent, 0.0)

                hype_raw = (
                    self.weights["news_z"]   * news_term +
                    self.weights["trends_z"] * max(trends_z, 0.0) +
                    self.weights["volume_z"] * max(volz, 0.0) +
                    self.weights["velocity_z"]* max(vel_z, 0.0)
                )
                hype_score = float(np.clip(50 + 15*hype_raw, 0, 100))  # center ~50; scale

                eligible = (hype_score >= hype_threshold) and ema20_ok and (volz >= vol_z_threshold)

                rows.append({
                    "product_id": pid,
                    "symbol": a["symbol"],
                    "name": a["name"],
                    "price": price,
                    "ema20_ok": bool(ema20_ok),
                    "vol_z": float(volz),
                    "news_mentions": int(n.get("mentions", 0.0)),
                    "avg_sentiment": round(avg_sent,3),
                    "trends_roc": round(float(n.get("trends_roc", 0.0)),3),
                    "hype_score": round(hype_score,2),
                    "eligible": bool(eligible),
                })
            except Exception as ex:
                rows.append({
                    "product_id": pid, "symbol": a["symbol"], "name": a["name"],
                    "error": str(ex), "eligible": False
                })
        out = pd.DataFrame(rows).sort_values(["eligible","hype_score","vol_z"], ascending=[False,False,False]).reset_index(drop=True)
        return out
