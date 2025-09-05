from typing import List, Dict, Any
import pandas as pd, numpy as np
from math import exp
from data_sources import CoinbaseMarketData, ema, atr, donchian_high, volume_z

def _confidence(score, threshold, confirms_true, confirms_total, vol_z, vol_thr):
    s = 1.0 / (1.0 + exp(-(score - threshold) / 8.0))
    confirms = (confirms_true / max(1, confirms_total))
    vol_factor = min(1.0, max(0.0, (vol_z - vol_thr + 0.5)))
    return max(0.0, min(1.0, 0.45*s + 0.45*confirms + 0.10*vol_factor))

def _label(c):
    return "High" if c >= 0.75 else ("Medium" if c >= 0.5 else "Low")

class QuantStrategy:
    """
    Robust trend/stat rules: EMA trend, Donchian breakout, vol targeting proxy.
    """
    def __init__(self, assets: List[Dict[str, Any]]):
        self.assets = assets or []
        self.cb = CoinbaseMarketData()

    def _market_df(self, pid: str, hours: int=500, granularity: int=3600):
        end = pd.Timestamp.utcnow(); start = end - pd.Timedelta(hours=hours)
        df = self.cb.candles(pid, granularity=granularity, start=start, end=end)
        if df.empty: return df
        df["ema20"]  = ema(df["close"].astype(float), 20)
        df["ema50"]  = ema(df["close"].astype(float), 50)
        df["ema200"] = ema(df["close"].astype(float), 200)
        df["atr14"]  = atr(df, 14)
        df["donchian20"] = donchian_high(df, 20)
        return df

    def scan(self, threshold: float = 75.0, vol_z_threshold: float = 1.0):
        if not self.assets: return pd.DataFrame(), pd.DataFrame()
        rows=[]
        for a in self.assets:
            pid=a["product_id"]; sym=a["symbol"]; name=a["name"]
            try:
                mdf = self._market_df(pid)
                if mdf.empty or len(mdf)<200:
                    rows.append({"product_id": pid,"symbol": sym,"name": name,"eligible": False,"score": 0,"confidence":0,"confidence_label":"Low"})
                    continue

                price=float(mdf["close"].iloc[-1])
                ema_trend = float(mdf["ema20"].iloc[-1] - mdf["ema50"].iloc[-1]) / max(mdf["ema50"].iloc[-1], 1e-9)
                above_200 = price > float(mdf["ema200"].iloc[-1])
                breakout  = price > float(mdf["donchian20"].iloc[-2]) if not pd.isna(mdf["donchian20"].iloc[-2]) else False
                v_z       = float(volume_z(mdf, 240))

                raw = (0.6 * max(ema_trend,0.0)) + (0.3 * (1.0 if breakout else 0.0)) + (0.1 * max(v_z,0.0)/3.0)
                score = float(np.clip(50 + 100*raw, 0, 100))

                confirms_true = int(above_200) + int(breakout) + int(v_z >= vol_z_threshold)
                conf = _confidence(score, threshold, confirms_true, 3, v_z, vol_z_threshold)
                label = _label(conf)

                eligible = above_200 and breakout and (score >= threshold) and (v_z >= vol_z_threshold)

                rows.append({
                    "product_id": pid, "symbol": sym, "name": name, "price": price,
                    "vol_z": round(v_z,2), "trend_strength": round(ema_trend,3),
                    "above_200": bool(above_200), "donchian_breakout": bool(breakout),
                    "score": round(score,2), "eligible": bool(eligible),
                    "confidence": round(100*conf,1), "confidence_label": label,
                    "explain": "QuantScore: EMA(20/50) trend + 20-day Donchian breakout + volume health"
                })
            except Exception as ex:
                rows.append({"product_id": pid,"symbol": sym,"name": name,"error": str(ex),"eligible": False,"score":0,"confidence":0,"confidence_label":"Low"})
        out = pd.DataFrame(rows).sort_values(["eligible","confidence","score","vol_z"], ascending=[False,False,False,False]).reset_index(drop=True)
        return out, pd.DataFrame()
