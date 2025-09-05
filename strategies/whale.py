from typing import List, Dict, Any, Tuple
import pandas as pd, numpy as np, requests
from math import exp
from data_sources import CoinbaseMarketData, ema, atr, donchian_high, volume_z
from requests_cache import CachedSession

def _confidence(score, threshold, confirms_true, confirms_total, vol_z, vol_thr, regime_ok):
    s = 1.0 / (1.0 + exp(-(score - threshold) / 8.0))
    confirms = (confirms_true / max(1, confirms_total))
    vol_factor = min(1.0, max(0.0, (vol_z - vol_thr + 0.5)))
    base = 0.4*s + 0.4*confirms + 0.2*vol_factor
    return 0.0 if not regime_ok else max(0.0, min(1.0, base))

def _label(conf):
    return "High" if conf >= 0.75 else ("Medium" if conf >= 0.5 else "Low")

class WhaleStrategy:
    """
    Follow-the-money: unusual trade size, order-book imbalance, volume spike, regime.
    """
    def __init__(self, assets: List[Dict[str, Any]], big_trade_usd: float = 250_000.0):
        self.assets = assets or []
        self.cb = CoinbaseMarketData()
        self.http = CachedSession(cache_name="whale_cache", backend="memory", expire_after=60)
        self.big_trade_usd = big_trade_usd

    def _market_df(self, pid: str, hours: int=240, granularity: int=3600):
        end = pd.Timestamp.utcnow(); start = end - pd.Timedelta(hours=hours)
        df = self.cb.candles(pid, granularity=granularity, start=start, end=end)
        if df.empty: return df
        df["ema50"] = ema(df["close"].astype(float), 50)
        df["ema200"] = ema(df["close"].astype(float), 200)
        df["atr14"] = atr(df, 14)
        df["donchian20"] = donchian_high(df, 20)
        return df

    def _trades_recent(self, pid: str, limit: int = 200) -> pd.DataFrame:
        url = f"{self.cb.EXCHANGE_BASE}/products/{pid}/trades"
        r = self.http.get(url, params={"limit": limit}, timeout=10)
        r.raise_for_status()
        data = r.json()
        if not isinstance(data, list): return pd.DataFrame()
        df = pd.DataFrame(data)
        if df.empty: return df
        df["time"] = pd.to_datetime(df["time"], utc=True)
        df["price"] = df["price"].astype(float)
        df["size"] = df["size"].astype(float)
        df["notional"] = df["price"] * df["size"]
        return df

    def _orderbook_imbalance(self, pid: str, depth: int = 50) -> float:
        url = f"{self.cb.EXCHANGE_BASE}/products/{pid}/book"
        r = self.http.get(url, params={"level": 2}, timeout=10)
        r.raise_for_status()
        bk = r.json()
        bids = bk.get("bids", [])[:depth]
        asks = bk.get("asks", [])[:depth]
        bid_vol = sum(float(x[1]) for x in bids)
        ask_vol = sum(float(x[1]) for x in asks)
        total = bid_vol + ask_vol
        if total == 0: return 0.0
        return (bid_vol - ask_vol) / total

    def _regime_ok(self) -> bool:
        try:
            btc = self._market_df("BTC-USD")
            return (not btc.empty) and (btc["close"].iloc[-1] > btc["ema200"].iloc[-1])
        except Exception:
            return True

    def scan(self, threshold: float = 70.0, vol_z_threshold: float = 1.2):
        if not self.assets: return pd.DataFrame(), pd.DataFrame()
        regime = self._regime_ok()

        peer_stats = []
        details_rows = []

        for a in self.assets:
            pid=a["product_id"]; sym=a["symbol"]; name=a["name"]
            try:
                mdf = self._market_df(pid)
                if mdf.empty or len(mdf)<50:
                    peer_stats.append({"product_id": pid, "big_notional": 0.0, "ob_imb": 0.0, "vol_z": 0.0})
                    continue

                trades = self._trades_recent(pid, limit=200)
                big_notional = float(trades.loc[trades["notional"]>=self.big_trade_usd,"notional"].sum()) if not trades.empty else 0.0
                ob_imb = float(self._orderbook_imbalance(pid, depth=50))
                v_z = float(volume_z(mdf, 240))

                if not trades.empty:
                    tops = trades.sort_values("notional", ascending=False).head(5)
                    for _, t in tops.iterrows():
                        details_rows.append({
                            "product_id": pid, "symbol": sym, "time": t["time"], "side": t.get("side",""),
                            "price": float(t["price"]), "size": float(t["size"]), "notional": float(t["notional"])
                        })

                peer_stats.append({"product_id": pid, "big_notional": big_notional, "ob_imb": ob_imb, "vol_z": v_z})
            except Exception:
                peer_stats.append({"product_id": pid, "big_notional": 0.0, "ob_imb": 0.0, "vol_z": 0.0})

        stats = pd.DataFrame(peer_stats)
        if stats.empty:
            return pd.DataFrame(), pd.DataFrame(details_rows)

        def _z(s):
            std = s.std(ddof=1); return (s - s.mean())/(std if std else 1.0)
        stats["notional_z"] = _z(stats["big_notional"])
        stats["ob_imb_z"]   = _z(stats["ob_imb"])
        stats["vol_z_peer"] = _z(stats["vol_z"])

        rows=[]
        for a in self.assets:
            pid=a["product_id"]; sym=a["symbol"]; name=a["name"]
            try:
                mdf = self._market_df(pid)
                if mdf.empty or len(mdf)<50:
                    rows.append({"product_id": pid,"symbol": sym,"name": name,"eligible": False,"score": 0,"regime_ok": regime,"confidence":0,"confidence_label":"Low"})
                    continue
                price=float(mdf["close"].iloc[-1])
                ema50_ok = price > float(mdf["ema50"].iloc[-1])
                donchian_ok = price > float(mdf["donchian20"].iloc[-2]) if not pd.isna(mdf["donchian20"].iloc[-2]) else False

                s = stats.loc[stats["product_id"]==pid].squeeze()
                notional_z = max(float(s.get("notional_z",0.0)),0.0)
                ob_imb_z   = max(float(s.get("ob_imb_z",0.0)),0.0)
                v_z        = float(s.get("vol_z",0.0))

                raw = 0.5*notional_z + 0.3*ob_imb_z + 0.2*max(v_z,0.0)
                score = float(np.clip(50 + 15*raw, 0, 100))

                confirms_true = int(ema50_ok) + int(donchian_ok) + int(v_z >= vol_z_threshold) + int(regime)
                conf = _confidence(score, threshold, confirms_true, 4, v_z, vol_z_threshold, regime)
                label = _label(conf)

                eligible = (score >= threshold) and ema50_ok and (v_z >= vol_z_threshold) and donchian_ok and regime

                rows.append({
                    "product_id": pid, "symbol": sym, "name": name, "price": price,
                    "vol_z": round(v_z,2), "ema50_ok": bool(ema50_ok), "donchian_breakout": bool(donchian_ok),
                    "score": round(score,2), "eligible": bool(eligible), "regime_ok": regime,
                    "confidence": round(100*conf,1), "confidence_label": label,
                    "explain": "WhaleScore: big-trade notional + order-book imbalance + volume; EMA50 & Donchian confirm"
                })
            except Exception as ex:
                rows.append({"product_id": pid,"symbol": sym,"name": name,"error": str(ex),"eligible": False,"score":0,"regime_ok": regime,"confidence":0,"confidence_label":"Low"})
        out = pd.DataFrame(rows).sort_values(["eligible","confidence","score","vol_z"], ascending=[False,False,False,False]).reset_index(drop=True)
        details = pd.DataFrame(details_rows).sort_values("notional", ascending=False)
        return out, details
