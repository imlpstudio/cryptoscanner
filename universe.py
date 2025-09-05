import re, pandas as pd
from typing import List, Dict, Optional
from data_sources import CoinbaseMarketData

class UniverseBuilder:
    """
    Builds a dynamic Coinbase universe with liquidity screens.
    - Keeps USD/USDC spot products
    - Requires 30-day median hourly notional >= threshold
    - Returns asset dicts: {product_id, symbol, name, keywords}
    """
    def __init__(self):
        self.cb = CoinbaseMarketData()

    def _is_spot_usd(self, prod: dict) -> bool:
        pid = prod.get("id") or prod.get("product_id") or ""
        return bool(re.search(r"-(USD|USDC)$", pid)) and not prod.get("trading_disabled", False)

    def build(self, max_assets: int = 40, min_median_notional_usd: float = 5e5) -> List[Dict]:
        prods = [p for p in self.cb.list_products() if self._is_spot_usd(p)]
        # Rank by 30d median hourly notional
        ranked = []
        for p in prods:
            pid = p.get("id") or p.get("product_id")
            try:
                df = self.cb.candles(pid, granularity=3600)  # 1h (Coinbase returns last N)
                if len(df) < 24*20:  # ~20 days minimum
                    continue
                df["notional"] = df["close"].astype(float) * df["volume"].astype(float)
                med = float(df["notional"].tail(24*30).median())
                if med >= min_median_notional_usd:
                    sym = pid.split("-")[0]
                    ranked.append((med, {
                        "product_id": pid,
                        "symbol": sym,
                        "name": sym,               # Coinbase lacks full names in this endpoint; sym is fine
                        "keywords": [sym.lower(), f"{sym.lower()} crypto"]
                    }))
            except Exception:
                continue
        ranked.sort(key=lambda x: x[0], reverse=True)
        return [a for _, a in ranked[:max_assets]]
