import re, pandas as pd
from typing import List, Dict, Optional, Iterable
from data_sources import CoinbaseMarketData

class UniverseBuilder:
    """
    Builds a dynamic Coinbase universe with liquidity screens.
    - Keeps USD/USDC spot products (configurable)
    - Computes 30-day median hourly notional and filters by threshold
    - Returns dicts with median_notional so UI can list/sort
    """
    def __init__(self):
        self.cb = CoinbaseMarketData()

    def _is_spot_quote(self, product_id: str, quote_set: Iterable[str]) -> bool:
        return any(product_id.endswith(f"-{q}") for q in quote_set)

    def build(
        self,
        max_assets: int = 120,
        min_median_notional_usd: float = 3e5,
        quote_currencies: Iterable[str] = ("USD","USDC"),
    ) -> List[Dict]:
        prods = []
        for p in self.cb.list_products():
            pid = p.get("id") or p.get("product_id") or ""
            if not pid:
                continue
            if p.get("trading_disabled", False):
                continue
            if not self._is_spot_quote(pid, quote_currencies):
                continue
            prods.append(pid)

        ranked = []
        for pid in prods:
            try:
                df = self.cb.candles(pid, granularity=3600)  # 1h bars
                if df.empty or len(df) < 24*20:
                    continue
                df["notional"] = df["close"].astype(float) * df["volume"].astype(float)
                med = float(df["notional"].tail(24*30).median())
                if med >= min_median_notional_usd:
                    sym = pid.split("-")[0]
                    ranked.append((med, {
                        "product_id": pid,
                        "symbol": sym,
                        "name": sym,
                        "keywords": [sym.lower(), f"{sym.lower()} crypto"],
                        "median_notional": med
                    }))
            except Exception:
                continue

        ranked.sort(key=lambda x: x[0], reverse=True)
        return [a for _, a in ranked[:max_assets]]
