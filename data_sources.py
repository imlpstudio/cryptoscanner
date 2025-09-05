import os, time, math, datetime as dt, requests, logging, re, yaml
from typing import List, Dict, Optional, Tuple
import pandas as pd
import numpy as np
import feedparser
from urllib.parse import urlparse
from requests_cache import CachedSession

# Google Trends
from pytrends.request import TrendReq

# Sentiment (bundled; no NLTK download needed)
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    _SIA = SentimentIntensityAnalyzer()
except Exception:
    _SIA = None

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# -----------------------------
# Coinbase Market Data
# -----------------------------
class CoinbaseMarketData:
    EXCHANGE_BASE = "https://api.exchange.coinbase.com"
    ADV_BASE = "https://api.coinbase.com/api/v3/brokerage"

    def __init__(self):
        self.session = CachedSession(cache_name="cb_cache", backend="memory", expire_after=120)
        self.session.headers.update({"User-Agent": "HypeHunter/2.0"})
        self.key = os.getenv("CB_API_KEY"); self.secret = os.getenv("CB_API_SECRET")
        self.passphrase = os.getenv("CB_API_PASSPHRASE")

    def list_products(self) -> List[Dict]:
        if self.key and self.secret:
            try:
                url = f"{self.ADV_BASE}/market/products"
                r = self.session.get(url, timeout=10); r.raise_for_status()
                data = r.json(); prods = data.get("products", [])
                if prods: return prods
            except Exception as e:
                logger.warning(f"ADV list_products failed, fallback: {e}")
        url = f"{self.EXCHANGE_BASE}/products"
        r = self.session.get(url, timeout=10); r.raise_for_status()
        return r.json()

    def candles(self, product_id: str, granularity: int = 3600,
                start: Optional[dt.datetime]=None, end: Optional[dt.datetime]=None) -> pd.DataFrame:
        params = {"granularity": granularity}
        if start is not None: params["start"] = start.isoformat()
        if end is not None: params["end"] = end.isoformat()
        url = f"{self.EXCHANGE_BASE}/products/{product_id}/candles"
        r = self.session.get(url, params=params, timeout=10); r.raise_for_status()
        arr = r.json()
        if not isinstance(arr, list): raise RuntimeError(f"Unexpected candles payload for {product_id}: {arr}")
        df = pd.DataFrame(arr, columns=["time", "low", "high", "open", "close", "volume"])
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True).dt.tz_convert("UTC")
        return df.sort_values("time").reset_index(drop=True)

# -----------------------------
# RSS News & Headline Sentiment
# -----------------------------
DEFAULT_FEEDS = [
    "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "https://cointelegraph.com/rss",
    "https://decrypt.co/feed",
    "https://www.theblock.co/rss",
    "https://news.google.com/rss/search?q=crypto&hl=en-US&gl=US&ceid=US:en",
]

def _domain(u: str) -> str:
    try:
        netloc = urlparse(u).netloc.lower()
        return netloc[4:] if netloc.startswith("www.") else netloc
    except Exception:
        return ""

class NewsScanner:
    def __init__(self, feeds: Optional[List[str]]=None, weights_path: str="config/news_weights.yaml"):
        self.feeds = feeds or DEFAULT_FEEDS
        self.session = CachedSession(cache_name="rss_cache", backend="memory", expire_after=300)
        self.session.headers.update({"User-Agent": "HypeHunter/2.0"})
        self.weights = {"default": 1.0, "weights": {}}
        if os.path.exists(weights_path):
            try:
                with open(weights_path,"r") as f: self.weights = yaml.safe_load(f) or self.weights
            except Exception: pass
        self.last_details = pd.DataFrame()

    def _match_asset(self, title: str, text: str, keywords: List[str]) -> bool:
        s = f"{title} {text}".lower()
        for kw in keywords:
            pat = r"\b" + re.escape(kw.lower()) + r"\b"
            if re.search(pat, s): return True
        return False

    def _w(self, url: str) -> float:
        d = _domain(url or "")
        return float(self.weights.get("weights", {}).get(d, self.weights.get("default", 1.0)))

    def scan(self, assets: List[Dict], lookback_hours: int=24) -> Tuple[pd.DataFrame, pd.DataFrame]:
        cutoff = pd.Timestamp.utcnow() - pd.Timedelta(hours=lookback_hours)
        rows = []
        for feed in self.feeds:
            try:
                d = feedparser.parse(feed)
                for e in d.entries:
                    if hasattr(e, "published_parsed") and e.published_parsed:
                        pub = pd.Timestamp(*e.published_parsed[:6], tz="UTC")
                    elif hasattr(e, "updated_parsed") and e.updated_parsed:
                        pub = pd.Timestamp(*e.updated_parsed[:6], tz="UTC")
                    else:
                        pub = pd.Timestamp.utcnow()
                    if pub < cutoff: continue

                    title = getattr(e, "title", "") or ""
                    summary = getattr(e, "summary", "") or ""
                    link = getattr(e, "link", "") or ""
                    text = f"{title}. {summary}".strip()
                    sent = (_SIA.polarity_scores(text)["compound"] if _SIA else 0.0)
                    w = self._w(link)

                    for a in assets:
                        if self._match_asset(title, summary, a.get("keywords", [])):
                            rows.append({
                                "product_id": a["product_id"], "title": title, "url": link,
                                "published": pub, "sentiment": sent, "weight": w, "domain": _domain(link), "feed": feed
                            })
            except Exception as ex:
                logger.warning(f"RSS error {feed}: {ex}")

        details = pd.DataFrame(rows)
        self.last_details = details.copy()

        if details.empty:
            agg = pd.DataFrame(columns=["product_id","mentions","avg_sentiment","headline_velocity","news_w_mentions","news_w_sentiment"])
            return agg, details

        # weighted metrics
        details["w_sent"] = details["sentiment"] * details["weight"]
        g = details.groupby("product_id")
        agg = pd.DataFrame({
            "mentions": g["title"].count(),
            "avg_sentiment": g["sentiment"].mean(),
            "headline_velocity": g["published"].count() / max((lookback_hours/6),1.0),
            "news_w_mentions": (g["weight"].sum()),
            "news_w_sentiment": (g["w_sent"].sum() / g["weight"].sum())
        }).reset_index()

        return agg, details

# -----------------------------
# Google Trends (pytrends)
# -----------------------------
class TrendsScanner:
    def __init__(self, geo: str="US"):
        self.tr = TrendReq(hl="en-US", tz=360); self.geo = geo

    def interest_24h_metrics(self, keywords: List[str]) -> Tuple[float, float]:
        best_series = None
        for kw in keywords:
            try:
                self.tr.build_payload([kw], timeframe="now 7-d", geo=self.geo)
                df = self.tr.interest_over_time()
                if df.empty: continue
                s = df.iloc[:,0].astype(float)
                if best_series is None or s.max() > best_series.max(): best_series = s
            except Exception:
                time.sleep(0.5); continue
        if best_series is None or len(best_series) < 25: return (0.0, 0.0)
        last = float(best_series.iloc[-1]); prev = float(best_series.iloc[-25])
        return (last, (last - prev) / max(prev, 1.0))

# -----------------------------
# Helpers
# -----------------------------
def zscore(series: pd.Series) -> float:
    if len(series) < 5: return 0.0
    m = series.mean(); s = series.std(ddof=1)
    if s == 0 or math.isnan(s): return 0.0
    return float((series.iloc[-1] - m) / s)

def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def atr(df: pd.DataFrame, period: int=14) -> pd.Series:
    high = df["high"].astype(float); low = df["low"].astype(float); close = df["close"].astype(float)
    prev_close = close.shift(1)
    tr = pd.concat([high-low, (high-prev_close).abs(), (low-prev_close).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def donchian_high(df: pd.DataFrame, window: int=20) -> pd.Series:
    return df["high"].rolling(window).max()

def volume_z(df: pd.DataFrame, lookback: int=240) -> float:
    v = df["volume"].astype(float)
    if len(v) < lookback: lookback = max(30, len(v)//2)
    return zscore(v.tail(lookback))
