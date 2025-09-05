import os, time, math, datetime as dt, requests, logging, re
from typing import List, Dict, Optional, Tuple
import pandas as pd
import numpy as np
import feedparser

# Google Trends
from pytrends.request import TrendReq

# Sentiment (bundled lexicon; no NLTK download needed)
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
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "HypeHunter/1.0"})
        self.key = os.getenv("CB_API_KEY")
        self.secret = os.getenv("CB_API_SECRET")
        self.passphrase = os.getenv("CB_API_PASSPHRASE")

    def list_products(self) -> List[Dict]:
        # Try Advanced Trade (some setups may still allow unauth GETs)
        if self.key and self.secret:
            try:
                url = f"{self.ADV_BASE}/market/products"
                r = self.session.get(url, timeout=10)
                r.raise_for_status()
                data = r.json()
                products = data.get("products", [])
                if products:
                    return products
            except Exception as e:
                logger.warning(f"ADV list_products failed, falling back to Exchange /products: {e}")

        # Fallback: Exchange public endpoint
        url = f"{self.EXCHANGE_BASE}/products"
        r = self.session.get(url, timeout=10)
        r.raise_for_status()
        return r.json()

    def candles(self, product_id: str, granularity: int = 3600,
                start: Optional[dt.datetime]=None, end: Optional[dt.datetime]=None) -> pd.DataFrame:
        """
        Returns DataFrame with columns: time, low, high, open, close, volume.
        Coinbase format: [[time, low, high, open, close, volume], ...]
        """
        params = {"granularity": granularity}
        if start is not None: params["start"] = start.isoformat()
        if end is not None: params["end"] = end.isoformat()
        url = f"{self.EXCHANGE_BASE}/products/{product_id}/candles"
        r = self.session.get(url, params=params, timeout=10)
        r.raise_for_status()
        arr = r.json()
        if not isinstance(arr, list):
            raise RuntimeError(f"Unexpected candles payload for {product_id}: {arr}")
        df = pd.DataFrame(arr, columns=["time", "low", "high", "open", "close", "volume"])
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True).dt.tz_convert("UTC")
        df = df.sort_values("time").reset_index(drop=True)
        return df

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

class NewsScanner:
    def __init__(self, feeds: Optional[List[str]]=None):
        self.feeds = feeds or DEFAULT_FEEDS
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "HypeHunter/1.0"})

    @staticmethod
    def _match_asset(title: str, text: str, keywords: List[str]) -> bool:
        s = f"{title} {text}".lower()
        for kw in keywords:
            pat = r"\b" + re.escape(kw.lower()) + r"\b"
            if re.search(pat, s):
                return True
        return False

    def scan(self, assets: List[Dict], lookback_hours: int=24) -> pd.DataFrame:
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

                    if pub < cutoff:
                        continue

                    title = getattr(e, "title", "") or ""
                    summary = getattr(e, "summary", "") or ""
                    text = f"{title}. {summary}".strip()
                    sent = (_SIA.polarity_scores(text)["compound"] if _SIA else 0.0)

                    for a in assets:
                        if self._match_asset(title, summary, a.get("keywords", [])):
                            rows.append({
                                "product_id": a["product_id"],
                                "title": title,
                                "published": pub,
                                "sentiment": sent,
                                "feed": feed
                            })
            except Exception as ex:
                logger.warning(f"RSS error {feed}: {ex}")
        df = pd.DataFrame(rows)
        if df.empty:
            return pd.DataFrame(columns=["product_id","mentions","avg_sentiment","headline_velocity"])
        agg = df.groupby("product_id").agg(
            mentions=("title","count"),
            avg_sentiment=("sentiment","mean"),
            headline_velocity=("published", lambda s: s.count() / max((lookback_hours/6),1.0))
        ).reset_index()
        return agg

# -----------------------------
# Google Trends (pytrends)
# -----------------------------
class TrendsScanner:
    def __init__(self, geo: str="US"):
        self.tr = TrendReq(hl="en-US", tz=360)
        self.geo = geo

    def interest_24h_metrics(self, keywords: List[str]) -> Tuple[float, float]:
        """
        Return (last_value, roc_24h) using last 7 days hourly data;
        roc_24h = (last - value_24h_ago) / max(value_24h_ago, 1)
        """
        best_series = None
        for kw in keywords:
            try:
                self.tr.build_payload([kw], timeframe="now 7-d", geo=self.geo)
                df = self.tr.interest_over_time()
                if df.empty:
                    continue
                s = df.iloc[:,0].astype(float)
                if best_series is None or s.max() > best_series.max():
                    best_series = s
            except Exception:
                time.sleep(0.5)
                continue
        if best_series is None or len(best_series) < 25:
            return (0.0, 0.0)
        last = float(best_series.iloc[-1])
        prev = float(best_series.iloc[-25])
        roc = (last - prev) / max(prev, 1.0)
        return (last, roc)

# -----------------------------
# Helpers
# -----------------------------
def zscore(series: pd.Series) -> float:
    if len(series) < 5:
        return 0.0
    m = series.mean()
    s = series.std(ddof=1)
    if s == 0 or math.isnan(s):
        return 0.0
    return float((series.iloc[-1] - m) / s)

def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def atr(df: pd.DataFrame, period: int=14) -> pd.Series:
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def volume_z(df: pd.DataFrame, lookback: int=240) -> float:
    v = df["volume"].astype(float)
    if len(v) < lookback:
        lookback = max(30, len(v)//2)
    base = v.tail(lookback)
    return zscore(base)
