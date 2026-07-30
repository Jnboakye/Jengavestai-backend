# stocks.py - Stock catalogue + live market data for JengaVest
#
# Providers:
#   - Quotes   -> Finnhub  /quote            (FINNHUB_API_KEY)      [free tier]
#   - History  -> Twelve Data /time_series    (TWELVE_DATA_API_KEY)  [free Basic tier]
#
# These power the HTTP endpoints the frontend calls:
#   GET /stocks                      -> list_stocks()
#   GET /stocks/{ticker}/quote       -> get_quote()
#   GET /stocks/{ticker}/history     -> get_history()
#
# If a provider is unreachable or a key is missing, the functions raise and the
# API layer returns a clean error the frontend falls back from (bundled snapshot
# for the quote, synthetic series for the chart), so the UI never breaks.

import os
from datetime import datetime, timezone

import requests

FINNHUB_BASE = "https://finnhub.io/api/v1"
TWELVE_DATA_BASE = "https://api.twelvedata.com"

# ─────────────────────────────────────────────────────────
# CATALOGUE — the tradable universe the frontend can search.
# name/sector are served from here; price/change/history come live.
# ─────────────────────────────────────────────────────────

CATALOGUE = [
    {"ticker": "AAPL", "name": "Apple Inc.", "sector": "Technology"},
    {"ticker": "MSFT", "name": "Microsoft Corporation", "sector": "Technology"},
    {"ticker": "GOOGL", "name": "Alphabet Inc.", "sector": "Technology"},
    {"ticker": "AMZN", "name": "Amazon.com Inc.", "sector": "Consumer"},
    {"ticker": "NVDA", "name": "NVIDIA Corporation", "sector": "Technology"},
    {"ticker": "META", "name": "Meta Platforms Inc.", "sector": "Technology"},
    {"ticker": "TSLA", "name": "Tesla Inc.", "sector": "Automotive"},
    {"ticker": "BRK.B", "name": "Berkshire Hathaway", "sector": "Financials"},
    {"ticker": "JPM", "name": "JPMorgan Chase & Co.", "sector": "Financials"},
    {"ticker": "V", "name": "Visa Inc.", "sector": "Financials"},
    {"ticker": "MA", "name": "Mastercard Inc.", "sector": "Financials"},
    {"ticker": "UNH", "name": "UnitedHealth Group", "sector": "Healthcare"},
    {"ticker": "JNJ", "name": "Johnson & Johnson", "sector": "Healthcare"},
    {"ticker": "LLY", "name": "Eli Lilly and Co.", "sector": "Healthcare"},
    {"ticker": "WMT", "name": "Walmart Inc.", "sector": "Consumer"},
    {"ticker": "PG", "name": "Procter & Gamble", "sector": "Consumer"},
    {"ticker": "HD", "name": "Home Depot Inc.", "sector": "Consumer"},
    {"ticker": "DIS", "name": "Walt Disney Co.", "sector": "Communication"},
    {"ticker": "NFLX", "name": "Netflix Inc.", "sector": "Communication"},
    {"ticker": "KO", "name": "Coca-Cola Co.", "sector": "Consumer"},
    {"ticker": "PEP", "name": "PepsiCo Inc.", "sector": "Consumer"},
    {"ticker": "XOM", "name": "Exxon Mobil Corp.", "sector": "Energy"},
    {"ticker": "CVX", "name": "Chevron Corporation", "sector": "Energy"},
    {"ticker": "BAC", "name": "Bank of America", "sector": "Financials"},
    {"ticker": "ADBE", "name": "Adobe Inc.", "sector": "Technology"},
    {"ticker": "CRM", "name": "Salesforce Inc.", "sector": "Technology"},
    {"ticker": "AMD", "name": "Advanced Micro Devices", "sector": "Technology"},
    {"ticker": "NKE", "name": "Nike Inc.", "sector": "Consumer"},
    {"ticker": "SPY", "name": "S&P 500 ETF", "sector": "ETF"},
    {"ticker": "QQQ", "name": "Invesco QQQ ETF", "sector": "ETF"},
]

_BY_TICKER = {s["ticker"]: s for s in CATALOGUE}

# range -> (Twelve Data interval, number of data points to request)
RANGE_MAP = {
    "1D": ("5min", 78),    # ~one trading day of 5-minute bars
    "1W": ("30min", 65),   # ~one week of 30-minute bars
    "1M": ("1day", 22),    # ~one month of trading days
    "1Y": ("1day", 252),   # ~one year of trading days
    "5Y": ("1week", 260),  # ~five years of weekly bars
}


def _finnhub_key() -> str:
    key = os.getenv("FINNHUB_API_KEY")
    if not key:
        raise RuntimeError("FINNHUB_API_KEY is not set")
    return key


def _twelvedata_key() -> str:
    key = os.getenv("TWELVE_DATA_API_KEY")
    if not key:
        raise RuntimeError("TWELVE_DATA_API_KEY is not set")
    return key


def list_stocks():
    """The full tradable catalogue (no live prices — cheap and instant)."""
    return CATALOGUE


def search_stocks(query: str):
    """Filter the catalogue by ticker or name."""
    q = (query or "").strip().lower()
    if not q:
        return CATALOGUE
    return [
        s for s in CATALOGUE
        if q in s["ticker"].lower() or q in s["name"].lower()
    ]


def get_quote(ticker: str) -> dict:
    """Live quote for a single ticker via Finnhub /quote."""
    key = _finnhub_key()
    sym = ticker.upper()
    meta = _BY_TICKER.get(sym, {"ticker": sym, "name": sym, "sector": "—"})

    resp = requests.get(
        f"{FINNHUB_BASE}/quote",
        params={"symbol": sym, "token": key},
        timeout=10,
    )
    quote = resp.json()

    price = quote.get("c")
    if not price:
        raise ValueError(f"No price data returned for {ticker}")

    prev = quote.get("pc") or price
    change = quote.get("dp")
    if change is None:
        change = ((price - prev) / prev * 100) if prev else 0.0

    return {
        "ticker": meta["ticker"],
        "name": meta["name"],
        "sector": meta["sector"],
        "price": round(float(price), 2),
        "change": round(float(change), 2),
        "previous_close": round(float(prev), 2),
        "high": quote.get("h"),
        "low": quote.get("l"),
        "open": quote.get("o"),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def get_history(ticker: str, range_key: str = "1M") -> dict:
    """
    Historical price series for a ticker and range (1D/1W/1M/1Y/5Y) via
    Twelve Data /time_series. The free Basic plan supports this for US stocks.
    """
    key = _twelvedata_key()
    sym = ticker.upper()
    interval, outputsize = RANGE_MAP.get(range_key, RANGE_MAP["1M"])

    resp = requests.get(
        f"{TWELVE_DATA_BASE}/time_series",
        params={
            "symbol": sym,
            "interval": interval,
            "outputsize": outputsize,
            "apikey": key,
        },
        timeout=15,
    )
    data = resp.json()

    if data.get("status") != "ok" or not data.get("values"):
        reason = data.get("message") or data.get("status") or f"HTTP {resp.status_code}"
        raise ValueError(f"No time series for {ticker}: {reason}")

    # Twelve Data returns newest-first; reverse to chronological order.
    values = list(reversed(data["values"]))
    points = []
    for v in values:
        close = v.get("close")
        stamp = v.get("datetime")
        if close is None or stamp is None:
            continue
        points.append({"t": stamp, "value": round(float(close), 2)})

    return {"ticker": sym, "range": range_key, "points": points}
