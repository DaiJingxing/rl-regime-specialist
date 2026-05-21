from __future__ import annotations

from dataclasses import dataclass
from io import StringIO
import json
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd


DEFAULT_TICKERS = [
    "SPY.US",
    "QQQ.US",
    "DIA.US",
    "IWM.US",
    "EFA.US",
    "EEM.US",
    "TLT.US",
    "IEF.US",
    "LQD.US",
    "HYG.US",
    "GLD.US",
    "SLV.US",
    "USO.US",
    "UNG.US",
    "XLF.US",
    "XLK.US",
    "XLV.US",
    "XLE.US",
    "XLI.US",
    "XLP.US",
    "XLY.US",
    "XLU.US",
    "XLB.US",
    "XLRE.US",
]


@dataclass(frozen=True)
class DownloadedPrices:
    ticker: str
    path: str
    rows: int
    start: str
    end: str


def _stooq_symbol(ticker: str) -> str:
    return ticker.strip().lower()


def _yahoo_symbol(ticker: str) -> str:
    return ticker.strip().upper().removesuffix(".US")


def _to_epoch(date_text: str) -> int:
    return int(pd.Timestamp(date_text, tz="UTC").timestamp())


def _read_yahoo_daily(ticker: str, start: str, end: str | None) -> pd.DataFrame:
    start_epoch = _to_epoch(start)
    end_epoch = _to_epoch(end or pd.Timestamp.today(tz="UTC").strftime("%Y-%m-%d"))
    symbol = _yahoo_symbol(ticker)
    params = urlencode(
        {
            "period1": start_epoch,
            "period2": end_epoch,
            "interval": "1d",
            "events": "history",
            "includeAdjustedClose": "true",
        }
    )
    request = Request(
        f"https://query2.finance.yahoo.com/v8/finance/chart/{symbol}?{params}",
        headers={"User-Agent": "Mozilla/5.0"},
    )
    with urlopen(request, timeout=30) as response:
        payload = json.loads(response.read().decode("utf-8", errors="replace"))
    result = payload.get("chart", {}).get("result") or []
    if not result:
        return pd.DataFrame()
    data = result[0]
    timestamps = data.get("timestamp") or []
    quote = (data.get("indicators", {}).get("quote") or [{}])[0]
    adjclose = (data.get("indicators", {}).get("adjclose") or [{}])[0].get("adjclose")
    if not timestamps or "close" not in quote:
        return pd.DataFrame()
    df = pd.DataFrame(
        {
            "Date": pd.to_datetime(timestamps, unit="s", utc=True).tz_convert(None).date,
            "Open": quote.get("open"),
            "High": quote.get("high"),
            "Low": quote.get("low"),
            "Close": adjclose or quote.get("close"),
            "Volume": quote.get("volume"),
        }
    )
    return df


def download_stooq_daily(
    tickers: list[str],
    output_dir: str = "data/stooq",
    start: str = "2005-01-01",
    end: str | None = None,
    min_rows: int = 500,
) -> list[DownloadedPrices]:
    """Download daily OHLCV data from Stooq and cache one CSV per ticker."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    start_yyyymmdd = start.replace("-", "")
    end_yyyymmdd = (end or pd.Timestamp.today().strftime("%Y-%m-%d")).replace("-", "")
    downloaded: list[DownloadedPrices] = []

    for ticker in tickers:
        try:
            df = _read_yahoo_daily(ticker, start, end)
            if df.empty:
                params = urlencode({"s": _stooq_symbol(ticker), "i": "d", "d1": start_yyyymmdd, "d2": end_yyyymmdd})
                url = f"https://stooq.com/q/d/l/?{params}"
                with urlopen(url, timeout=30) as response:
                    body = response.read().decode("utf-8", errors="replace")
                if not body.startswith("Date,"):
                    continue
                df = pd.read_csv(StringIO(body))
        except Exception:
            try:
                params = urlencode({"s": _stooq_symbol(ticker), "i": "d", "d1": start_yyyymmdd, "d2": end_yyyymmdd})
                url = f"https://stooq.com/q/d/l/?{params}"
                with urlopen(url, timeout=30) as response:
                    body = response.read().decode("utf-8", errors="replace")
                if not body.startswith("Date,"):
                    continue
                df = pd.read_csv(StringIO(body))
            except Exception:
                continue
        if df.empty or "Close" not in df.columns:
            continue
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date", "Close"]).sort_values("Date")
        if len(df) < min_rows:
            continue
        path = out / f"{ticker.replace('.', '_').lower()}.csv"
        df.to_csv(path, index=False)
        downloaded.append(
            DownloadedPrices(
                ticker=ticker,
                path=str(path),
                rows=len(df),
                start=str(df["Date"].iloc[0].date()),
                end=str(df["Date"].iloc[-1].date()),
            )
        )
    return downloaded


def load_downloaded_close_series(downloads: list[DownloadedPrices]) -> dict[str, pd.Series]:
    series: dict[str, pd.Series] = {}
    for item in downloads:
        df = pd.read_csv(item.path, parse_dates=["Date"])
        close = pd.to_numeric(df["Close"], errors="coerce")
        s = pd.Series(close.to_numpy(), index=df["Date"], name=item.ticker).dropna()
        if len(s) >= 50:
            series[item.ticker] = s
    return series
