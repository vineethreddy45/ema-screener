"""Gate 1 fundamentals snapshot for the 20/30 EMA screener.

Builds a per-universe snapshot CSV (fundamentals_<out>.csv) with basic
value/quality fields from yfinance, plus a VALUE flag computed by
compute_value(). screener.py merges the snapshot into its reports when
the file exists; when it does not, the screener runs exactly as before.

Gate 1 decides *what is worth owning*; the EMA signals (Gate 2) decide
*when to act*. A ticker is dual-confirmed when it has an EMA signal AND
VALUE = "✓".

Usage:
  .venv/bin/python fundamentals.py --tickers tickers_us500.txt --out us500
  .venv/bin/python fundamentals.py --tickers tickers_in500.txt --suffix .NS --out in500
  .venv/bin/python fundamentals.py --tickers mylist.txt --out test --limit 15
"""
from __future__ import annotations

import argparse
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import yfinance as yf

from screener import load_tickers

BASE = Path(__file__).resolve().parent
ET = ZoneInfo("America/New_York")

MAX_RETRIES = 3


def _num(info: dict, *keys: str):
    """First present, numeric value among keys, else None.

    yfinance exposes PEG under both 'trailingPegRatio' and the older
    'pegRatio' depending on version/ticker, so callers pass both.
    """
    for k in keys:
        v = info.get(k)
        if v is None:
            continue
        try:
            f = float(v)
        except (TypeError, ValueError):
            continue
        if f != f:  # NaN
            continue
        return f
    return None


def fetch_one(symbol: str) -> dict:
    """Fetch one ticker's fundamentals; never raises.

    Retries with exponential backoff so transient Yahoo rate limits do
    not kill the ticker, and returns an all-None row on total failure so
    one bad ticker cannot kill the run.
    """
    row = {"ticker": symbol, "trailing_pe": None, "forward_pe": None,
           "peg": None, "roe": None, "debt_to_equity": None,
           "free_cashflow": None, "market_cap": None,
           "earnings_growth": None, "revenue_growth": None,
           "sector": None}
    for attempt in range(MAX_RETRIES):
        try:
            info = yf.Ticker(symbol).info or {}
            if info:
                row.update({
                    "trailing_pe": _num(info, "trailingPE"),
                    "forward_pe": _num(info, "forwardPE"),
                    "peg": _num(info, "trailingPegRatio", "pegRatio"),
                    "roe": _num(info, "returnOnEquity"),
                    # yfinance reports D/E as a percentage: 85.3 ~= 0.85x.
                    "debt_to_equity": _num(info, "debtToEquity"),
                    "free_cashflow": _num(info, "freeCashflow"),
                    "market_cap": _num(info, "marketCap"),
                    "earnings_growth": _num(info, "earningsGrowth"),
                    "revenue_growth": _num(info, "revenueGrowth"),
                    "sector": info.get("sector") or None,
                })
                return row
        except Exception:  # noqa: BLE001 - retry, then tolerate failure
            pass
        if attempt < MAX_RETRIES - 1:
            time.sleep((2 ** attempt) + random.uniform(0, 0.5))
    return row


# ---------------------------------------------------------------------------
# Gate 1 — value/quality flag. Tune the thresholds here; this is the only
# place they live. Missing data is handled explicitly: a field that is
# absent simply does not vote, EXCEPT that ROE must be present and at
# least one of FCF / PEG / P/E must be present, otherwise there is not
# enough information to judge and the flag is "–" (insufficient data).
# Financial-sector tickers often lack freeCashflow (and report naturally
# high D/E); per the rule above, missing FCF never fails a ticker.
# ---------------------------------------------------------------------------
ROE_MIN = 0.15        # quality: return on equity above 15%
DE_MAX = 100.0        # safety: debt/equity below ~1.0x (yfinance % units)
PEG_MAX = 1.5         # price sanity: PEG below 1.5 when available
PE_FALLBACK_MAX = 25  # price sanity fallback when PEG is missing


def compute_value(row: dict) -> tuple[str, str]:
    """Return (value_flag, reason) for one snapshot row.

    value_flag is "✓" (passes Gate 1), "–" (insufficient data), or ""
    (enough data, does not pass). reason is a short display string built
    only from values that are actually present — nothing is invented.
    """
    roe = row.get("roe")
    fcf = row.get("free_cashflow")
    de = row.get("debt_to_equity")
    peg = row.get("peg")
    pe = row.get("trailing_pe")

    # Reason string from whatever is present, e.g. "ROE 22%, PEG 1.1, D/E 0.4".
    parts = []
    if roe is not None:
        parts.append(f"ROE {roe * 100:.0f}%")
    if fcf is not None:
        parts.append("FCF +" if fcf > 0 else "FCF -")
    if de is not None:
        parts.append(f"D/E {de / 100:.1f}")
    if peg is not None:
        parts.append(f"PEG {peg:.1f}")
    elif pe is not None:
        parts.append(f"P/E {pe:.0f}")
    reason = ", ".join(parts)

    # Enough-data rule: ROE plus at least one of FCF / PEG / P/E.
    if roe is None or (fcf is None and peg is None and pe is None):
        return "–", reason or "insufficient data"

    quality = roe > ROE_MIN and (fcf is None or fcf > 0)
    safety = de is None or de < DE_MAX
    if peg is not None:
        price_ok = peg < PEG_MAX
    else:
        price_ok = pe is not None and pe < PE_FALLBACK_MAX
    return ("✓" if (quality and safety and price_ok) else ""), reason


def build_snapshot(tickers: list[str], workers: int) -> pd.DataFrame:
    rows: dict[str, dict] = {}
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fetch_one, t): t for t in tickers}
        for fut in as_completed(futures):
            t = futures[fut]
            try:
                rows[t] = fut.result()
            except Exception:  # noqa: BLE001 - fetch_one never raises, belt+braces
                rows[t] = {"ticker": t}
            done += 1
            if done % 25 == 0 or done == len(tickers):
                ok = sum(1 for r in rows.values()
                         if r.get("trailing_pe") is not None
                         or r.get("roe") is not None)
                print(f"  progress: {done}/{len(tickers)} fetched "
                      f"({ok} with data)", flush=True)
    stamp = datetime.now(ET).strftime("%Y-%m-%d %H:%M ET")
    out_rows = []
    for t in tickers:  # preserve universe order
        r = dict(rows.get(t, {"ticker": t}))
        flag, reason = compute_value(r)
        r["value"] = flag
        r["value_reason"] = reason
        r["fetched_at"] = stamp
        out_rows.append(r)
    return pd.DataFrame(out_rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", default="tickers.txt")
    ap.add_argument("--suffix", default="")
    ap.add_argument("--out", default="watchlist",
                    help="snapshot name: fundamentals_<out>.csv")
    ap.add_argument("--limit", type=int, default=0,
                    help="only fetch the first N tickers (testing)")
    ap.add_argument("--workers", type=int, default=4,
                    help="parallel fetch threads (polite default 4)")
    args = ap.parse_args()

    tickers = load_tickers(args.tickers, args.suffix)
    if args.limit:
        tickers = tickers[: args.limit]
    print(f"fetching fundamentals for {len(tickers)} tickers "
          f"(workers={args.workers}) ...", flush=True)
    df = build_snapshot(tickers, args.workers)
    out_path = BASE / f"fundamentals_{args.out}.csv"
    df.to_csv(out_path, index=False)
    n_val = int((df["value"] == "✓").sum())
    n_ins = int((df["value"] == "–").sum())
    n_data = int(df["trailing_pe"].notna().sum())
    print(f"wrote {out_path} ({len(df)} rows) "
          f"with P/E data={n_data} VALUE✓={n_val} insufficient={n_ins}")


if __name__ == "__main__":
    main()
