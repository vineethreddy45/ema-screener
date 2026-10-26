"""EMA screener (20/30 signals + 50/200 trend) — US and India universes.

Signals (per user's rules, unchanged):
  BUY   - close crossed above BOTH EMA20 and EMA30 (fresh breakout)
          AND close is above EMA200 (long-term trend agrees)
  WATCH - price is near (within 2%) of EMA20 or EMA30, no fresh cross
  AVOID - close below BOTH EMAs (downgrade / breakdown)

Trend layer (EMA50 / EMA200, added Oct 26, 2026):
  GOLDEN CROSS - EMA50 crossed above EMA200 on the latest bar
  DEATH CROSS  - EMA50 crossed below EMA200 on the latest bar
  BULLISH      - EMA50 above EMA200 and close above EMA200
  BEARISH      - EMA50 below EMA200 and close below EMA200
  NEUTRAL      - anything else; blank when fewer than 200 bars exist

Pattern tag (Pattern column):
  DD-SUPPORT / DD-RESIST - two consecutive doji candles within 2% of
  EMA20/EMA30 (the EMAs acting as dynamic support/resistance), either
  fresh (awaiting confirmation) or confirmed by the next bar closing
  beyond the pair's range in the pattern's direction.

Usage:
  .venv/bin/python screener.py --tickers tickers_us500.txt --label "US Top 500" --out us500
  .venv/bin/python screener.py --tickers tickers_in500.txt --suffix .NS --label "India Top 500" --out in500
"""
from __future__ import annotations

import argparse
import html
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import yfinance as yf

BASE = Path(__file__).resolve().parent
ET = ZoneInfo("America/New_York")
NEAR_PCT = 0.02  # within 2% of an EMA counts as "near"
DOJI_BODY_RATIO = 0.10  # body <= 10% of the bar's range counts as a doji


def load_tickers(path: str, suffix: str) -> list[str]:
    tickers = []
    for line in (BASE / path).read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            tickers.append(line.upper() + suffix)
    # de-dupe while preserving order
    return list(dict.fromkeys(tickers))


def _is_doji(o: float, h: float, l: float, c: float) -> bool:
    rng = h - l
    return rng > 0 and abs(c - o) <= DOJI_BODY_RATIO * rng


def pattern_for(ohlc: pd.DataFrame, ema20: pd.Series, ema30: pd.Series) -> str:
    """Detect a double doji at EMA support/resistance in the last 3 bars.

    A double doji = two consecutive doji candles. The EMAs act as dynamic
    support/resistance, so the pair must sit within NEAR_PCT of EMA20 or
    EMA30. If the pair is the two most recent bars the pattern is fresh
    (awaiting confirmation); if a third bar followed it and closed beyond
    the pair's range in the pattern's direction, it is marked confirmed.
    """
    n = len(ohlc)
    if n < 2:
        return ""
    o = ohlc["Open"].astype(float)
    h = ohlc["High"].astype(float)
    l = ohlc["Low"].astype(float)
    c = ohlc["Close"].astype(float)

    def pair_at(i2: int, i1: int, ref: int) -> str:
        """Return DD-SUPPORT / DD-RESIST if bars i2,i1 are both dojis sitting
        near an EMA as of bar `ref`, else ''."""
        if not (_is_doji(o.iloc[i2], h.iloc[i2], l.iloc[i2], c.iloc[i2])
                and _is_doji(o.iloc[i1], h.iloc[i1], l.iloc[i1], c.iloc[i1])):
            return ""
        price = float(c.iloc[ref])
        e20, e30 = float(ema20.iloc[ref]), float(ema30.iloc[ref])
        nearest = e20 if abs(price - e20) / e20 <= abs(price - e30) / e30 else e30
        if abs(price - nearest) / nearest > NEAR_PCT:
            return ""
        return "DD-SUPPORT" if price >= nearest else "DD-RESIST"

    # Fresh pair: the last two bars.
    fresh = pair_at(-2, -1, -1)
    if fresh:
        return fresh + " (awaiting confirmation)"

    # Confirmed pair: dojis at -3,-2 and the last bar closed beyond the
    # pair's range in the direction the level implies.
    if n >= 3:
        base = pair_at(-3, -2, -3)
        if base:
            pair_high = max(float(h.iloc[-3]), float(h.iloc[-2]))
            pair_low = min(float(l.iloc[-3]), float(l.iloc[-2]))
            last_c, last_o = float(c.iloc[-1]), float(o.iloc[-1])
            if base == "DD-SUPPORT" and last_c > pair_high and last_c > last_o:
                return "DD-SUPPORT (confirmed)"
            if base == "DD-RESIST" and last_c < pair_low and last_c < last_o:
                return "DD-RESIST (confirmed)"
    return ""


def signal_for(ohlc: pd.DataFrame) -> dict:
    """Compute signal from the last two daily bars plus pattern tags."""
    close = ohlc["Close"].astype(float)
    ema20 = close.ewm(span=20, adjust=False).mean()
    ema30 = close.ewm(span=30, adjust=False).mean()
    ema50 = close.ewm(span=50, adjust=False).mean()
    ema200 = close.ewm(span=200, adjust=False).mean()

    c, pc = float(close.iloc[-1]), float(close.iloc[-2])
    e20, pe20 = float(ema20.iloc[-1]), float(ema20.iloc[-2])
    e30, pe30 = float(ema30.iloc[-1]), float(ema30.iloc[-2])
    date = close.index[-1].strftime("%Y-%m-%d")

    above_both = c > e20 and c > e30
    below_both = c < e20 and c < e30
    fresh_cross_up = above_both and (pc <= pe20 or pc <= pe30)
    fresh_cross_down = below_both and (pc >= pe20 or pc >= pe30)
    near = (abs(c - e20) / e20 <= NEAR_PCT) or (abs(c - e30) / e30 <= NEAR_PCT)

    above_200 = len(close) >= 200 and c > float(ema200.iloc[-1])
    if fresh_cross_up and above_200:
        signal = "BUY"
    elif fresh_cross_up:
        # Fresh 20/30 breakout against (or unverifiable vs) the EMA200
        # trend line — downgraded per user's rule (Oct 26, 2026).
        signal = "WATCH"
    elif fresh_cross_down:
        signal = "AVOID"
    elif above_both and near:
        signal = "WATCH"
    elif below_both:
        signal = "AVOID"
    elif near:
        signal = "WATCH"
    else:
        signal = "AVOID" if c < e20 else "WATCH"

    # 50/200 trend layer — only meaningful with a full 200-bar history.
    e50_out, e200_out, trend = "", "", ""
    if len(close) >= 50:
        e50_out = round(float(ema50.iloc[-1]), 2)
    if len(close) >= 200:
        e50, pe50 = float(ema50.iloc[-1]), float(ema50.iloc[-2])
        e200 = float(ema200.iloc[-1])
        pe200 = float(ema200.iloc[-2])
        e200_out = round(e200, 2)
        if pe50 <= pe200 and e50 > e200:
            trend = "GOLDEN CROSS"
        elif pe50 >= pe200 and e50 < e200:
            trend = "DEATH CROSS"
        elif e50 > e200 and c > e200:
            trend = "BULLISH"
        elif e50 < e200 and c < e200:
            trend = "BEARISH"
        else:
            trend = "NEUTRAL"

    return {
        "date": date,
        "signal": signal,
        "pattern": pattern_for(ohlc, ema20, ema30),
        "close": round(c, 2),
        "ema20": round(e20, 2),
        "ema30": round(e30, 2),
        "ema50": e50_out,
        "ema200": e200_out,
        "trend": trend,
        "chg_pct": round((c - pc) / pc * 100, 2),
    }


def empty_row(ticker: str, note: str) -> dict:
    return {"ticker": ticker, "signal": note, "pattern": "", "date": "",
            "close": "", "ema20": "", "ema30": "", "ema50": "",
            "ema200": "", "trend": "", "chg_pct": ""}


def scan(tickers: list[str]) -> pd.DataFrame:
    rows = []
    try:
        data = yf.download(tickers, period="2y", interval="1d",
                           group_by="ticker", progress=False,
                           auto_adjust=True, threads=True)
    except Exception as e:  # noqa: BLE001 - total download failure
        return pd.DataFrame([empty_row(t, f"ERROR: {e}") for t in tickers])

    single = len(tickers) == 1
    for t in tickers:
        try:
            if single:
                df = data
            else:
                df = data[t] if t in data.columns.get_level_values(0) else None
            if df is None or "Close" not in df.columns:
                rows.append(empty_row(t, "NO DATA"))
                continue
            ohlc = df[["Open", "High", "Low", "Close"]].dropna()
            if len(ohlc) < 35:
                rows.append(empty_row(t, "NO DATA"))
                continue
            r = signal_for(ohlc)
            r["ticker"] = t
            rows.append(r)
        except Exception as e:  # noqa: BLE001 - keep scanning the rest
            rows.append(empty_row(t, f"ERROR: {e}"))
    return pd.DataFrame(rows)


def merge_fundamentals(df: pd.DataFrame, out_name: str) -> pd.DataFrame:
    """Merge the Gate 1 fundamentals snapshot (fundamentals_<out>.csv,
    built by fundamentals.py) into scan results when it exists.

    Adds columns pe, peg, roe, de (ratio, not yfinance %), value and
    value_reason. When no snapshot exists the frame is returned
    unchanged, so the screener behaves exactly as before.
    """
    snap_path = BASE / f"fundamentals_{out_name}.csv"
    if not snap_path.exists():
        return df
    try:
        snap = pd.read_csv(snap_path)
    except Exception:  # noqa: BLE001 - a bad snapshot must not break the run
        return df
    if "ticker" not in snap.columns:
        return df
    df = df.copy()
    lookup = {}
    for _, s in snap.iterrows():
        def _f(col: str):
            v = s.get(col)
            return None if v is None or (isinstance(v, float) and pd.isna(v)) else v
        de_raw = _f("debt_to_equity")
        lookup[str(s["ticker"])] = {
            "pe": _f("trailing_pe"),
            "peg": _f("peg"),
            "roe": _f("roe"),
            # snapshot stores yfinance's percentage (85.3); reports use the ratio
            "de": round(float(de_raw) / 100, 2) if de_raw is not None else "",
            "value": str(_f("value") or ""),
            "value_reason": str(_f("value_reason") or ""),
        }
    for col in ["pe", "peg", "roe", "de", "value", "value_reason"]:
        df[col] = [lookup.get(str(t), {}).get(col, "") for t in df["ticker"]]
    return df


def _is_dual_confirmed(r) -> bool:
    """EMA signal (BUY or DD-SUPPORT pattern) AND Gate 1 VALUE ✓."""
    return str(r.get("value", "")) == "✓" and (
        r["signal"] == "BUY"
        or str(r["pattern"]).startswith("DD-SUPPORT")
    )


def to_html(df: pd.DataFrame, label: str, note: str) -> str:
    now = datetime.now(ET).strftime("%A, %B %d, %Y %I:%M %p ET")
    order = {"BUY": 0, "WATCH": 1, "AVOID": 2}
    df = df.copy()
    df["_o"] = df["signal"].map(order).fillna(3)
    df = df.sort_values(["_o", "ticker"]).drop(columns="_o")

    has_fund = "value" in df.columns

    def fund_cells(r) -> str:
        def fmt(v, f):
            return f(v) if isinstance(v, (int, float)) and not pd.isna(v) else ""
        val = str(r.get("value", ""))
        reason = str(r.get("value_reason", ""))
        val_cell = (f"{val} <small>({html.escape(reason)})</small>"
                    if val and reason else val)
        return (
            f"<td>{fmt(r.get('pe'), lambda v: f'{v:.1f}')}</td>"
            f"<td>{fmt(r.get('peg'), lambda v: f'{v:.2f}')}</td>"
            f"<td>{fmt(r.get('roe'), lambda v: f'{v * 100:.0f}%')}</td>"
            f"<td>{fmt(r.get('de'), lambda v: f'{v:.2f}')}</td>"
            f"<td>{val_cell}</td>"
        )

    colors = {"BUY": "#d9ead3", "WATCH": "#fff2cc", "AVOID": "#f4cccc"}
    cells = []
    for _, r in df.iterrows():
        bg = colors.get(r["signal"], "#ffffff")
        pat = str(r["pattern"])
        pat_cell = f"<b>{html.escape(pat)}</b>" if pat else ""
        cells.append(
            "<tr style='background:{}'>".format(bg)
            + "".join(f"<td>{html.escape(str(r[c]))}</td>"
                      for c in ["ticker", "signal", "close",
                                "ema20", "ema30", "chg_pct", "date"])
            + f"<td>{html.escape(str(r.get('ema50', '')))}</td>"
            + f"<td>{html.escape(str(r.get('ema200', '')))}</td>"
            + f"<td>{html.escape(str(r.get('trend', '')))}</td>"
            + f"<td>{pat_cell}</td>"
            + (fund_cells(r) if has_fund else "")
            + "</tr>"
        )
    counts = df["signal"].value_counts()
    summary = " &nbsp; ".join(
        f"<b>{s}</b>: {counts.get(s, 0)}" for s in ["BUY", "WATCH", "AVOID"]
    )
    n_pat = int((df["pattern"].astype(str) != "").sum())
    summary += f"<br><b>Double-doji at EMA level</b>: {n_pat}"
    if "trend" in df.columns:
        tcounts = df["trend"].astype(str).value_counts()
        cl = pd.to_numeric(df["close"], errors="coerce")
        e2 = pd.to_numeric(df["ema200"], errors="coerce")
        n_above200 = int((cl > e2).sum())
        summary += (
            f"<br><b>50/200 trend</b>: Golden crosses: "
            f"{tcounts.get('GOLDEN CROSS', 0)} &nbsp; Death crosses: "
            f"{tcounts.get('DEATH CROSS', 0)} &nbsp; Bullish: "
            f"{tcounts.get('BULLISH', 0)} &nbsp; Bearish: "
            f"{tcounts.get('BEARISH', 0)} &nbsp; Above EMA200: {n_above200}"
        )

    dual_section = ""
    if has_fund:
        dual = df[df.apply(_is_dual_confirmed, axis=1)]
        summary += f"<br><b>Dual-confirmed (EMA + VALUE ✓)</b>: {len(dual)}"
        if len(dual):
            rows = []
            for _, r in dual.iterrows():
                pat = str(r["pattern"])
                pat_cell = f"<b>{html.escape(pat)}</b>" if pat else ""
                rows.append(
                    "<tr style='background:#cfe2f3'>"
                    + "".join(f"<td>{html.escape(str(r[c]))}</td>"
                              for c in ["ticker", "signal", "close"])
                    + f"<td>{pat_cell}</td>"
                    + fund_cells(r)
                    + "</tr>"
                )
            dual_section = f"""
<h3>Dual-confirmed — EMA signal + VALUE ✓</h3>
<p>Gate 1 (value/quality) says worth owning; Gate 2 (EMA) says act now.</p>
<table border="1" cellpadding="6" cellspacing="0">
<tr><th>Ticker</th><th>Signal</th><th>Close</th><th>Pattern</th>
<th>P/E</th><th>PEG</th><th>ROE</th><th>D/E</th><th>VALUE</th></tr>
{''.join(rows)}
</table>"""
    fund_headers = ("<th>P/E</th><th>PEG</th><th>ROE</th><th>D/E</th>"
                    "<th>VALUE</th>") if has_fund else ""
    return f"""<html><body style="font-family:Arial,sans-serif">
<h2>EMA Screener (20/30 + 50/200) — {html.escape(label)} — {now}</h2>
<p>{summary}</p>
{dual_section}
<table border="1" cellpadding="6" cellspacing="0">
<tr><th>Ticker</th><th>Signal</th><th>Close</th><th>EMA20</th>
<th>EMA30</th><th>Chg%</th><th>Bar Date</th><th>EMA50</th><th>EMA200</th>
<th>Trend (50/200)</th><th>Pattern</th>{fund_headers}</tr>
{''.join(cells)}
</table>
<p style="color:#666;font-size:12px">Signals are informational only, not financial advice.
{html.escape(note)}</p>
</body></html>"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", default="tickers.txt")
    ap.add_argument("--suffix", default="")
    ap.add_argument("--label", default="Watchlist")
    ap.add_argument("--out", default="ema-report")
    ap.add_argument("--note", default="Based on the latest available daily bar.")
    args = ap.parse_args()

    tickers = load_tickers(args.tickers, args.suffix)
    df = scan(tickers)
    df = merge_fundamentals(df, args.out)
    out = BASE / "reports"
    out.mkdir(exist_ok=True)
    stamp = datetime.now(ET).strftime("%Y-%m-%d")
    html_path = out / f"{args.out}-{stamp}.html"
    html_path.write_text(to_html(df, args.label, args.note))
    df.to_csv(out / f"{args.out}-{stamp}.csv", index=False)
    counts = df["signal"].value_counts()
    extra = ""
    if "value" in df.columns:
        n_dual = int(df.apply(_is_dual_confirmed, axis=1).sum())
        extra = f" VALUE✓ dual-confirmed={n_dual}"
    trend_extra = ""
    if "trend" in df.columns:
        tc = df["trend"].astype(str).value_counts()
        trend_extra = (f" GC={tc.get('GOLDEN CROSS', 0)}"
                       f" DC={tc.get('DEATH CROSS', 0)}")
    print(f"wrote {html_path} ({len(df)} tickers) "
          f"BUY={counts.get('BUY', 0)} WATCH={counts.get('WATCH', 0)} "
          f"AVOID={counts.get('AVOID', 0)}{extra}{trend_extra}")


if __name__ == "__main__":
    main()
