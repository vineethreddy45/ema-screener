"""Streamlit app for the EMA screener.

Runs the same logic as screener.py live in the browser:
  Gate 2 — 20/30 EMA signals (BUY requires close above EMA200)
  50/200 trend layer (golden/death cross, BULLISH/BEARISH/NEUTRAL)
  Double-doji pattern tags at EMA support/resistance
  Gate 1 — VALUE overlay from the weekly fundamentals snapshots

Yahoo Finance rate-limits Streamlit Cloud's shared IPs, so downloads go
through a curl_cffi browser-impersonation session (the fix that worked
for the Intrinsic Value Checker), with retries and a plain fallback.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st
import yfinance as yf

import screener as scr

st.set_page_config(page_title="EMA Screener", page_icon="📈", layout="wide")


def _session():
    """Browser-impersonation session so Yahoo doesn't block cloud IPs."""
    try:
        from curl_cffi import requests as cffi_requests

        return cffi_requests.Session(impersonate="chrome")
    except Exception:
        try:
            import requests

            s = requests.Session()
            s.headers.update({"User-Agent": "Mozilla/5.0 (Windows NT 10.0; "
                                           "Win64; x64) AppleWebKit/537.36"})
            return s
        except Exception:
            return None


@st.cache_data(ttl=900, show_spinner=False)
def run_scan(tickers: tuple[str, ...], out_name: str) -> pd.DataFrame:
    """Download 2y of daily bars and score every ticker with screener.py."""
    tickers = list(tickers)
    data = None
    last_err = None
    for _ in range(2):
        try:
            data = yf.download(tickers, period="2y", interval="1d",
                               group_by="ticker", progress=False,
                               auto_adjust=True, threads=True,
                               session=_session())
            if data is not None and len(data):
                break
        except Exception as e:  # noqa: BLE001 - retry once, then report
            last_err = e
    if data is None or not len(data):
        raise RuntimeError(f"Yahoo Finance returned no data. {last_err or ''}")

    rows = []
    single = len(tickers) == 1
    for t in tickers:
        try:
            df = data if single else (
                data[t] if t in data.columns.get_level_values(0) else None)
            if df is None or "Close" not in df.columns:
                rows.append(scr.empty_row(t, "NO DATA"))
                continue
            ohlc = df[["Open", "High", "Low", "Close"]].dropna()
            if len(ohlc) < 35:
                rows.append(scr.empty_row(t, "NO DATA"))
                continue
            r = scr.signal_for(ohlc)
            r["ticker"] = t
            rows.append(r)
        except Exception as e:  # noqa: BLE001 - keep scanning the rest
            rows.append(scr.empty_row(t, f"ERROR: {e}"))
    out = pd.DataFrame(rows)
    return scr.merge_fundamentals(out, out_name)


UNIVERSES = {
    "US Top 500 (S&P 500)": ("tickers_us500.txt", "", "us500"),
    "India Top 500 (Nifty 500)": ("tickers_in500.txt", ".NS", "in500"),
    "Custom tickers": (None, "", "custom"),
}

st.title("📈 EMA Screener — US & India Top 500")
st.caption("20/30 EMA signals · 50/200 trend layer · double-doji patterns · "
           "Gate 1 value overlay. Signals are informational only, "
           "not financial advice.")

with st.sidebar:
    st.header("Scan")
    choice = st.selectbox("Universe", list(UNIVERSES))
    custom = ""
    if choice == "Custom tickers":
        custom = st.text_area(
            "Tickers (comma or space separated; add .NS for NSE India)",
            "AAPL, MSFT, NVDA, RELIANCE.NS, TCS.NS")
    only_buy = st.checkbox("Show BUY only", value=False)
    go = st.button("Run scan", type="primary")

if go:
    fname, suffix, out_name = UNIVERSES[choice]
    if fname:
        tickers = scr.load_tickers(fname, suffix)
    else:
        tickers = [t.strip().upper()
                   for t in custom.replace(",", " ").split() if t.strip()]
    if not tickers:
        st.error("No tickers to scan.")
        st.stop()
    with st.spinner(f"Scanning {len(tickers)} tickers — a full 500-name "
                    "scan takes a couple of minutes…"):
        try:
            df = run_scan(tuple(tickers), out_name)
        except Exception as e:  # noqa: BLE001
            st.error(str(e))
            st.stop()
    st.session_state["df"] = df
    st.session_state["universe"] = choice

df = st.session_state.get("df")
if df is None:
    st.info("Pick a universe in the sidebar and hit **Run scan**.")
    st.stop()

counts = df["signal"].value_counts()
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("BUY", int(counts.get("BUY", 0)))
c2.metric("WATCH", int(counts.get("WATCH", 0)))
c3.metric("AVOID", int(counts.get("AVOID", 0)))
if "trend" in df.columns:
    tc = df["trend"].astype(str).value_counts()
    c4.metric("Golden crosses", int(tc.get("GOLDEN CROSS", 0)))
    c5.metric("Death crosses", int(tc.get("DEATH CROSS", 0)))

if "value" in df.columns:
    dual = df[df.apply(scr._is_dual_confirmed, axis=1)]
    st.subheader(f"Dual-confirmed — EMA signal + VALUE ✓ ({len(dual)})")
    if len(dual):
        st.dataframe(dual, width="stretch", hide_index=True)
    else:
        st.caption("No dual-confirmed names in this scan.")

show = df[df["signal"] == "BUY"] if only_buy else df
order = {"BUY": 0, "WATCH": 1, "AVOID": 2}
show = show.assign(_o=show["signal"].map(order).fillna(3)) \
           .sort_values(["_o", "ticker"]).drop(columns="_o")
st.subheader(f"Results — {st.session_state.get('universe', '')} "
             f"({len(show)} shown)")
st.dataframe(show, width="stretch", hide_index=True)
st.download_button("Download CSV", show.to_csv(index=False),
                   file_name="ema-scan.csv", mime="text/csv")

with st.expander("How the signals work"):
    st.markdown(
        "- **BUY** — close freshly crossed above both EMA20 and EMA30 "
        "**and** closed above EMA200\n"
        "- **WATCH** — within 2% of EMA20/EMA30, or a fresh breakout still "
        "below EMA200\n"
        "- **AVOID** — close below both EMAs\n"
        "- **Trend (50/200)** — GOLDEN/DEATH CROSS on a fresh EMA50↔EMA200 "
        "cross; BULLISH = EMA50 > EMA200 and close > EMA200; BEARISH the "
        "mirror; NEUTRAL otherwise\n"
        "- **Pattern** — double doji at EMA support/resistance "
        "(DD-SUPPORT / DD-RESIST)\n"
        "- **VALUE ✓ (Gate 1)** — ROE > 15%, FCF > 0, D/E < ~1.0, PEG < 1.5 "
        "(or P/E < 25), from the weekly fundamentals snapshot")
