# EMA Screener — US Top 500 & India Top 500

Daily stock screener over the S&P 500 (503 names) and the Nifty 500, built
around exponential moving averages with a value/quality overlay. It produces
an HTML + CSV report per universe per day.

## Signal layers

**Gate 2 — EMA timing (20/30)**
- **BUY** — close freshly crossed above both EMA20 and EMA30 *and* closed
  above EMA200 (long-term trend agrees)
- **WATCH** — price within 2% of EMA20 or EMA30, or a fresh 20/30 breakout
  that is still below EMA200
- **AVOID** — close below both EMAs

**50/200 trend layer**
- **GOLDEN CROSS / DEATH CROSS** — EMA50 crossed EMA200 on the latest bar
- **BULLISH** — EMA50 above EMA200 and close above EMA200
- **BEARISH** — the mirror image; **NEUTRAL** otherwise

**Pattern tag** — two consecutive doji candles within 2% of EMA20/EMA30 are
tagged DD-SUPPORT or DD-RESIST (awaiting confirmation, or confirmed when the
next bar closes beyond the pair's range).

**Gate 1 — value/quality (`fundamentals.py`)** — a weekly snapshot of P/E,
PEG, ROE, debt/equity and free cash flow per ticker. A name passes VALUE ✓
when ROE > 15%, FCF > 0, D/E < ~1.0 and PEG < 1.5 (or P/E < 25). Reports get
a "Dual-confirmed" section: EMA signal **and** VALUE ✓.

## Usage

```bash
pip install -r requirements.txt

# US Top 500
python screener.py --tickers tickers_us500.txt --label "US Top 500" --out us500

# India Top 500 (NSE symbols)
python screener.py --tickers tickers_in500.txt --suffix .NS --label "India Top 500" --out in500

# Weekly fundamentals snapshot (writes fundamentals_<universe>.csv)
python fundamentals.py
```

Reports are written to `reports/` as `<universe>-YYYY-MM-DD.html` and `.csv`.
Price history is fetched with a 2-year window so EMA200 is well converged.

## Streamlit app

`app.py` runs the same screener live in the browser — pick the US or India
universe (or enter custom tickers), hit **Run scan**, and get the metrics,
dual-confirmed section and full results table with a CSV download. It reuses
`screener.py`'s logic and the fundamentals snapshots, and routes Yahoo
downloads through a browser-impersonation session (Yahoo rate-limits
Streamlit Cloud's shared IPs). Deploy: point Streamlit Cloud at this repo
with `app.py` as the main file.

`brief.py` builds an optional LLM morning brief over the daily CSVs.

## Disclaimer

Signals are informational only, not financial advice.
