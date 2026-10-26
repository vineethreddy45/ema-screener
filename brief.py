"""LLM morning brief over the daily EMA screener CSVs.

Reads today's us500/in500 reports, finds fresh BUY breakouts (vs the
previous report), and asks an LLM to write a short market brief.

Usage:
  export ANTHROPIC_API_KEY=...   # or OPENAI_API_KEY=...
  .venv/bin/python brief.py --provider anthropic
  .venv/bin/python brief.py --provider anthropic --dry-run   # print prompt only

Never paste your API key in chat. Keep it in an env var or a local .env file
that you do not commit.
"""
from __future__ import annotations

import argparse
import json
import os
import urllib.request
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

BASE = Path(__file__).resolve().parent
ET = ZoneInfo("America/New_York")
REPORTS = BASE / "reports"


def latest_reports(prefix: str, n: int = 2) -> list[Path]:
    files = sorted(REPORTS.glob(f"{prefix}-*.csv"))
    return files[-n:] if files else []


def load_csv(path: Path) -> dict:
    import pandas as pd
    df = pd.read_csv(path)
    buys = sorted(df.loc[df["signal"] == "BUY", "ticker"].tolist())
    counts = df["signal"].value_counts()
    return {
        "date": path.stem.split("-", 1)[1],
        "buys": buys,
        "buy_n": int(counts.get("BUY", 0)),
        "watch_n": int(counts.get("WATCH", 0)),
        "avoid_n": int(counts.get("AVOID", 0)),
        "total": len(df),
    }


def build_prompt(us: dict, india: dict, us_prev: dict | None,
                 india_prev: dict | None) -> str:
    def section(name: str, cur: dict, prev: dict | None) -> str:
        new_buys = sorted(set(cur["buys"]) - set(prev["buys"])) if prev else cur["buys"]
        return (
            f"{name} ({cur['date']}): {cur['total']} scanned — "
            f"{cur['buy_n']} BUY, {cur['watch_n']} WATCH, {cur['avoid_n']} AVOID.\n"
            f"BUY tickers: {', '.join(cur['buys']) or 'none'}\n"
            f"New BUYs vs previous session: {', '.join(new_buys) or 'none'}"
        )

    return f"""You are a market analyst writing a brief morning note for a retail investor.
Signals come from a 20/30 EMA screener: BUY = fresh cross above both EMAs.

{section('US Top 500 (S&P 500)', us, us_prev)}

{section('India Top 500 (Nifty 500)', india, india_prev)}

Write ~200 words: notable names among the fresh breakouts, what's new versus the
previous session, and a one-line sector read if tickers cluster. Plain language,
no hype. End with exactly: "Signals are informational only, not financial advice."
Do not invent prices or news beyond the tickers given."""


def call_anthropic(prompt: str, model: str) -> str:
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        raise SystemExit("Set ANTHROPIC_API_KEY first.")
    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=json.dumps({
            # 2000 leaves headroom: Sonnet 5.5's adaptive thinking runs by
            # default and its thinking blocks count against max_tokens.
            "model": model, "max_tokens": 2000,
            "messages": [{"role": "user", "content": prompt}],
        }).encode(),
        headers={"x-api-key": key, "anthropic-version": "2023-06-01",
                 "content-type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=90) as r:
        body = json.load(r)
    # Sonnet 5.5 can return thinking blocks before the text block; pick the
    # first text block instead of assuming content[0].
    for block in body["content"]:
        if block.get("type") == "text":
            return block["text"]
    raise SystemExit("No text block in API response.")


def call_openai(prompt: str, model: str) -> str:
    key = os.environ.get("OPENAI_API_KEY", "")
    if not key:
        raise SystemExit("Set OPENAI_API_KEY first.")
    req = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions",
        data=json.dumps({
            "model": model, "max_tokens": 600,
            "messages": [{"role": "user", "content": prompt}],
        }).encode(),
        headers={"authorization": f"Bearer {key}",
                 "content-type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=90) as r:
        body = json.load(r)
    return body["choices"][0]["message"]["content"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", choices=["anthropic", "openai"],
                    default="anthropic")
    ap.add_argument("--model", default="")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the prompt without calling the API")
    args = ap.parse_args()

    us_files = latest_reports("us500")
    in_files = latest_reports("in500")
    if not us_files or not in_files:
        raise SystemExit("No screener reports found — run screener.py first.")

    us = load_csv(us_files[-1])
    india = load_csv(in_files[-1])
    us_prev = load_csv(us_files[-2]) if len(us_files) > 1 else None
    india_prev = load_csv(in_files[-2]) if len(in_files) > 1 else None

    prompt = build_prompt(us, india, us_prev, india_prev)
    if args.dry_run:
        print(prompt)
        return

    model = args.model or ("claude-sonnet-5-5"
                           if args.provider == "anthropic"
                           else "gpt-4o-mini")
    brief = (call_anthropic(prompt, model) if args.provider == "anthropic"
             else call_openai(prompt, model))

    stamp = datetime.now(ET).strftime("%Y-%m-%d")
    out = REPORTS / f"brief-{stamp}.md"
    header = f"# Market Brief — {stamp}\n\n"
    out.write_text(header + brief + "\n")
    print(f"wrote {out}\n\n{brief}")


if __name__ == "__main__":
    main()
