#!/usr/bin/env python3
import os, sys, time, csv, yaml, math, json, pathlib, logging, datetime as dt
from typing import List, Dict, Optional
import requests
import pandas as pd

try:
    from zoneinfo import ZoneInfo  # Python 3.9+
except ImportError:
    import pytz
    class ZoneInfo:  # fallback adapter
        def __init__(self, name): self.tz = pytz.timezone(name)
        def __getattr__(self, k): return getattr(self.tz, k)

# -------------- Helpers --------------
def setup_logger():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

def read_yaml(path: pathlib.Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def ensure_dir(p: pathlib.Path):
    p.mkdir(parents=True, exist_ok=True)

def today_ymd() -> str:
    return dt.date.today().isoformat()

def save_outputs(df: pd.DataFrame, base: pathlib.Path, write_parquet: bool, write_csv: bool):
    if write_parquet:
        df.to_parquet(str(base.with_suffix(".parquet")), index=False)
    if write_csv:
        df.to_csv(str(base.with_suffix(".csv")), index=False)

# -------------- Constituents via FMP --------------
def get_sp500_from_fmp(api_key: str) -> List[str]:
    url = f"https://financialmodelingprep.com/api/v3/sp500_constituent?apikey={api_key}"
    r = requests.get(url, timeout=30); r.raise_for_status()
    return sorted({x["symbol"].upper() for x in r.json() if "symbol" in x})

def get_nasdaq100_from_fmp(api_key: str) -> List[str]:
    url = f"https://financialmodelingprep.com/api/v3/nasdaq_constituents?apikey={api_key}"
    r = requests.get(url, timeout=30); r.raise_for_status()
    return sorted({x["symbol"].upper() for x in r.json() if "symbol" in x})

# -------------- Polygon aggregates --------------
def polygon_minute_agg(api_key: str, symbol: str, day_iso: str, multiplier: int, adjusted: bool, limit: int) -> pd.DataFrame:
    base = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/{multiplier}/minute/{day_iso}/{day_iso}"
    params = {"adjusted": str(adjusted).lower(), "sort":"asc", "limit": str(limit), "apiKey": api_key}
    url = base
    results = []
    while True:
        r = requests.get(url, params=params, timeout=60)
        if r.status_code == 429:
            time.sleep(1.0); continue
        r.raise_for_status()
        data = r.json()
        if data.get("status") != "OK":
            break
        rows = data.get("results", [])
        if not rows: break
        results.extend(rows)
        nxt = data.get("next_url")
        if not nxt: break
        url = nxt; params = {"apiKey": api_key}
    if not results:
        return pd.DataFrame(columns=["ts","open","high","low","close","volume","n","vwap","symbol"])
    return pd.DataFrame([{
        "ts": x["t"], "open": x.get("o"), "high": x.get("h"), "low": x.get("l"),
        "close": x.get("c"), "volume": x.get("v"), "n": x.get("n"), "vwap": x.get("vw"),
        "symbol": symbol
    } for x in results])

def pick_1pm_snapshot(min_df: pd.DataFrame, tz_local: str, target_hms: str) -> Optional[pd.Series]:
    """Select the bar at exactly local target time if present; otherwise last bar strictly before target."""
    if min_df.empty: return None
    try:
        tz = ZoneInfo(tz_local)
    except Exception:
        tz = ZoneInfo(tz_local)
    dt_local = pd.to_datetime(min_df["ts"], unit="ms", utc=True).dt.tz_convert(tz)
    min_df = min_df.copy()
    min_df["dt_local"] = dt_local
    min_df["time_local"] = min_df["dt_local"].dt.strftime("%H:%M:%S")

    exact = min_df[min_df["time_local"] == target_hms]
    if not exact.empty:
        return exact.iloc[0]

    before = min_df[min_df["time_local"] < target_hms]
    if before.empty: return None
    return before.iloc[-1]

def main():
    setup_logger()
    if len(sys.argv) < 2:
        print("Usage: python polygon_snapshot_1pm.py /path/to/polygon_1pm_config.yaml")
        sys.exit(1)

    cfg = read_yaml(pathlib.Path(sys.argv[1]))
    api_key = cfg["polygon_api_key"]
    fmp_key = cfg["fmp_api_key"]
    start = dt.date.fromisoformat(cfg["start_date"])
    end = dt.date.fromisoformat(cfg["end_date"]) if cfg["end_date"] else dt.date.today()

    tz_local = cfg["tz_local"]
    target = cfg["snapshot_time_local"]
    mult = int(cfg["multiplier"])
    adjusted = bool(cfg["adjusted"])
    limit_per_call = int(cfg["limit_per_call"])

    out_dir = pathlib.Path(cfg["output_dir"]); ensure_dir(out_dir)

    # Build universe (S&P500, Nasdaq-100)
    univ = []
    if cfg["universe"].get("sp500", True):
        logging.info("Fetching S&P 500 from FMP")
        univ += get_sp500_from_fmp(fmp_key)
    if cfg["universe"].get("nasdaq100", True):
        logging.info("Fetching Nasdaq-100 from FMP")
        univ += get_nasdaq100_from_fmp(fmp_key)
    tickers = sorted(set(univ))
    logging.info(f"Total tickers: {len(tickers)}")

    records = []
    for i, sym in enumerate(tickers, start=1):
        sym_dir = out_dir / sym
        ensure_dir(sym_dir)

        daily_rows = []
        for n in range((end - start).days + 1):
            d = start + dt.timedelta(days=n)
            day_iso = d.isoformat()

            tries = 0
            while True:
                try:
                    df_min = polygon_minute_agg(api_key, sym, day_iso, mult, adjusted, limit_per_call)
                    break
                except Exception as e:
                    tries += 1
                    if tries >= cfg["retry"]["max_tries"]:
                        logging.warning(f"{sym} {d} failed: {e}")
                        df_min = pd.DataFrame()
                        break
                    time.sleep(cfg["retry"]["backoff_secs"] * (1.5 ** (tries-1)))

            row = pick_1pm_snapshot(df_min, tz_local, target)
            if row is None:
                continue  # holiday, no trading or early close before 1pm

            daily_rows.append({
                "date": str(row["dt_local"].date()),
                "symbol": sym,
                "open": row["open"],
                "high": row["high"],
                "low": row["low"],
                "close": row["close"],
                "volume": row["volume"],
                "trades": row["n"],
                "vwap": row["vwap"],
                "bar_ts_utc_ms": int(row["ts"]),
                "snapshot_time_local": target,
            })
            time.sleep(cfg["rate_limit_sleep_secs"])

        if not daily_rows:
            continue

        df_snap = pd.DataFrame(daily_rows).sort_values("date")
        base = (out_dir / sym / f"{sym}_1pmNY")
        save_outputs(df_snap, base, cfg["formats"]["parquet"], cfg["formats"]["csv"])

        records.append({
            "symbol": sym,
            "rows": int(len(df_snap)),
            "first": df_snap["date"].iloc[0],
            "last": df_snap["date"].iloc[-1],
            "csv": str(base.with_suffix(".csv").resolve()) if cfg["formats"]["csv"] else None,
            "parquet": str(base.with_suffix(".parquet").resolve()) if cfg["formats"]["parquet"] else None
        })

        if i % cfg["log_every"] == 0:
            logging.info(f"{i}/{len(tickers)} saved {sym} rows={len(df_snap)}")

    if records:
        manifest = pd.DataFrame(records).sort_values("symbol")
        manifest.to_csv(out_dir / "manifest_1pm.csv", index=False)
        manifest.to_parquet(out_dir / "manifest_1pm.parquet", index=False)
        logging.info(f"Done. Wrote {len(records)} symbols. Manifest at {out_dir/'manifest_1pm.csv'}")
    else:
        logging.info("No symbols produced data; check keys/date range/market days.")

if __name__ == "__main__":
    main()
