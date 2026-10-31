"""Optional, explicit Wayback recovery. Never called by the backtest or checks."""
import argparse
import csv
import hashlib
import json
import math
import re
import subprocess
import sys
import uuid
from datetime import date, datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = "https://github.com/Acelogic/WayBackMachineStockScraper.git"
REVISION = "154db4fff923f3b3c92618ad79aa30bdc5efabf2"
CHECKOUT = ROOT / ".local-tools" / "wayback-scraper"
STAGING = ROOT / "recovered-data"
FIELDS = ("Date", "Open", "High", "Low", "Close", "Volume", "Adj Close")


def now():
    return datetime.now(timezone.utc).isoformat()


def ticker(value):
    value = value.upper().strip()
    if not re.fullmatch(r"[A-Z][A-Z0-9.\^-]{0,15}", value):
        raise ValueError(f"Invalid ticker: {value!r}")
    return value


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    # Exclusive creation prevents silently replacing an inspection or receipt.
    with path.open("x", encoding="utf-8") as output:
        json.dump(value, output, indent=2, allow_nan=False)
        output.write("\n")


def git(*args):
    return subprocess.check_output(["git", *args], text=True).strip()


def verify_checkout():
    if not (CHECKOUT / "scraper.py").is_file():
        raise ValueError("Run the setup command first.")
    if git("-C", str(CHECKOUT), "rev-parse", "HEAD") != REVISION:
        raise ValueError("Upstream checkout is not at the pinned revision.")
    if git("-C", str(CHECKOUT), "status", "--porcelain", "--untracked-files=no"):
        raise ValueError("Upstream tracked files have local changes; refusing to execute them.")


def setup():
    if not CHECKOUT.exists():
        CHECKOUT.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--no-checkout", UPSTREAM, str(CHECKOUT)], check=True)
        subprocess.run(["git", "-C", str(CHECKOUT), "checkout", "--detach", REVISION], check=True)
    verify_checkout()
    print(f"Verified upstream revision {REVISION}")
    print("Install optional requirements before fetch: python -m pip install -r requirements-recovery.txt")


def run_path(name):
    if not re.fullmatch(r"\d{8}T\d{6}Z-[0-9a-f]{8}", name):
        raise ValueError("Use a run ID printed by fetch.")
    path = STAGING / name
    if STAGING.is_symlink() or path.resolve().parent != STAGING.resolve() or path.is_symlink():
        raise ValueError("Run must be directly inside the recovery staging directory.")
    return path


def validate_csv(path):
    """Structural checks only. Do not fill gaps or infer corporate actions."""
    with path.open(newline="", encoding="utf-8-sig") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames is None or len(set(reader.fieldnames)) != len(reader.fieldnames):
            raise ValueError("Missing or duplicate CSV headers")
        if not set(FIELDS).issubset(reader.fieldnames):
            raise ValueError("Expected full upstream OHLCV and adjusted-close columns")
        dates = []
        for row in reader:
            if None in row:
                raise ValueError("Extra CSV cells")
            day = date.fromisoformat(row["Date"])
            if day.isoformat() != row["Date"] or day > datetime.now(timezone.utc).date():
                raise ValueError("Non-ISO or future observation date")
            values = {field: float(row[field]) for field in FIELDS[1:]}
            if not all(math.isfinite(value) for value in values.values()):
                raise ValueError("Nonfinite numeric observation")
            if any(values[field] <= 0 for field in ("Open", "High", "Low", "Close", "Adj Close")):
                raise ValueError("Prices must be positive")
            if values["Volume"] < 0 or not values["Volume"].is_integer():
                raise ValueError("Volume must be a nonnegative integer")
            if (values["Low"] > min(values["Open"], values["Close"]) or
                    values["High"] < max(values["Open"], values["Close"]) or
                    values["Low"] > values["High"]):
                raise ValueError("Inconsistent raw OHLC prices")
            dates.append(day.isoformat())
    if not dates or dates != sorted(set(dates)):
        raise ValueError("Expected nonempty, unique, ascending observation dates")
    return {"rows": len(dates), "start": dates[0], "end": dates[-1], "sha256": digest(path)}


def inspect_run(name):
    directory = run_path(name)
    metadata = json.loads((directory / "run.json").read_text(encoding="utf-8"))
    if metadata["revision"] != REVISION or not metadata["completed"]:
        raise ValueError("Run did not complete at the supported upstream revision.")
    reports = {}
    for symbol in metadata["tickers"]:
        symbol = ticker(symbol)
        path = directory / f"{symbol}.csv"
        try:
            if path.is_symlink():
                raise ValueError("Staged CSV must not be a symlink")
            report = validate_csv(path)
            if metadata["files"].get(path.name) != report["sha256"]:
                raise ValueError("Staged CSV changed since retrieval")
            reports[symbol] = {"valid": True, **report}
        except (OSError, ValueError, TypeError, KeyError) as error:
            reports[symbol] = {"valid": False, "error": str(error)}
    print(json.dumps(reports, indent=2))
    print("Structural validation only: inspect coverage, adjustment basis, and security identity before import.")
    return reports


def fetch(symbols, timeout):
    verify_checkout()
    symbols = list(dict.fromkeys(ticker(symbol) for symbol in symbols))
    if not symbols:
        raise ValueError("Provide at least one ticker.")
    name = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-") + uuid.uuid4().hex[:8]
    directory = run_path(name)
    directory.mkdir(parents=True)
    metadata = {"upstream": UPSTREAM, "revision": REVISION, "tickers": symbols,
                "started_utc": now(), "completed": False}
    print(f"Staging run: {name}", flush=True)
    try:
        result = subprocess.run(
            [sys.executable, "-I", str(CHECKOUT / "scraper.py"), *symbols,
             "--output-dir", str(directory), "--delay", "2"],
            cwd=CHECKOUT, timeout=timeout, check=False,
        )
        metadata["completed"] = result.returncode == 0
        metadata["returncode"] = result.returncode
    except subprocess.TimeoutExpired:
        metadata["error"] = f"Download exceeded {timeout} seconds; partial run cannot be imported."
    finally:
        metadata["finished_utc"] = now()
        metadata["files"] = {path.name: digest(path) for path in directory.glob("*.csv")}
        write_json(directory / "run.json", metadata)
    if not metadata["completed"]:
        raise ValueError(metadata.get("error", "Upstream scraper failed; inspect the output above."))
    reports = inspect_run(name)
    if not all(report["valid"] for report in reports.values()):
        raise ValueError("One or more downloads are missing or invalid. Nothing was imported.")


def import_price(name, symbol, destination=None):
    symbol = ticker(symbol)
    directory = run_path(name)
    reports = inspect_run(name)  # Validate again immediately before copying.
    metadata = json.loads((directory / "run.json").read_text(encoding="utf-8"))
    report = reports.get(symbol, {})
    if not report.get("valid"):
        raise ValueError("Requested ticker is missing or failed structural validation.")
    source = directory / f"{symbol}.csv"
    if metadata["files"].get(source.name) != report["sha256"]:
        raise ValueError("Staged CSV changed since retrieval; refusing import.")
    content = source.read_bytes()
    if hashlib.sha256(content).hexdigest() != report["sha256"]:
        raise ValueError("Staged CSV changed during import.")
    data = destination if destination is not None else ROOT / "data"
    data.mkdir(parents=True, exist_ok=True)
    target = data / source.name
    receipt = data / f"{symbol}.recovery.json"
    if target.exists() or target.is_symlink() or receipt.exists() or receipt.is_symlink():
        raise ValueError("An existing price file or receipt will not be overwritten.")
    # Reserve the receipt first; exclusive CSV creation also protects against races.
    record = {**metadata, "run": name, "ticker": symbol, "validation": report,
              "imported_utc": now(), "scope": "structural checks; no membership or delisting certification"}
    write_json(receipt, record)
    try:
        with target.open("xb") as output:
            output.write(content)
    except Exception:
        receipt.unlink()
        raise
    print(f"Imported {symbol} with provenance receipt. Existing blacklist and ticker mappings still apply.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("setup", help="Clone and verify the pinned upstream tool")
    download = commands.add_parser("fetch", help="Explicit network download into a new staging run")
    download.add_argument("tickers", nargs="+")
    download.add_argument("--timeout", type=int, default=180, choices=range(1, 901), metavar="1..900")
    inspect = commands.add_parser("inspect", help="Check staged CSVs without importing")
    inspect.add_argument("run")
    promote = commands.add_parser("import", help="Import one reviewed CSV without replacing existing data")
    promote.add_argument("run")
    promote.add_argument("ticker")
    args = parser.parse_args()
    try:
        if args.command == "setup":
            setup()
        elif args.command == "fetch":
            fetch(args.tickers, args.timeout)
        elif args.command == "inspect":
            if not all(item["valid"] for item in inspect_run(args.run).values()):
                return 1
        else:
            import_price(args.run, args.ticker)
    except (OSError, ValueError, KeyError, subprocess.SubprocessError) as error:
        print(f"Recovery error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
