"""Offline recovery integration tests. All prices and subprocess results are fixtures."""
import contextlib
import importlib.util
import io
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("recover_prices", ROOT / "tools/recover_prices.py")
recovery = importlib.util.module_from_spec(spec)
spec.loader.exec_module(recovery)
FIXTURE = (ROOT / "tests/fixtures/wayback_prices.csv").read_text(encoding="utf-8")
RUN = "20261025T120000Z-1234abcd"


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.stage = self.root / "staging"
        self.directory = self.stage / RUN
        self.directory.mkdir(parents=True)
        self.csv = self.directory / "TEST.csv"
        self.csv.write_text(FIXTURE, encoding="utf-8")
        self.metadata = {"revision": recovery.REVISION, "upstream": recovery.UPSTREAM,
                         "tickers": ["TEST"], "completed": True,
                         "files": {"TEST.csv": recovery.digest(self.csv)}}
        self.save_metadata()
        self.patch = patch.object(recovery, "STAGING", self.stage)
        self.patch.start()
        self.addCleanup(self.patch.stop)
        self.output = contextlib.redirect_stdout(io.StringIO())
        self.output.__enter__()
        self.addCleanup(self.output.__exit__, None, None, None)

    def save_metadata(self):
        (self.directory / "run.json").write_text(json.dumps(self.metadata), encoding="utf-8")

    def test_valid_schema_and_adjustment_inputs(self):
        result = recovery.validate_csv(self.csv)
        self.assertEqual((result["rows"], result["start"], result["end"]),
                         (3, "2009-01-02", "2009-01-06"))

    def test_invalid_observations(self):
        cases = [
            FIXTURE.replace("100,104,98", "nan,104,98"),
            FIXTURE.replace("100,104,98", "0,104,98"),
            FIXTURE.replace("100,104,98", "100,99,98"),
            FIXTURE.replace("1000,51", "-1,51"),
            FIXTURE.replace("1000,51", "1.5,51"),
            FIXTURE.replace("2009-01-05", "2009-01-02"),
            FIXTURE.replace("2009-01-05", "2008-01-05"),
            FIXTURE.replace("2009-01-02", "2999-01-02"),
            FIXTURE.replace("Adj Close", "Missing"),
            FIXTURE.splitlines()[0] + "\n",
        ]
        for text in cases:
            with self.subTest(text=text):
                self.csv.write_text(text, encoding="utf-8")
                with self.assertRaises(ValueError):
                    recovery.validate_csv(self.csv)

    def test_reject_ticker_and_run_traversal(self):
        for symbol in ("../TEST", "--force", "A/B", "", "A B"):
            with self.assertRaises(ValueError):
                recovery.ticker(symbol)
        for run in ("../data", "data", RUN + "/../data"):
            with self.assertRaises(ValueError):
                recovery.run_path(run)

    def test_import_preserves_bytes_and_records_origin(self):
        destination = self.root / "data"
        recovery.import_price(RUN, "test", destination)
        self.assertEqual((destination / "TEST.csv").read_bytes(), self.csv.read_bytes())
        receipt = json.loads((destination / "TEST.recovery.json").read_text())
        self.assertEqual(receipt["revision"], recovery.REVISION)
        self.assertEqual(receipt["validation"]["sha256"], recovery.digest(self.csv))

    def test_existing_data_never_replaced(self):
        destination = self.root / "data"
        destination.mkdir()
        existing = destination / "TEST.csv"
        existing.write_text("original", encoding="utf-8")
        with self.assertRaises(ValueError):
            recovery.import_price(RUN, "TEST", destination)
        self.assertEqual(existing.read_text(), "original")
        self.assertFalse((destination / "TEST.recovery.json").exists())

    def test_existing_receipt_never_replaced(self):
        destination = self.root / "data"
        destination.mkdir()
        receipt = destination / "TEST.recovery.json"
        receipt.write_text("original", encoding="utf-8")
        with self.assertRaises(ValueError):
            recovery.import_price(RUN, "TEST", destination)
        self.assertEqual(receipt.read_text(), "original")
        self.assertFalse((destination / "TEST.csv").exists())

    def test_changed_download_rejected(self):
        self.csv.write_text(FIXTURE.replace("1000,51", "2000,51"), encoding="utf-8")
        with self.assertRaises(ValueError):
            recovery.import_price(RUN, "TEST", self.root / "data")
        self.assertFalse((self.root / "data").exists())

    def test_incomplete_or_wrong_revision_rejected(self):
        for field, value in (("completed", False), ("revision", "wrong")):
            original = self.metadata[field]
            self.metadata[field] = value
            self.save_metadata()
            with self.assertRaises(ValueError):
                recovery.inspect_run(RUN)
            self.metadata[field] = original

    def test_missing_csv_returns_failed_inspection(self):
        self.csv.unlink()
        self.assertFalse(recovery.inspect_run(RUN)["TEST"]["valid"])

    def test_process_success_without_prices_is_failure(self):
        with patch.object(recovery, "verify_checkout"), patch.object(
                recovery.subprocess, "run", return_value=subprocess.CompletedProcess([], 0)):
            with self.assertRaises(ValueError):
                recovery.fetch(["TEST"], 1)

    def test_exclusive_copy_failure_keeps_existing_file(self):
        destination = self.root / "data"
        destination.mkdir()
        original_open = Path.open
        def racing_open(path, mode="r", *args, **kwargs):
            if mode == "xb":
                path.write_text("concurrent writer", encoding="utf-8")
            return original_open(path, mode, *args, **kwargs)
        with patch.object(Path, "open", racing_open):
            with self.assertRaises(FileExistsError):
                recovery.import_price(RUN, "TEST", destination)
        self.assertEqual((destination / "TEST.csv").read_text(), "concurrent writer")
        self.assertFalse((destination / "TEST.recovery.json").exists())

    def test_checkout_revision_and_cleanliness(self):
        with patch.object(recovery, "CHECKOUT", self.root):
            (self.root / "scraper.py").write_text("# fixture")
            for answers in (("wrong",), (recovery.REVISION, " M scraper.py")):
                with patch.object(recovery, "git", side_effect=answers):
                    with self.assertRaises(ValueError):
                        recovery.verify_checkout()

    def test_fetch_uses_staging_and_explicit_tickers(self):
        def fake_run(command, **kwargs):
            output = Path(command[command.index("--output-dir") + 1])
            (output / "TEST.csv").write_text(FIXTURE, encoding="utf-8")
            self.assertIn("-I", command)
            self.assertNotIn("--all", command)
            self.assertNotIn("--merge", command)
            self.assertEqual(kwargs["timeout"], 30)
            return subprocess.CompletedProcess(command, 0)
        with patch.object(recovery, "verify_checkout"), patch.object(recovery.subprocess, "run", side_effect=fake_run):
            recovery.fetch(["test", "TEST"], 30)
        runs = [path for path in self.stage.iterdir() if path.name != RUN]
        self.assertEqual(len(runs), 1)
        record = json.loads((runs[0] / "run.json").read_text())
        self.assertEqual(record["tickers"], ["TEST"])
        self.assertTrue(record["completed"])

    def test_timeout_or_failed_process_cannot_be_imported(self):
        for effect in (subprocess.TimeoutExpired("fixture", 1), subprocess.CompletedProcess([], 1)):
            kwargs = {"side_effect": effect} if isinstance(effect, Exception) else {"return_value": effect}
            with patch.object(recovery, "verify_checkout"), patch.object(recovery.subprocess, "run", **kwargs):
                with self.assertRaises(ValueError):
                    recovery.fetch(["TEST"], 1)
        records = [json.loads(path.read_text()) for path in self.stage.glob("*/run.json") if path.parent.name != RUN]
        self.assertEqual(len(records), 2)
        self.assertTrue(all(not item["completed"] for item in records))


if __name__ == "__main__":
    unittest.main()
