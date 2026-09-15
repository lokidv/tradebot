# -*- coding: utf-8 -*-
"""تست‌های پذیرشِ فاز ۱g — خطا دیگر نامرئی نیست.

معیار: بلعیدنِ استثنا مجاز است (رفتار عوض نمی‌شود)، ولی بی‌صدا بودنش نه.
"""
import json
import logging
import os
import sys
import tempfile
import unittest


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "bot"))
import _hermetic  # noqa: E402,F401  — پیش از هر ماژولِ ربات: داده به پوشهٔ موقت

import log  # noqa: E402


class _LogMixin:
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self._old_dir, self._old_path = log.LOG_DIR, log.LOG_PATH
        log.LOG_DIR = self.tmp.name
        log.LOG_PATH = os.path.join(self.tmp.name, "bot.log")
        lg = logging.getLogger("ctp")
        for h in list(lg.handlers):           # ماژول‌های دیگر شاید لاگر را ساخته باشند
            h.close()
            lg.removeHandler(h)
        log._logger = None
        log._seen.clear()
        for k in log._counts:
            log._counts[k] = 0

    def tearDown(self):
        lg = log.get_logger()
        for h in list(lg.handlers):
            h.close()
            lg.removeHandler(h)
        log._logger = None
        log.LOG_DIR, log.LOG_PATH = self._old_dir, self._old_path
        log._seen.clear()
        self.tmp.cleanup()

    def rows(self):
        if not os.path.exists(log.LOG_PATH):
            return []
        with open(log.LOG_PATH, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]


class StructuredOutputTests(_LogMixin, unittest.TestCase):
    def test_each_line_is_valid_json_with_the_expected_fields(self):
        log.info("سلام", symbol="BTCUSDT", tf="1h")
        row = self.rows()[0]
        self.assertEqual(row["level"], "info")
        self.assertEqual(row["msg"], "سلام")
        self.assertEqual(row["symbol"], "BTCUSDT")
        self.assertEqual(row["tf"], "1h")
        self.assertIsInstance(row["ts"], float)

    def test_levels_are_counted(self):
        log.info("a")
        log.warn("b")
        log.error("c")
        c = log.counts()
        self.assertEqual((c["info"], c["warning"], c["error"]), (1, 1, 1))

    def test_tail_can_filter_by_level(self):
        log.info("a")
        log.warn("b")
        self.assertEqual(len(log.tail(10)), 2)
        self.assertEqual([r["msg"] for r in log.tail(10, level="warning")], ["b"])


class SwallowedExceptionTests(_LogMixin, unittest.TestCase):
    def test_exc_records_the_traceback_and_the_call_site(self):
        try:
            raise ValueError("boom")
        except ValueError:
            log.exc("fetching funding")
        row = self.rows()[0]
        self.assertEqual(row["level"], "warning")
        self.assertIn("fetching funding", row["msg"])
        self.assertIn("ValueError: boom", row["exc"])
        self.assertIn("test_logging.py", row["where"])
        self.assertEqual(row["func"], "test_exc_records_the_traceback_and_the_call_site")

    def test_repeated_failures_from_one_site_are_deduplicated(self):
        for _ in range(50):
            try:
                raise RuntimeError("network down")
            except RuntimeError:
                log.exc("kline fetch")
        self.assertEqual(len(self.rows()), 1, "یک قطعی شبکه نباید لاگ را پر کند")

    def test_suppressed_repeats_are_counted_in_the_next_record(self):
        for _ in range(4):
            try:
                raise RuntimeError("x")
            except RuntimeError:
                log.exc("a")
        log._seen.clear()                     # مثل گذشتنِ پنجرهٔ ۶۰ ثانیه
        try:
            raise RuntimeError("x")
        except RuntimeError:
            log.exc("a")
        self.assertEqual(len(self.rows()), 2)

    def test_two_different_sites_are_logged_separately(self):
        def site_a():
            try:
                raise KeyError("a")
            except KeyError:
                log.exc()

        def site_b():
            try:
                raise KeyError("b")
            except KeyError:
                log.exc()

        site_a()
        site_b()
        rows = self.rows()
        self.assertEqual(len(rows), 2)
        self.assertNotEqual(rows[0]["where"], rows[1]["where"])


class ImportOrderTests(unittest.TestCase):
    """``log.exc()`` پیش از ``import log`` یعنی NameError هنگامِ بالا آمدنِ برنامه.

    این دقیقاً در main.py رخ داد: بلوکِ پیکربندیِ کنسول در خطِ ۱۲ اجرا می‌شد و
    ``import log`` در خطِ ۲۵ بود؛ هر خطا در reconfigure برنامه را از کار می‌انداخت.
    """

    def test_log_is_imported_before_its_first_module_level_use(self):
        offenders = []
        bot = os.path.join(ROOT, "bot")
        for name in sorted(os.listdir(bot)):
            if not name.endswith(".py") or name == "log.py":
                continue
            with open(os.path.join(bot, name), "r", encoding="utf-8") as f:
                lines = f.readlines()
            uses = [i for i, ln in enumerate(lines) if "log.exc(" in ln or "log.warn(" in ln
                    or "log.info(" in ln or "log.error(" in ln]
            if not uses:
                continue
            imports = [i for i, ln in enumerate(lines) if ln.strip() == "import log"]
            if not imports:
                offenders.append(f"{name}: log بدونِ import")
                continue
            # فقط استفاده‌های سطحِ ماژول (بدونِ تورفتگیِ تابع) پیش از import خطرناک‌اند
            for i in uses:
                if i < imports[0]:
                    offenders.append(f"{name}:{i + 1} پیش از import log")
        self.assertEqual(offenders, [], offenders)


class SourceAuditTests(unittest.TestCase):
    """هیچ ``except: pass`` خاموشی نباید در ماژول‌های هسته باقی مانده باشد."""

    CORE = ["main.py", "market.py", "calib.py", "autotrader.py", "engine.py",
            "paper.py", "shadow.py", "candidates.py", "broker.py", "advisor.py"]

    def test_no_bare_silent_handler_remains(self):
        offenders = []
        for name in self.CORE:
            path = os.path.join(ROOT, "bot", name)
            with open(path, "r", encoding="utf-8") as f:
                lines = f.readlines()
            for i, line in enumerate(lines[:-1]):
                if "silent-ok" in line:          # استثنای صریح و مستند
                    continue
                if line.lstrip().startswith("except ") and line.rstrip().endswith((":", "BLE001")):
                    nxt = lines[i + 1].strip()
                    if nxt == "pass":
                        offenders.append(f"{name}:{i + 2}")
        self.assertEqual(offenders, [], f"خطای بی‌صدا باقی مانده: {offenders}")


if __name__ == "__main__":
    unittest.main()
