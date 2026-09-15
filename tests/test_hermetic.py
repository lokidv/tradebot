# -*- coding: utf-8 -*-
"""تست‌ها هرگز به دادهٔ واقعی دست نمی‌زنند.

یک اجرای کاملِ تست‌ها ۲۲ رویدادِ GATE_CHANGE ساختگی در ژورنالِ واقعی می‌نوشت
(«4h|zx|long مجاز شد»، «ریسک ۰٫۵٪»)؛ گزارش آن‌ها را «۴۸ تغییرِ گیت در دورهٔ
اثبات» می‌شمرد و مرحلهٔ سایه را رد می‌کرد — یعنی با هر اجرای تست ساعتِ اثبات صفر
می‌شد و مرحلهٔ سایه هرگز نمی‌توانست قبول شود.
"""
import ast
import glob
import os
import re
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "bot"))
import _hermetic  # noqa: E402,F401  — پیش از هر ماژولِ ربات: داده به پوشهٔ موقت

import autotrader  # noqa: E402
import broker  # noqa: E402
import calib  # noqa: E402
import candidates  # noqa: E402
import fill_quality  # noqa: E402
import gates  # noqa: E402
import journal  # noqa: E402
import log  # noqa: E402
import market  # noqa: E402
import paper  # noqa: E402
import paths  # noqa: E402
import report  # noqa: E402
import research  # noqa: E402
import shadow  # noqa: E402

BOT = os.path.join(ROOT, "bot")
TESTS = os.path.join(ROOT, "tests")
BOT_MODULES = {os.path.splitext(os.path.basename(p))[0] for p in glob.glob(os.path.join(BOT, "*.py"))}


def _inside(path, root):
    path, root = os.path.normcase(os.path.abspath(path)), os.path.normcase(os.path.abspath(root))
    return os.path.commonpath([path, root]) == root


class DataDirTests(unittest.TestCase):
    def test_tests_do_not_use_the_real_data_dir(self):
        self.assertFalse(_inside(paths.DATA_DIR, os.path.join(BOT, "data")))

    def test_every_data_file_lives_under_the_data_dir(self):
        files = {
            "autotrader.CFG_PATH": autotrader.CFG_PATH, "broker.CFG_PATH": broker.CFG_PATH,
            "calib.CALIB_PATH": calib.CALIB_PATH, "candidates.CAND_PATH": candidates.CAND_PATH,
            "candidates.RESULT_PATH": candidates.RESULT_PATH, "fill_quality.PATH": fill_quality.PATH,
            "gates.GATES_PATH": gates.GATES_PATH, "journal.PATH": journal.PATH,
            "log.LOG_DIR": log.LOG_DIR, "market.HIST_DIR": market.HIST_DIR, "paper.PATH": paper.PATH,
            "report.REPORT_DIR": report.REPORT_DIR, "report.HEALTH_SAMPLES": report.HEALTH_SAMPLES,
            "research.PREREG_PATH": research.PREREG_PATH, "research.FINAL_DIR": research.FINAL_DIR,
            "shadow.SHADOW_PATH": shadow.SHADOW_PATH,
        }
        for name, path in files.items():
            self.assertTrue(_inside(path, paths.DATA_DIR), f"{name} = {path}")

    def test_no_module_builds_its_own_data_path(self):
        """هر مسیرِ داده فقط از paths — وگرنه تست‌ها دوباره به دادهٔ واقعی نشت می‌کنند."""
        offenders = []
        for p in glob.glob(os.path.join(BOT, "*.py")):
            if os.path.basename(p) == "paths.py":
                continue
            with open(p, "r", encoding="utf-8") as f:
                for i, line in enumerate(f, 1):
                    if "__file__" in line and re.search(r"""["']data["']""", line):
                        offenders.append(f"{os.path.basename(p)}:{i}")
        self.assertEqual(offenders, [])


class TestFilesImportHermeticFirst(unittest.TestCase):
    def test_hermetic_precedes_every_bot_import(self):
        problems = []
        for p in sorted(glob.glob(os.path.join(TESTS, "test_*.py"))):
            with open(p, "r", encoding="utf-8") as f:
                tree = ast.parse(f.read(), filename=p)
            hermetic_line, first_bot = None, None
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    names = [a.name.split(".")[0] for a in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                    names = [node.module.split(".")[0]]
                else:
                    continue
                for n in names:
                    if n == "_hermetic":
                        hermetic_line = min(hermetic_line or node.lineno, node.lineno)
                    elif n in BOT_MODULES or n.startswith("test_"):
                        first_bot = min(first_bot or node.lineno, node.lineno)
            name = os.path.basename(p)
            if hermetic_line is None:
                problems.append(f"{name}: _hermetic import نشده")
            elif first_bot is not None and first_bot < hermetic_line:
                problems.append(f"{name}: ماژولِ ربات در خطِ {first_bot} پیش از _hermetic ({hermetic_line})")
        self.assertEqual(problems, [])


if __name__ == "__main__":
    unittest.main()
