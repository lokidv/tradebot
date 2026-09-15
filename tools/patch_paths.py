# -*- coding: utf-8 -*-
"""یک‌بار: همهٔ مسیرهای داده از bot/paths.py — تا تست‌ها بتوانند کلِ پوشه را جابه‌جا کنند."""
import os
import sys

BOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bot")

EDITS = {
    "autotrader.py": [('CFG_PATH = os.path.join(os.path.dirname(__file__), "data", "autobot.json")',
                       'CFG_PATH = paths.data("autobot.json")')],
    "fill_quality.py": [('PATH = os.path.join(os.path.dirname(__file__), "data", "fill_quality.json")',
                         'PATH = paths.data("fill_quality.json")')],
    "market.py": [('HIST_DIR = os.path.join(os.path.dirname(__file__), "data", "hist")',
                   'HIST_DIR = paths.data("hist")')],
    "log.py": [('LOG_DIR = os.path.join(os.path.dirname(__file__), "data", "logs")',
                'LOG_DIR = paths.data("logs")')],
    "research.py": [('DATA_DIR = os.path.join(os.path.dirname(__file__), "data")',
                     'DATA_DIR = paths.DATA_DIR')],
    "gates.py": [('GATES_PATH = os.path.join(os.path.dirname(__file__), "data", "gates.json")',
                  'GATES_PATH = paths.data("gates.json")')],
    "main.py": [('log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "logs", "rebuild.log")',
                 'log_path = paths.data("logs", "rebuild.log")')],
    "report.py": [('DATA_DIR = os.path.join(os.path.dirname(__file__), "data")',
                   'DATA_DIR = paths.DATA_DIR')],
    "paper.py": [('DATA_DIR = os.path.join(os.path.dirname(__file__), "data")',
                  'DATA_DIR = paths.DATA_DIR')],
    "broker.py": [('CFG_PATH = os.path.join(os.path.dirname(__file__), "data", "config.json")',
                   'CFG_PATH = paths.data("config.json")')],
    "journal.py": [('DATA_DIR = os.path.join(os.path.dirname(__file__), "data")',
                    'DATA_DIR = paths.DATA_DIR')],
    "candidates.py": [('DATA_DIR = os.path.join(os.path.dirname(__file__), "data")',
                       'DATA_DIR = paths.DATA_DIR')],
    "shadow.py": [('SHADOW_PATH = os.path.join(os.path.dirname(__file__), "data", "signals_log.json")',
                   'SHADOW_PATH = paths.data("signals_log.json")')],
    "calib.py": [('CALIB_PATH = os.path.join(os.path.dirname(__file__), "data", "calib.json")',
                  'CALIB_PATH = paths.data("calib.json")')],
}


def main():
    for name, edits in EDITS.items():
        path = os.path.join(BOT, name)
        with open(path, "r", encoding="utf-8", newline="") as f:
            src = f.read()
        nl = "\r\n" if "\r\n" in src else "\n"
        for old, new in edits:
            n = src.count(old)
            if n != 1:
                sys.exit(f"{name}: expected 1 occurrence, found {n}: {old}")
            src = src.replace(old, new)
        if f"{nl}import paths{nl}" not in src:
            anchor = f"{nl}import os{nl}"
            if src.count(anchor) != 1:
                sys.exit(f"{name}: 'import os' anchor count {src.count(anchor)}")
            src = src.replace(anchor, f"{nl}import os{nl}import paths{nl}", 1)
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write(src)
        print("patched", name)


if __name__ == "__main__":
    main()
