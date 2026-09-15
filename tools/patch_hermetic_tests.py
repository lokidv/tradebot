# -*- coding: utf-8 -*-
"""یک‌بار: «import _hermetic» را درست پس از افزودنِ bot به sys.path در هر فایلِ تست بگذار."""
import glob
import os
import sys

TESTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tests")
ANCHOR = 'sys.path.insert(0, os.path.join(ROOT, "bot"))'
LINE = "import _hermetic  # noqa: E402,F401  — پیش از هر ماژولِ ربات: داده به پوشهٔ موقت"


def main():
    for path in sorted(glob.glob(os.path.join(TESTS, "test_*.py"))):
        with open(path, "r", encoding="utf-8", newline="") as f:
            src = f.read()
        if "import _hermetic" in src:
            continue
        nl = "\r\n" if "\r\n" in src else "\n"
        if src.count(ANCHOR) != 1:
            sys.exit(f"{os.path.basename(path)}: anchor count {src.count(ANCHOR)}")
        src = src.replace(ANCHOR, ANCHOR + nl + LINE, 1)
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write(src)
        print("patched", os.path.basename(path))


if __name__ == "__main__":
    main()
