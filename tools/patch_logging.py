# -*- coding: utf-8 -*-
"""پچِ فاز ۱g — جایگزینیِ ``except ...: pass`` های خاموش با لاگِ ساخت‌یافته.

رفتارِ برنامه عوض نمی‌شود (استثنا همچنان بلعیده می‌شود) ولی دیگر نامرئی نیست.
فقط بلوک‌هایی هدف‌اند که بدنه‌شان دقیقاً ``pass`` است؛ بلوک‌هایی که مسیرِ
جایگزین دارند دست‌نخورده می‌مانند.
"""
import io
import os
import re

BOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bot")
TARGETS = ["main.py", "market.py", "calib.py", "autotrader.py", "engine.py",
           "advisor.py", "broker.py", "shadow.py", "edge_book.py", "meta_gate.py",
           "candidates.py", "features.py", "paper.py"]

# except ...:  [# comment]\n <indent> pass
PAT = re.compile(r"(?P<head>except [^\n:]*:[^\n]*\n)(?P<ind>[ \t]+)pass(?=\s*\n)")

IMPORT_ANCHORS = [
    ("main.py", "import market\n", "import log\nimport market\n"),
    ("market.py", "import httpx\n", "import httpx\n\nimport log\n"),
    ("calib.py", "import bracket\n", "import bracket\n"),
    ("autotrader.py", "import journal\n", "import journal\nimport log\n"),
    ("engine.py", "import bracket\n", "import bracket\n"),
    ("advisor.py", "from datetime import datetime, timezone\n",
     "from datetime import datetime, timezone\n\nimport log\n"),
    ("broker.py", "import httpx\n", "import httpx\n\nimport log\n"),
    ("shadow.py", "import bracket\n", "import bracket\nimport log\n"),
    ("edge_book.py", "import shadow\n", "import log\nimport shadow\n"),
    ("candidates.py", "import bracket\n", "import bracket\nimport log\n"),
    ("features.py", "import engine\n", "import engine\nimport log\n"),
    ("paper.py", "from datetime import datetime, timezone\n",
     "from datetime import datetime, timezone\n\nimport log\n"),
]

total = 0
for name in TARGETS:
    path = os.path.join(BOT, name)
    src = io.open(path, encoding="utf-8").read()
    if name in ("calib.py", "engine.py") and "\nimport log\n" not in src:
        src = src.replace("import bracket\n", "import bracket\nimport log\n", 1)
    else:
        for target, anchor, repl in IMPORT_ANCHORS:
            if target == name and "\nimport log\n" not in src and anchor in src:
                src = src.replace(anchor, repl, 1)
                break

    def _rep(m):
        global total
        total += 1
        return f"{m.group('head')}{m.group('ind')}log.exc()"

    new = PAT.sub(_rep, src)
    if new != src:
        io.open(path, "w", encoding="utf-8", newline="\n").write(new)
        print(f"{name}: patched")

print(f"total silent handlers now logged: {total}")
