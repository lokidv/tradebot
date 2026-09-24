# -*- coding: utf-8 -*-
"""صفحه‌ها هرگز از کشِ حدسیِ مرورگر نشان داده نمی‌شوند — «بعضی وقت‌ها ظاهرِ قدیمی» از همین بود."""
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "bot"))
import _hermetic  # noqa: E402,F401  — پیش از هر ماژولِ ربات: داده به پوشهٔ موقت

from fastapi.testclient import TestClient  # noqa: E402

import main  # noqa: E402


class CacheHeaderTests(unittest.TestCase):
    def setUp(self):
        self.c = TestClient(main.app)

    def test_pages_must_revalidate(self):
        for path in ("/", "/lab"):
            r = self.c.get(path)
            self.assertEqual(r.status_code, 200)
            self.assertIn("no-cache", r.headers.get("cache-control", ""))

    def test_main_page_is_the_new_one(self):
        self.assertIn("میز تصمیم", self.c.get("/").text)

    def test_api_is_never_stored(self):
        r = self.c.get("/api/gates")
        self.assertEqual(r.headers.get("cache-control"), "no-store")


if __name__ == "__main__":
    unittest.main()
