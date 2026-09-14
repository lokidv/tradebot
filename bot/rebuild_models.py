# -*- coding: utf-8 -*-
"""بازسازی مستقل مدل‌ها از خط فرمان، بدون نیاز به روشن‌بودن وب‌سرور."""
import argparse
import json
import sys

import calib
import market


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--top", type=int, default=100, help="تعداد نمادهای پرحجم برای آموزش")
    parser.add_argument("--tfs", nargs="+", default=["1d", "4h", "1h", "15m"])
    args = parser.parse_args()

    symbols, _ = market.get_top_symbols(max(10, args.top))
    calib.build(symbols[:args.top], tfs=tuple(args.tfs))
    status = calib.status()
    print(json.dumps(status, ensure_ascii=False))
    if status.get("error"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
