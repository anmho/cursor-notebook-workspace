#!/usr/bin/env python3
"""Print a concise local ML workstation report."""

from __future__ import annotations

import json
import sys

from ml_lab.device import device_report, run_smoke_test


def main() -> int:
    print(json.dumps({"python": sys.version, "device": device_report().as_dict()}, indent=2))
    print(json.dumps({"smoke_test": run_smoke_test()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
