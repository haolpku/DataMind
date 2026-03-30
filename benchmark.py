"""Compatibility shim — redirects to benchmark.run.main()."""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from benchmark.run import main

if __name__ == "__main__":
    print("[WARN] benchmark.py 已迁移到 benchmark/ 包，建议使用:")
    print("       python -m benchmark.run --questions ...\n")
    main()
