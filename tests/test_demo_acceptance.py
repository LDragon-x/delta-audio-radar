"""End-to-end acceptance: the offline demo must pass all assertions."""
import subprocess
import sys
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_demo_pass():
    r = subprocess.run(
        [sys.executable, "main.py", "--demo"],
        cwd=ROOT, capture_output=True, text=True, encoding="utf-8",
        errors="replace", timeout=300,
    )
    assert r.returncode == 0, r.stderr[-2000:]
    assert "DEMO PASS" in r.stdout, r.stdout[-2000:]
