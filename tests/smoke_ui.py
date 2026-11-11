"""UI smoke tests (headless Chromium, phone-sized viewport) against the stub server.

Usage: python tests/smoke_ui.py [port]
Checks:
  1. Tapping a start button brings the board into view on a phone (regression test).
  2. Easy-mode auto-hint uses a light search (<= 50 sims).
"""
import json
import subprocess
import sys
import time
import urllib.request
from playwright.sync_api import sync_playwright

PORT = sys.argv[1] if len(sys.argv) > 1 else "5099"
BASE = f"http://127.0.0.1:{PORT}"
failures = []


def check(name, ok, detail=""):
    print(("PASS" if ok else "FAIL"), name, detail)
    if not ok:
        failures.append(name)


def visible_fraction(page, selector):
    return page.evaluate("""(sel) => {
        const r = document.querySelector(sel).getBoundingClientRect();
        const vis = Math.max(0, Math.min(r.bottom, innerHeight) - Math.max(r.top, 0));
        return r.height ? vis / r.height : 0;
    }""", selector)


def main():
    srv = subprocess.Popen([sys.executable, "tests/stub_server.py", PORT])
    try:
        for _ in range(50):
            try:
                urllib.request.urlopen(BASE + "/api/info", timeout=1)  # nosec B310
                break
            except Exception:
                time.sleep(0.2)
        with sync_playwright() as p:
            browser = p.chromium.launch(args=["--no-sandbox"])
            for h in (852, 700):
                page = browser.new_page(viewport={"width": 393, "height": h}, has_touch=True, is_mobile=True)
                page.goto(BASE + "/", wait_until="domcontentloaded")
                page.wait_for_timeout(800)
                page.click("#btnPlayerFirst")
                page.wait_for_timeout(1200)  # smooth scroll
                frac = visible_fraction(page, "#boardArea")
                check(f"board in view after Start ({h}px tall)", frac >= 0.9, f"visible={frac:.0%}")
                page.close()

            page = browser.new_page(viewport={"width": 393, "height": 852}, has_touch=True, is_mobile=True)
            page.goto(BASE + "/", wait_until="domcontentloaded")
            page.wait_for_timeout(800)
            page.select_option("#difficultySelect", "easy")
            page.evaluate("document.getElementById('simsSlider').value = 1600")
            page.click("#btnPlayerFirst")
            page.wait_for_timeout(2000)  # auto-hint fires 500ms after turn starts
            sims = json.load(urllib.request.urlopen(BASE + "/api/move_sims"))  # nosec B310
            check("easy auto-hint uses <= 50 sims", bool(sims) and max(sims) <= 50, f"sims={sims}")
            browser.close()
    finally:
        srv.terminate()
    sys.exit(1 if failures else 0)


main()
