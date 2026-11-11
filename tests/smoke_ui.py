"""UI smoke tests (headless Chromium, phone-sized viewport) against the stub server.

Usage: python tests/smoke_ui.py [port]
Checks:
  1. Tapping a start button brings the board into view on a phone (regression test).
  2. Easy-mode auto-hint uses a light search (<= 50 sims).
  3. Requests carry a client id + increasing epochs and the page sends /api/cancel.
  4. A late AI reply that belongs to a previous game is ignored after a restart.
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
            page.close()

            # 3. request ids + cancel calls
            page = browser.new_page(viewport={"width": 393, "height": 852}, has_touch=True, is_mobile=True)
            page.goto(BASE + "/", wait_until="domcontentloaded")
            page.wait_for_timeout(800)
            page.click("#btnPlayerFirst")
            page.click('.column[data-col="3"]')
            page.wait_for_function("document.querySelectorAll('.chip-2').length >= 1", timeout=15000)
            page.click('.column[data-col="4"]')
            page.wait_for_function("document.querySelectorAll('.chip-2').length >= 2", timeout=15000)
            page.wait_for_timeout(500)
            log = json.load(urllib.request.urlopen(BASE + "/api/_requests"))  # nosec B310
            cid = log["requests"][-1].get("client_id")  # this page's id (earlier pages used their own)
            mine = [r for r in log["requests"] if r.get("client_id") == cid]
            moves = [r for r in mine if r["_path"] == "move" and r.get("kind") == "move"]
            epochs = [r.get("epoch") for r in moves]
            ids = {r.get("client_id") for r in mine}
            check("move requests carry client_id + increasing epoch",
                  len(moves) >= 2 and epochs == sorted(epochs) and len(set(epochs)) == len(epochs) and len(ids) == 1 and None not in ids,
                  f"epochs={epochs} ids={len(ids)}")
            kinds = {k for c in log["cancels"] if c.get("client_id") == cid for k in c.get("upto", {})}
            check("page sends /api/cancel for hint+assess on a move", {"hint", "assess"} <= kinds, f"kinds={sorted(kinds)}")
            page.close()

            # 4. late AI reply after restart is ignored
            urllib.request.urlopen(BASE + "/api/_delay/1500")  # nosec B310
            page = browser.new_page(viewport={"width": 393, "height": 852}, has_touch=True, is_mobile=True)
            page.goto(BASE + "/", wait_until="domcontentloaded")
            page.wait_for_timeout(800)
            page.click("#btnAiFirst")            # AI request now in flight (1.5s)
            page.wait_for_timeout(200)
            page.click("#btnPlayerFirst")        # restart before the reply arrives
            page.wait_for_timeout(2500)          # the old reply lands during this window
            pieces = page.evaluate("document.querySelectorAll('.chip-1, .chip-2').length")
            check("late AI reply from a previous game is ignored", pieces == 0, f"pieces={pieces}")
            browser.close()
    finally:
        srv.terminate()
    sys.exit(1 if failures else 0)


main()
