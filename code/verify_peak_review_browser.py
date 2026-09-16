"""Exercise the standalone review in a local headless Chrome via DevTools."""
import base64
import json
from pathlib import Path
import subprocess
import tempfile
import time

import requests
import websocket

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments/2026-09-16_peak_height_selection"


def main():
    chrome = Path("C:/Program Files/Google/Chrome/Application/chrome.exe")
    profile = tempfile.mkdtemp(prefix="boiling-peak-browser-")
    browser_log = (OUT / "browser_launch.log").open("w", encoding="utf-8")
    proc = subprocess.Popen([str(chrome), "--headless=new", "--disable-gpu", "--no-first-run",
        "--remote-debugging-port=19346", "--remote-allow-origins=http://localhost:19346",
        "--user-data-dir=" + profile, "about:blank"], stdout=browser_log, stderr=browser_log,
        creationflags=subprocess.CREATE_NO_WINDOW)
    ws = None
    try:
        for attempt in range(80):
            if proc.poll() is not None:
                raise RuntimeError(f"Browser exited with {proc.returncode}; see browser_launch.log")
            try:
                pages = requests.get("http://localhost:19346/json", timeout=1).json()
                break
            except requests.RequestException:
                time.sleep(.2)
        else:
            raise RuntimeError("Local browser did not start")
        page = next(p for p in pages if p.get("type") == "page")
        ws = websocket.create_connection(page["webSocketDebuggerUrl"], timeout=10,
                                         origin="http://localhost:19346")
        serial = 0
        events = []
        def call(method, params=None):
            nonlocal serial
            serial += 1
            ws.send(json.dumps({"id": serial, "method": method, "params": params or {}}))
            while True:
                result = json.loads(ws.recv())
                if "method" in result:
                    events.append(result)
                if result.get("id") == serial:
                    if "error" in result:
                        raise RuntimeError(result)
                    return result.get("result", {})
        def evaluate(expression):
            result = call("Runtime.evaluate", {"expression": expression, "returnByValue": True})
            if "exceptionDetails" in result:
                raise RuntimeError(result)
            return result["result"].get("value")
        call("Page.enable")
        call("Runtime.enable")
        call("Emulation.setDeviceMetricsOverride", {"width": 1280, "height": 1260, "deviceScaleFactor": 1, "mobile": False})
        navigation = call("Page.navigate", {"url": (OUT / "peak_threshold_review.html").as_uri()})
        for attempt in range(80):
            if evaluate("document.readyState === 'complete' && !!document.getElementById('status') && !!document.getElementById('status').textContent"):
                break
            time.sleep(.1)
        else:
            diagnostics = {"navigation": navigation, "state": evaluate("JSON.stringify({url:location.href,title:document.title,body:document.body.innerText.slice(0,2000)})"),
                           "events": [e for e in events if e.get("method") == "Runtime.exceptionThrown"]}
            (OUT / "browser_failure.json").write_text(json.dumps(diagnostics, ensure_ascii=False, indent=2), encoding="utf-8")
            raise RuntimeError("Review did not initialize; see browser_failure.json")
        assert evaluate("data.length") == 49
        evaluate("el('day').value='2025.07.09_0.3_1'; chooseDay(); el('wav').value=data.findIndex(w=>w.day==='2025.07.09_0.3_1'&&w.wav==='index=14.643516'); chooseWav();")
        checks = []
        for value, expected in [(1, 13), (10, 5), (.3, 16), (3, 9)]:
            text = evaluate(f"document.querySelector('button[data-v=\"{value}\"]').click(); el('status').textContent")
            assert f"{expected}/60" in text, text
            checks.append({"threshold_plot_units": value, "expected_count": expected, "status": text})
        evaluate("document.querySelector('button[data-v=\"1\"]').click(); el('second').value=4; draw();")
        assert 3.45 < evaluate("current.peaks[4]") < 3.47
        screenshot = call("Page.captureScreenshot", {"format": "png", "captureBeyondViewport": True})
        (OUT / "review_screenshot.png").write_bytes(base64.b64decode(screenshot["data"]))
        evaluate("el('second').value=9; draw();")
        assert "横線未満" in evaluate("el('value').textContent")
        evaluate("el('day').value='2025.06.18_0.3_3'; chooseDay();")
        assert "全60秒を評価" in evaluate("el('status').textContent")
        (OUT / "browser_verification.json").write_text(json.dumps({"browser": "headless Chrome", "checks": checks,
            "all_49_wavs_loaded": True, "below_line_status_correct": True, "test_day_unfiltered_notice": True},
            ensure_ascii=False, indent=2), encoding="utf-8")
        print("Browser passed: 49 WAVs, threshold counts, spectrum selection, and test-day status")
    finally:
        if ws:
            ws.close()
        proc.terminate()
        proc.wait(timeout=15)
        browser_log.close()


if __name__ == "__main__":
    main()
