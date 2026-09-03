#!/home/user/playwright_venv/bin/python
"""Load a page in headless Chromium, screenshot it, and report what broke.

Verification harness for the Road 2 on-device demo (`webapp_ondevice/site/`).
Implementation agents cannot open a browser window, so this script is how they
see the app: it drives the page, captures a screenshot, and — crucially —
prints every console message, uncaught exception, and failed network request,
then turns that into an exit code.

Runs under the shared Playwright venv (built for the Road 1 webapp checks):

    ~/playwright_venv/bin/python webapp_ondevice/devcheck/check_page.py ...

The shebang points at that interpreter, so `./check_page.py ...` works too.

Exit codes
    0  clean: no error-level console messages, no page errors, no failed requests
    1  the page reported problems (see the FAIL line for the counts)
    2  the harness itself could not run the check (bad target, navigation
       failure, bad selector, JS evaluation error)

"Failed request" means either a network-level failure (DNS, connection reset,
aborted fetch) or an HTTP response with status >= 400 — a 404 on a model or
.wasm file is a response, not a network failure, and would otherwise be
invisible here. `--lenient` keeps printing those but stops them from failing
the run, for checks made before the model files are in place; it also excuses
the browser's own "Failed to load resource" console error for them, which is
the same event seen from the other side. Errors thrown by the app's own code
still fail a lenient run.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

def die(message: str) -> None:
    """Exit 2 — the harness could not run the check at all."""
    print(f"[harness] {message}", file=sys.stderr)
    sys.exit(2)


try:
    from playwright.sync_api import Error as PlaywrightError
    from playwright.sync_api import sync_playwright
except ImportError:  # pragma: no cover - environment problem, not logic
    die("playwright not importable. Run this with the shared Playwright venv:\n"
        "  ~/playwright_venv/bin/python webapp_ondevice/devcheck/check_page.py ...")


def resolve_target(target: str, hash_fragment: str | None) -> str:
    """Turn a URL or a filesystem path into a URL, with the fragment applied."""
    if target.startswith(("http://", "https://", "file://")):
        url = target
    else:
        path = Path(target).expanduser().resolve()
        if not path.exists():
            die(f"no such file: {path}  (and it is not a URL)")
        url = path.as_uri()

    if hash_fragment:
        url = url.split("#", 1)[0] + "#" + hash_fragment.lstrip("#")
    return url


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Screenshot a page and report console errors, page errors, failed requests.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Step order after load: --wait-selector, then --click (in order), "
            "then --wait-ms, then --js, then the screenshot."
        ),
    )
    ap.add_argument("target", help="http(s)/file URL, or a local path (converted to file://)")
    ap.add_argument("--hash", dest="hash_fragment", metavar="FRAG",
                    help='fragment to append, e.g. --hash "state=S6&clip=A"')
    ap.add_argument("--width", type=int, default=390, help="viewport width (default: 390)")
    ap.add_argument("--height", type=int, default=844, help="viewport height (default: 844)")
    ap.add_argument("--dark", action="store_true", help="emulate prefers-color-scheme: dark")
    ap.add_argument("--shot", metavar="OUT.PNG", help="write a screenshot here (skipped if omitted)")
    ap.add_argument("--full-page", action="store_true", help="capture the whole scrollable page")
    ap.add_argument("--wait-ms", type=int, metavar="N", help="wait N ms before capture")
    ap.add_argument("--wait-selector", metavar="CSS", help="wait for this selector before capture")
    ap.add_argument("--click", metavar="CSS", action="append", default=[],
                    help="click this selector before capture (repeatable, applied in order)")
    ap.add_argument("--js", metavar="EXPR",
                    help="evaluate a JS expression after load and print the JSON result")
    ap.add_argument("--lenient", action="store_true",
                    help="report failed requests but do not fail the run on them")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    url = resolve_target(args.target, args.hash_fragment)

    console: list[tuple[str, str]] = []   # (level, text)
    page_errors: list[str] = []
    failed: list[str] = []

    print("=== target ===")
    print(url)
    print(f"viewport {args.width}x{args.height}  scheme={'dark' if args.dark else 'light'}")

    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        context = browser.new_context(
            viewport={"width": args.width, "height": args.height},
            color_scheme="dark" if args.dark else "light",
        )
        page = context.new_page()

        # Listeners must be attached before navigation or early messages are lost.
        def on_console(msg):
            where = ""
            loc = msg.location or {}
            if loc.get("url"):
                where = f"  @ {loc['url']}:{loc.get('lineNumber', 0)}:{loc.get('columnNumber', 0)}"
            console.append((msg.type, msg.text + where))

        def on_request_failed(request):
            failed.append(f"{request.method} {request.url}  [{request.failure or 'failed'}]")

        def on_response(response):
            if response.status >= 400:
                failed.append(f"{response.request.method} {response.url}  [HTTP {response.status}]")

        page.on("console", on_console)
        page.on("pageerror", lambda exc: page_errors.append(str(exc)))
        page.on("requestfailed", on_request_failed)
        page.on("response", on_response)

        harness_error = None
        try:
            response = page.goto(url, wait_until="load")
            if response is not None and response.status >= 400:
                # goto() itself succeeded but the document is an error page.
                print(f"[harness] main document returned HTTP {response.status}")

            if args.wait_selector:
                page.wait_for_selector(args.wait_selector)
            for selector in args.click:
                page.click(selector)
            if args.wait_ms:
                page.wait_for_timeout(args.wait_ms)
        except PlaywrightError as exc:
            harness_error = str(exc).strip().splitlines()[0]

        js_result = None
        js_error = None
        if args.js and harness_error is None:
            try:
                js_result = page.evaluate(args.js)
            except PlaywrightError as exc:
                js_error = str(exc).strip().splitlines()[0]

        shot_note = None
        if args.shot and harness_error is None:
            out = Path(args.shot).expanduser().resolve()
            out.parent.mkdir(parents=True, exist_ok=True)
            page.screenshot(path=str(out), full_page=args.full_page)
            shot_note = f"{out}  ({'full-page' if args.full_page else 'viewport'})"

        browser.close()

    errors = [m for m in console if m[0] == "error"]
    # Chromium logs a console error for every failed resource load, so a 404 is
    # reported twice: once here, once as a failed request. --lenient must excuse
    # both halves or it would not excuse anything.
    resource_errors = [m for m in errors if m[1].startswith("Failed to load resource")]
    script_errors = [m for m in errors if not m[1].startswith("Failed to load resource")]

    print(f"\n=== console ({len(console)}) ===")
    for level, text in console:
        print(f"[{level}] {text}")

    print(f"\n=== page errors ({len(page_errors)}) ===")
    for text in page_errors:
        print(text)

    label = "failed requests (lenient)" if args.lenient else "failed requests"
    print(f"\n=== {label} ({len(failed)}) ===")
    for text in failed:
        print(text)

    if args.js:
        print("\n=== js ===")
        if js_error:
            print(f"[harness] evaluation failed: {js_error}")
        else:
            print(json.dumps(js_result, ensure_ascii=False, indent=2, default=str))

    if shot_note:
        print(f"\n=== shot ===\n{shot_note}")

    print("\n=== result ===")
    if harness_error:
        print(f"HARNESS ERROR: {harness_error}")
        return 2
    if js_error:
        print(f"HARNESS ERROR: --js evaluation failed: {js_error}")
        return 2

    counted_console = len(script_errors) if args.lenient else len(errors)
    counted_failures = 0 if args.lenient else len(failed)
    summary = (f"{len(errors)} console error(s), {len(page_errors)} page error(s), "
               f"{len(failed)} failed request(s)")
    if args.lenient and (failed or resource_errors):
        summary += (f" — {len(failed)} request(s) and {len(resource_errors)} "
                    f"resource-load console error(s) excused (--lenient)")
    if counted_console + len(page_errors) + counted_failures:
        print(f"FAIL: {summary}")
        return 1
    print(f"OK: {summary}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
