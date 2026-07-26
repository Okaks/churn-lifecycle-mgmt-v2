"""
Keeps a Streamlit Community Cloud app awake by actually loading it in a
real headless browser, not just pinging the URL with a plain HTTP request.

Why a plain request doesn't work: Streamlit serves a static HTML shell to
any GET request, even a sleeping app returns HTTP 200. The app only
actually "wakes" when a real browser executes its JavaScript and opens a
WebSocket connection - which is what this script does with Playwright.

If the app is asleep, it shows a "Yes, get this app back up!" button;
this script detects and clicks it, then waits for the app to finish
loading before exiting.
"""

import os
import sys
import time

from playwright.sync_api import sync_playwright

APP_URL = os.environ.get("STREAMLIT_APP_URL", "https://churn-lifecycle-mgmt-v2.streamlit.app/")
WAKE_BUTTON_TEXT = "Yes, get this app back up!"
TIMEOUT_MS = 60_000


def main():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        print(f"Visiting {APP_URL} ...")
        page.goto(APP_URL, timeout=TIMEOUT_MS, wait_until="domcontentloaded")

        # Give the page a moment to render either the live app or the sleep screen
        page.wait_for_timeout(3000)

        wake_button = page.get_by_text(WAKE_BUTTON_TEXT, exact=False)
        if wake_button.count() > 0:
            print("App was asleep - clicking wake button...")
            wake_button.first.click()
            # Wait for the app to actually finish restarting and render real content
            page.wait_for_timeout(15000)
            print("Wake triggered. App should be restarting.")
        else:
            print("App was already awake - visit alone resets its idle timer.")

        browser.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Keep-alive run failed: {e}", file=sys.stderr)
        sys.exit(1)
