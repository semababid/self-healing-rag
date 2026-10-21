"""Records docs/demo.gif: the web page answering a question that needs a retry.

Needs the web page running (streamlit run src/web.py) with Ollama, plus
Playwright and Pillow, which the project itself doesn't use:

    pip install playwright pillow
    playwright install chromium
    python scripts/record_demo.py [URL] [OUTPUT]

Defaults: http://localhost:8501 and docs/demo.gif.
"""

import io
import sys
import time

from PIL import Image
from playwright.sync_api import sync_playwright

URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8501"
OUT = sys.argv[2] if len(sys.argv) > 2 else "docs/demo.gif"
WIDTH = 960  # width of the GIF; the page is recorded at 1280 x 800

SCROLL = "document.querySelector('[data-testid=\"stMain\"]')"

frames: list[tuple[Image.Image, int]] = []  # (frame, milliseconds on screen)


def grab(page, ms: int) -> None:
    frames.append((Image.open(io.BytesIO(page.screenshot())).convert("RGB"), ms))


def record() -> None:
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1280, "height": 800},
                                color_scheme="light")
        page.goto(URL)
        page.get_by_text("Try an example:").wait_for(timeout=60_000)
        time.sleep(2)  # let fonts and icons load
        grab(page, 2200)

        page.get_by_role("button", name="Needs a retry").click()
        page.mouse.move(1270, 790)  # out of the way, so no hover effects show
        # A few frames of the spinner, not the whole wait.
        start = time.time()
        spinner = 0
        while not page.get_by_text("How it got there").is_visible():
            if spinner < 3 and time.time() - start > 0.6 * (spinner + 1):
                grab(page, 700)
                spinner += 1
            if time.time() - start > 180:
                sys.exit("No answer after 3 minutes. Is Ollama running?")
            time.sleep(0.2)
        time.sleep(1.5)  # badges and the chart finish drawing
        grab(page, 3500)

        # Scroll through the attempts, then hold at the bottom.
        height = page.evaluate(f"{SCROLL}.scrollHeight")
        position = 0
        while position < height - 800:
            position += 70
            page.evaluate(f"{SCROLL}.scrollTo(0, {position})")
            time.sleep(0.2)
            grab(page, 140)
        frames[-1] = (frames[-1][0], 3500)
        browser.close()


def save() -> None:
    resized = [(frame.resize((WIDTH, frame.height * WIDTH // frame.width), Image.LANCZOS), ms)
               for frame, ms in frames]
    # One palette built from all frames: a smaller file, and colours that only
    # appear in some frames (badges, the spinner) survive.
    mosaic = Image.new("RGB", (WIDTH, sum(frame.height for frame, _ in resized)))
    y = 0
    for frame, _ in resized:
        mosaic.paste(frame, (0, y))
        y += frame.height
    palette = mosaic.quantize(colors=256, method=Image.Quantize.MEDIANCUT)
    images = [frame.quantize(palette=palette, dither=Image.Dither.NONE) for frame, _ in resized]
    images[0].save(OUT, save_all=True, append_images=images[1:],
                   duration=[ms for _, ms in resized], loop=0, optimize=True)
    print(f"Saved {OUT}: {len(images)} frames, {sum(ms for _, ms in resized) / 1000:.1f} s")


if __name__ == "__main__":
    record()
    save()
