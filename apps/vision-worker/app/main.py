"""Vision worker entry: RTSP + plate OCR -> API live ingest."""

from __future__ import annotations

from apps.vision_worker.app.pipeline import main_loop


def main() -> None:
    main_loop()


if __name__ == "__main__":
    main()
