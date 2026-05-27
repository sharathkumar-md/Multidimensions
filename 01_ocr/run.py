from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def main() -> None:
    from src.pipeline import OCRPipeline

    pipeline = OCRPipeline()

    if len(sys.argv) > 1:
        pdf_paths = [Path(arg) for arg in sys.argv[1:]]
        missing = [p for p in pdf_paths if not p.exists()]
        if missing:
            for p in missing:
                print(f"ERROR: file not found: {p}")
            sys.exit(1)
        result = pipeline.run(pdf_paths=pdf_paths)
    else:
        result = pipeline.run()

    print(
        f"\nDone — {result.successful}/{result.total_documents} succeeded"
        f" | {result.failed} failed"
        f" | {result.total_processing_time_ms:.0f} ms total"
    )
    sys.exit(0 if result.failed == 0 else 1)


if __name__ == "__main__":
    main()
