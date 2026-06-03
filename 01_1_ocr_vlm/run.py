from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def main() -> None:
    from src.pipeline import run

    pdf_paths = [Path(arg) for arg in sys.argv[1:]] if len(sys.argv) > 1 else None
    run(pdf_paths=pdf_paths)


if __name__ == "__main__":
    main()
