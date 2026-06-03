import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.pipeline import run

if __name__ == "__main__":
    paths = [Path(a) for a in sys.argv[1:]] if len(sys.argv) > 1 else None
    run(paths)
