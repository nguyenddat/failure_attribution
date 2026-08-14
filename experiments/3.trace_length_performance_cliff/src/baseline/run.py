from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
SRC_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SRC_DIR))

from baseline.telbench.run import main as telbench_main  # noqa: E402
from baseline.trace_elephant.run import (  # noqa: E402
    main as trace_elephant_main,
)
from baseline.who_and_when.run import (  # noqa: E402
    main as who_and_when_main,
)


def main() -> None:
    who_and_when_main()
    trace_elephant_main()
    telbench_main()


if __name__ == "__main__":
    main()
