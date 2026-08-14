from __future__ import annotations

from experiments.single_fault.experiments.baseline.telbench.run import main as telbench_main
from experiments.single_fault.experiments.baseline.trace_elephant.run import (
    main as trace_elephant_main,
)
from experiments.single_fault.experiments.baseline.who_and_when.run import (
    main as who_and_when_main,
)


def main() -> None:
    who_and_when_main()
    trace_elephant_main()
    telbench_main()


if __name__ == "__main__":
    main()
