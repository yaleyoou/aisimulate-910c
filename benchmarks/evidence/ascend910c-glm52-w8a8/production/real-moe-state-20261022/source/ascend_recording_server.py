"""Launch the pinned serving implementation with count-only instrumentation."""

import importlib.metadata
import sys

from ascend_state_capture import install

install()


if __name__ == "__main__":
    entrypoint = next(entry for entry in importlib.metadata.distribution("sglang").entry_points if entry.name == "sglang")
    sys.argv[1:] = ["serve", *sys.argv[1:]]
    entrypoint.load()()
