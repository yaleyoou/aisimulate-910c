"""Launch pinned serving with routing and optional operator-input observations."""

import importlib.metadata
import os
import sys

from ascend_state_capture import install

state = install()
if os.environ.get("AISIM_CAPTURE_INPUTS") == "1":
    from ascend_input_capture import install as install_inputs

    install_inputs(state)

if os.environ.get("AISIM_MEMORY_DIR"):
    from ascend_memory_capture import install as install_memory

    install_memory()


if __name__ == "__main__":
    entrypoint = next(
        entry for entry in importlib.metadata.distribution("sglang").entry_points if entry.name == "sglang"
    )
    sys.argv[1:] = ["serve", *sys.argv[1:]]
    entrypoint.load()()
