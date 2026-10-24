"""Intentional negative control; never publish its output as performance data."""

import json
import os
import runpy
import sys

with open('/campaign/serving-setup-nodes.json') as stream:
    setup = json.load(stream)
for name, value in setup['environment'].items():
    if value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = value

import torch_npu

torch_npu.npu.NPUGraph.replay = lambda _graph: None
target = sys.argv.pop(1)
runpy.run_path(target, run_name='__main__')
