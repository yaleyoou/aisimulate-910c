#!/usr/bin/env bash
set -euo pipefail

repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
context_dir="$repo_dir/build/ascend910c-runtime"
image_name=${AISIMULATE_910C_IMAGE:-aisimulate-910c:glm52-w8a8}

if [[ ! -f "$context_dir/_runtime.abi3.so" || ! -f "$context_dir/native-identity.json" ]]; then
    printf 'Place the compiled native library and native-identity.json in %s; see md/910c-glm52-w8a8-usage.md\n' "$context_dir" >&2
    exit 1
fi

python3 - "$context_dir" <<'PY'
import hashlib
import json
import sys
import zipfile
from pathlib import Path

directory = Path(sys.argv[1])
identity = json.loads((directory / 'native-identity.json').read_text())
actual = hashlib.sha256((directory / '_runtime.abi3.so').read_bytes()).hexdigest()
if actual != identity['sha256']:
    raise SystemExit('Native library SHA256 does not match native-identity.json')
wheel = directory / identity['wheel']
if hashlib.sha256(wheel.read_bytes()).hexdigest() != identity['wheel_sha256']:
    raise SystemExit('Wheel SHA256 does not match native-identity.json')
with zipfile.ZipFile(wheel) as archive:
    packaged = hashlib.sha256(archive.read('aisimulate/_runtime.abi3.so')).hexdigest()
if packaged != actual:
    raise SystemExit('Packaged native library differs from the identified native library')
dependency_manifest = directory / 'dependency-downloads.json'
if dependency_manifest.exists():
    for dependency in json.loads(dependency_manifest.read_text()):
        artifact = directory / dependency['file']
        if hashlib.sha256(artifact.read_bytes()).hexdigest() != dependency['sha256']:
            raise SystemExit('Dependency wheel SHA256 mismatch: ' + dependency['file'])
print(json.dumps(identity, indent=2))
PY

exec docker build --network host --file "$repo_dir/python/aisimulate/docker/Dockerfile.ascend910c" \
    --tag "$image_name" "$context_dir"
