import concurrent.futures
import json
import pathlib
import subprocess

from prepare import BUILD, CONTAINER
from remote import run_python


def main():
    root = pathlib.Path(__file__).resolve().parent
    password = input()
    native = (root / "p2p_probe.cpp").read_text()
    build = BUILD[BUILD.index("cd /usr/local/Ascend"):]
    build += "\nmpic++ -std=c++17 -DOMPI_SKIP_MPICXX=1 -O2 -Wall -Wextra /tmp/p2p_probe.cpp -I/usr/local/Ascend/cann/include -L/usr/local/Ascend/cann/lib64 -lhccl -lacl_rt -o /tmp/p2p_probe\ncp /tmp/p2p_probe /tmp/p2p_queued\nsha256sum /tmp/p2p_probe /tmp/p2p_queued\n"
    source = """import subprocess
container=%r
subprocess.run(['docker','exec','-i',container,'tee','/tmp/p2p_probe.cpp'],input=%r,text=True,capture_output=True,check=True)
result=subprocess.run(['docker','exec',container,'bash','-lc',%r],text=True)
raise SystemExit(result.returncode)
""" % (CONTAINER, native, "set -eu\n" + build)

    def local():
        return subprocess.run(["python3", "-c", source], capture_output=True, text=True, timeout=900)

    def remote():
        return run_python(source, password, timeout=900)

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        futures = {executor.submit(local): "npu1-32", executor.submit(remote): "npu1-19"}
        results = {}
        for future in concurrent.futures.as_completed(futures):
            host = futures[future]
            result = future.result()
            (root / (host + "-native-build.log")).write_text(result.stdout + result.stderr)
            results[host] = result.returncode
            print(host, result.returncode, flush=True)
        (root / "native-build-results.json").write_text(json.dumps(results, indent=2) + "\n")
        if any(value != 0 for value in results.values()):
            raise SystemExit(1)


if __name__ == "__main__":
    main()
