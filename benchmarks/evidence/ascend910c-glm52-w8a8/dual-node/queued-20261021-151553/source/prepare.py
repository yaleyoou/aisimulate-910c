import concurrent.futures
import json
import pathlib
import subprocess

from remote import run_python


IMAGE = "quay.io/ascend/sglang@sha256:5f384cab0e16a4951ed0bdc54e1280f2108aeda6b8ede90ee527d8f4b5146443"
CONTAINER = "aisim-910c-hardware-20261021"
BUILD = """
set -eu
export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y --no-install-recommends openmpi-bin libopenmpi-dev openssh-server
cd /usr/local/Ascend/cann/tools/hccl_test
make MPI_HOME=/usr/lib/aarch64-linux-gnu/openmpi ASCEND_DIR=/usr/local/Ascend/cann CXXFLAGS='-std=c++11 -DOMPI_SKIP_MPICXX=1 -Werror -fstack-protector-strong -fPIE -pie -O2 -s -Wl,-z,relro -Wl,-z,now -Wl,-z,noexecstack -Wl,--copy-dt-needed-entries'
dpkg-query -W openmpi-bin libopenmpi-dev openssh-server
sha256sum bin/*_test
"""


def preparation_source():
    return """import json,pathlib,subprocess
container = %r
image = %r
existing=subprocess.run(['docker','inspect',container],capture_output=True,text=True)
if existing.returncode==0:
    raise RuntimeError('Task container already exists; inspect before reusing')
arguments=['docker','run','-d','--name',container,'--network','host','--privileged','--shm-size','16g','--mount','type=bind,src=/usr/local/Ascend/driver,dst=/usr/local/Ascend/driver,readonly','-e','HCCL_SOCKET_IFNAME=bond4','-e','HCCL_IF_BASE_PORT=27000','-e','HCCL_BUFFSIZE=650','--entrypoint','/bin/sleep']
for item in ['/etc/ascend_install.info','/usr/local/Ascend/firmware','/usr/local/sbin']:
    if pathlib.Path(item).exists():
        arguments += ['--mount','type=bind,src='+item+',dst='+item+',readonly']
for item in ['/dev/davinci'+str(index) for index in range(16)]+['/dev/davinci_manager','/dev/hisi_hdc','/dev/devmm_svm']:
    if pathlib.Path(item).exists():
        arguments += ['--device',item]
arguments += [image,'infinity']
subprocess.run(arguments,check=True)
result=subprocess.run(['docker','exec',container,'bash','-lc',%r],text=True)
raise SystemExit(result.returncode)
""" % (CONTAINER, IMAGE, BUILD)


def main():
    root = pathlib.Path(__file__).resolve().parent
    source = preparation_source()

    def local():
        return subprocess.run(["python3", "-c", source], capture_output=True, text=True, timeout=1200)

    def remote():
        password = input()
        return run_python(source, password, timeout=1200)

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        futures = {executor.submit(local): "npu1-32", executor.submit(remote): "npu1-19"}
        results = {}
        for future in concurrent.futures.as_completed(futures):
            host = futures[future]
            try:
                result = future.result()
                (root / (host + "-build.log")).write_text(result.stdout + result.stderr)
                results[host] = result.returncode
                print(host, "build exit", result.returncode, flush=True)
            except Exception as error:
                results[host] = repr(error)
                print(host, repr(error), flush=True)
        (root / "build-results.json").write_text(json.dumps(results, indent=2) + "\n")
        if any(value != 0 for value in results.values()):
            raise SystemExit(1)


if __name__ == "__main__":
    main()
