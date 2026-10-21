import json
import pathlib

from remote import run_python


def main():
    password = input()
    source = '''import subprocess, json
container="aisim-910c-hardware-20261021"
command=["docker","exec",container,"timeout","180","mpirun","--allow-run-as-root","--bind-to","none","-np","2","/tmp/p2p_probe","0,1","/tmp/remote-readiness","20","7","1048576"]
result=subprocess.run(command,capture_output=True,text=True,timeout=200)
print(json.dumps({"command":command,"returncode":result.returncode,"stdout":result.stdout,"stderr":result.stderr}))
'''
    result = run_python(source, password, timeout=220)
    root = pathlib.Path(__file__).resolve().parent
    (root / "remote-readiness.json").write_text(result.stdout + result.stderr)
    result.check_returncode()
    data = json.loads(result.stdout)
    print(json.dumps(data))
    if data["returncode"]:
        raise RuntimeError("Remote probe failed")


if __name__ == "__main__":
    main()
