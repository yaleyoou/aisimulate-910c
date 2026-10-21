import json
import pathlib
import subprocess

from remote import run_python


CONTAINER = "aisim-910c-hardware-20261021"


def main():
    password = input()
    root = pathlib.Path(__file__).resolve().parent
    subprocess.run([
        "docker", "exec", CONTAINER, "bash", "-c",
        "mkdir -p /root/.ssh && chmod 700 /root/.ssh && "
        "test ! -e /root/.ssh/aisim_key && "
        "ssh-keygen -q -t ed25519 -N '' -f /root/.ssh/aisim_key",
    ], check=True)
    public_key = subprocess.check_output([
        "docker", "exec", CONTAINER, "cat", "/root/.ssh/aisim_key.pub",
    ], text=True)
    source = f'''import subprocess, json
container = {CONTAINER!r}
ports = subprocess.check_output(["ss", "-ltn"], text=True)
if ":22290 " in ports:
    raise RuntimeError("Task SSH port is already occupied")
subprocess.run(["docker", "exec", container, "mkdir", "-p", "/root/.ssh", "/run/sshd"], check=True)
subprocess.run(["docker", "exec", "-i", container, "tee", "/root/.ssh/authorized_keys"], input={public_key!r}, text=True, capture_output=True, check=True)
subprocess.run(["docker", "exec", container, "chmod", "600", "/root/.ssh/authorized_keys"], check=True)
subprocess.run(["docker", "exec", "-d", container, "/usr/sbin/sshd", "-D", "-p", "22290", "-oListenAddress=10.120.72.19", "-oPermitRootLogin=prohibit-password", "-oPasswordAuthentication=no", "-oUsePAM=no"], check=True)
print(json.dumps({{"task_ssh_port": 22290, "host_ssh_unchanged": True}}))
'''
    result = run_python(source, password)
    result.check_returncode()
    checked = subprocess.run([
        "docker", "exec", CONTAINER, "ssh", "-F", "/dev/null", "-i", "/root/.ssh/aisim_key",
        "-p", "22290", "-oBatchMode=yes", "-oStrictHostKeyChecking=accept-new",
        "root@10.120.72.19", "hostname; command -v orted",
    ], capture_output=True, text=True, check=True)
    (root / "mpi-setup.json").write_text(json.dumps({
        "remote": json.loads(result.stdout), "check_stdout": checked.stdout,
        "check_stderr": checked.stderr, "key_scope": "temporary collection containers only",
    }, indent=2))
    print("MPI task-container SSH ready")


if __name__ == "__main__":
    main()
