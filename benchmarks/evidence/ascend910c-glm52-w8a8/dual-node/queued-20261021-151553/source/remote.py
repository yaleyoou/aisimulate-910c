import os
import shlex
import subprocess


def run_python(source, password, timeout=120):
    reader, writer = os.pipe()
    os.write(writer, (password + "\n").encode())
    os.close(writer)
    command = [
        "sshpass", "-d", str(reader), "ssh", "-F", "/dev/null",
        "-o", "ConnectTimeout=10", "-o", "StrictHostKeyChecking=accept-new",
        "youxingye@npu1-19", "sudo -S -p '' python3 -c " + shlex.quote(source),
    ]
    try:
        return subprocess.run(
            command, input=password + "\n", capture_output=True,
            text=True, pass_fds=(reader,), timeout=timeout,
        )
    finally:
        os.close(reader)
