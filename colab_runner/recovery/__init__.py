import json
import re
import subprocess
import time
import uuid


def session_state(colab, session):
    try:
        output = colab(["sessions"], capture=True).stdout
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    if "[colab] No active sessions found on server." in output:
        return "gone"
    names = re.findall(r"^\[([^\]]+)\] \S+ \| Hardware:", output, re.MULTILINE)
    if session in names:
        return "active"
    if names and "?" not in names:
        return "gone"
    return "unknown"


def recover(colab, session, receipt, local_receipt, deadline, sleep=time.sleep):
    missing = 0
    last_state = "unknown"
    while time.time() < deadline:
        last_state = session_state(colab, session)
        if last_state == "gone":
            missing += 1
            if missing == 2:
                return "gone"
        else:
            missing = 0
            if last_state == "active":
                try:
                    colab(["download", "-s", session, "content/" + receipt, str(local_receipt)], capture=True)
                    if json.loads(local_receipt.read_text()).get("synced") is True:
                        return "finished"
                except (OSError, ValueError, subprocess.SubprocessError):
                    pass
        if last_state == "unknown":
            return "unknown"
        sleep(min(30, max(0, deadline - time.time())))
    return last_state if last_state != "gone" else "unknown"


def replace_session(colab, gpu, deadline):
    for delay in (0, 30, 60):
        if time.time() + delay >= deadline:
            break
        if delay:
            time.sleep(delay)
        name = "ai-detection-" + uuid.uuid4().hex[:10]
        try:
            colab(["new", "-s", name, "--gpu", gpu, "--high-mem"], capture=True)
            return name
        except subprocess.CalledProcessError as error:
            output = error.stdout or ""
            if "Allocation refused" not in output and "capacity limit" not in output:
                raise
    raise RuntimeError("Replacement GPU unavailable within the remaining time budget.")
