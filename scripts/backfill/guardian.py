"""External SSH-session watchdog: resume the batch shell if its step dies.

This is intentionally launched independently of the srun controller step.
It never touches the original training children or any other allocation.
"""
import json
import os
from pathlib import Path
import signal
import sys
import time

queue, job = Path(sys.argv[1]), sys.argv[2]
allocation = json.loads((queue / "allocations" / (job + ".json")).read_text())
pid = allocation["batch_pid"]
stat = Path(f"/proc/{pid}/stat")
cmd = Path(f"/proc/{pid}/cmdline").read_bytes()
assert f"/job{job}/slurm_script".encode() in cmd
start = stat.read_text().split(") ", 1)[1].split()[19]
began = time.time()
status = queue / "nodes" / (job + ".json")
guardian_status = queue / "nodes" / (job + ".guardian.json")
guardian_status.write_text(json.dumps(dict(state="watching", pid=os.getpid(),
                                         job=job, started_at=began)))

def resume():
    try:
        fields = stat.read_text().split(") ", 1)[1].split()
        if fields[19] == start:
            os.kill(pid, signal.SIGCONT)
            print("resumed batch shell", job, pid, flush=True)
    except (FileNotFoundError, ProcessLookupError):
        pass

while True:
    try:
        if stat.read_text().split(") ", 1)[1].split()[19] != start:
            break
    except FileNotFoundError:
        break
    try:
        state = json.loads(status.read_text())
    except FileNotFoundError:
        state = {}
    if state.get("state") == "released":
        resume()
        break
    controller = state.get("controller_pid")
    alive = False
    if controller:
        try:
            command = Path(f"/proc/{controller}/cmdline").read_bytes()
            alive = b"backfill/node.py" in command and job.encode() in command
        except FileNotFoundError:
            pass
    failed = (controller and not alive) or (
        controller and time.time() - state.get("heartbeat", began) > 180)
    failed = failed or (not controller and time.time() - began > 120)
    if failed:
        print("watchdog detected missing or stalled controller", job, flush=True)
        if alive:
            try:
                os.kill(controller, signal.SIGTERM)
                time.sleep(20)
            except ProcessLookupError:
                pass
        resume()
        break
    time.sleep(10)
guardian_status.write_text(json.dumps(dict(state="finished", pid=os.getpid(),
                                         job=job, finished_at=time.time())))
