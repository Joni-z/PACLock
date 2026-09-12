"""Hold source packs, start protected allocation steps, then activate separately.

No cancellation or activation happens here: inspect ready receipts first.
"""
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time

queue = Path(sys.argv[1])
here = Path(__file__).resolve().parent
entries = json.loads((queue / "migration.json").read_text())
receipt = {"at": time.time(), "held": [], "controllers": [], "skipped": []}
for entry in entries:
    if entry["state"] != "PENDING":
        continue
    job = entry["job"]
    detail = subprocess.check_output(["scontrol", "show", "job", "-o", job], text=True)
    assert "UserId=yifanwang(" in detail and "JobState=PENDING" in detail, (job, detail)
    subprocess.run(["scontrol", "hold", job], check=True)
    receipt["held"].append(job)
    (queue / "bootstrap.json").write_text(json.dumps(receipt, indent=2))

for entry in entries:
    if entry["state"] != "RUNNING":
        continue
    job, node = entry["job"], entry["node"]
    detail = subprocess.check_output(["scontrol", "show", "job", "-o", job], text=True)
    if "JobState=RUNNING" not in detail:
        receipt["skipped"].append(job)
        continue
    assert "UserId=yifanwang(" in detail and f"NodeList={node}" in detail
    ncpu = int(re.search(r"NumCPUs=(\d+)", detail).group(1))
    status = queue / "nodes" / (job + ".json")
    if status.exists():
        assert json.loads(status.read_text()).get("state") == "released"
        status.unlink()
    # SSH runs outside the controller's srun step; losing that step therefore
    # cannot leave the allocation's batch shell stopped indefinitely.
    guardian = shlex.join(["nohup", "python3", str(here / "guardian.py"), str(queue), job])
    guardian += " > " + shlex.quote(str(queue / "logs" / ("guardian-" + job + ".out"))) + " 2>&1 < /dev/null &"
    subprocess.run(["ssh", "-o", "BatchMode=yes", node, guardian], check=True, timeout=20)
    logpath = queue / "logs" / ("node-" + job + ".out")
    inner = "module load pytorch/2.7.1 && exec " + shlex.join([
        "python3", "-u", str(here / "node.py"), "--queue", str(queue), "--job", job])
    with logpath.open("w") as log:
        proc = subprocess.Popen([
            "srun", "--jobid=" + job, "--overlap", "--nodes=1", "--ntasks=1",
            "--cpus-per-task=" + str(ncpu), "--cpu-bind=none", "bash", "-lc", inner],
            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True)
    receipt["controllers"].append(dict(job=job, node=node, srun_pid=proc.pid, cpus=ncpu))
    (queue / "bootstrap.json").write_text(json.dumps(receipt, indent=2))
    print("bootstrapped", job, node, "cpus", ncpu, flush=True)
print("Sources held; inspect nodes/*.json before cancelling and activating.", flush=True)
