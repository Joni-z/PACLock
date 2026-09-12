"""Read-only queue status, optionally verify root training processes per node."""
import argparse
from collections import Counter
import concurrent.futures
import getpass
import json
from pathlib import Path
import shlex
import subprocess
import time

ap = argparse.ArgumentParser()
ap.add_argument("queue")
ap.add_argument("--live", action="store_true")
ap.add_argument("--save")
args = ap.parse_args()
queue = Path(args.queue)
report = dict(at=time.time(), queue=str(queue), counts={}, datasets={}, nodes=[])
slurm = {}
if args.live:
    raw = subprocess.check_output(["squeue", "-h", "-u", getpass.getuser(),
                                   "-o", "%i|%T|%N"], text=True)
    slurm = {row.split("|")[0]: row.split("|")[1:] for row in raw.splitlines()}
for state in ("pending", "running", "done", "failed", "deferred"):
    entries = []
    for p in (queue / state).glob("*.json"):
        try:
            entries.append(json.loads(p.read_text()))
        except FileNotFoundError:
            pass
    report["counts"][state] = len(entries)
    report["datasets"][state] = dict(Counter(e["dataset"] for e in entries))
    print(state, len(entries), report["datasets"][state])
    if state in ("pending", "failed", "deferred"):
        for entry in entries:
            print(" ", entry["name"], entry.get("error", ""))

probe = r'''
import json, os, subprocess
from pathlib import Path
lines = subprocess.check_output(["ps", "-u", str(os.getuid()), "-o", "pid=,ppid=,stat=,args="], text=True)
rows = {}
for line in lines.splitlines():
    fields = line.strip().split(None, 3)
    if len(fields) == 4 and fields[3].startswith(("python3 -u -m paclock_bench.training.train", "python -u -m paclock_bench.training.train", "/")):
        if " -m paclock_bench.training.train " in fields[3]:
            rows[int(fields[0])] = (int(fields[1]), fields[2], fields[3])
roots = []
for pid, (parent, state, cmd) in rows.items():
    if parent in rows: continue
    try:
        env = Path(f"/proc/{pid}/environ").read_bytes().split(b"\0")
        gpu = [x.split(b"=", 1)[1].decode() for x in env if x.startswith(b"HIP_VISIBLE_DEVICES=")]
        roots.append(dict(pid=pid, state=state, gpu=gpu, backfill="PACLock-backfill-" in cmd))
    except FileNotFoundError: pass
print(json.dumps(roots))
'''

def inspect(path):
    state = json.loads(path.read_text())
    # A drained controller releases the batch shell while original training
    # can remain active. Controller state is not the allocation's Slurm state.
    live_job = slurm.get(state["job"], ["NOT_RUNNING", ""])
    state["slurm_state"] = live_job[0]
    if args.live and live_job[0] == "RUNNING":
        state["node"] = live_job[1]
        cmd = ["ssh", "-o", "BatchMode=yes", state["node"], shlex.join(["python3", "-c", probe])]
        state["training_roots"] = json.loads(subprocess.check_output(cmd, text=True, timeout=30))
        state["busy"] = sorted({int(g) for r in state["training_roots"]
                                for field in r["gpu"] for g in field.split(",") if g.isdigit()})
    return state

paths = [p for p in (queue / "nodes").glob("*.json") if ".guardian." not in p.name]
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
    report["nodes"] = list(pool.map(inspect, sorted(paths)))
for state in report["nodes"]:
    roots = state.get("training_roots", [])
    added = sum(r["backfill"] for r in roots)
    print(state["job"], state["state"], state.get("busy", []),
          "original=" + str(len(roots)-added), "backfill=" + str(added))
if args.live:
    roots = [r for s in report["nodes"] for r in s.get("training_roots", [])]
    report["live_total"] = len(roots)
    report["live_backfill"] = sum(r["backfill"] for r in roots)
    report["stopped_training"] = [r for r in roots if "T" in r["state"]]
    print("LIVE", len(roots), "backfill", report["live_backfill"],
          "stopped training", len(report["stopped_training"]))
if args.save:
    Path(args.save).write_text(json.dumps(report, indent=2))
