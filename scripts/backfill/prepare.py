"""Snapshot this user's PACLock pending jobs and running node allocations."""
import concurrent.futures
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import yaml

queue = Path(sys.argv[1])
queue.mkdir(exist_ok=False)
for folder in ("pending", "running", "done", "failed", "nodes", "allocations", "configs", "logs"):
    (queue / folder).mkdir()
base = Path("/work1/chenyuyou/yifanwang/Zhizhe")
repos = {str(base / "PACLock"), str(base / "PACLock-factorized")}
raw = subprocess.check_output(["squeue", "-h", "-u", "yifanwang", "-o", "%i|%j|%T|%N"], text=True)
entries = []
for row in raw.splitlines():
    jid, name, state, node = row.split("|")
    line = subprocess.check_output(["scontrol", "show", "job", "-o", jid], text=True)
    work = re.search(r"WorkDir=(\S+)", line).group(1)
    if work not in repos:
        continue
    entry = dict(job=jid, name=name, state=state, node=node, repo=work)
    if state == "PENDING" and name.startswith(("FACT_", "SELF_")):
        cmd = re.search(r"Command=(.*?) WorkDir=", line).group(1).strip()
        specs = shlex.split(cmd)[1:]
        assert len(specs) == 4, (jid, specs)
        entry["specs"] = specs
        entry["items"] = []
        for spec in specs:
            cfgname, sep, seed = spec.partition(":")
            seed = int(seed) if sep else 0
            data = (Path(work) / cfgname).read_bytes()
            cfg = yaml.safe_load(data)
            ds = cfg["dataset"]
            # Conservative admission floors; the one-hour shutdown reserve
            # is additionally enforced in each generated runtime config.
            min_hours = {"tusz": 16, "chbmit": 14, "iiic": 9,
                         "isruc": 9, "tuev": 7, "tuar": 4, "sleepedf": 4}[ds]
            key = f"{24-min_hours:02d}_{jid}_{Path(cfgname).stem}_s{seed}"
            item = dict(source_job=jid, source_name=name, repo=work, config=cfgname,
                        seed=seed, name=cfg["name"], dataset=ds, min_remaining_hours=min_hours,
                        sha256=hashlib.sha256(data).hexdigest())
            (queue / "pending" / (key + ".json")).write_text(json.dumps(item, indent=2))
            entry["items"].append(key)
        entries.append(entry)
    elif state == "RUNNING":
        end = re.search(r"EndTime=(\S+)", line).group(1)
        entry["end_epoch"] = datetime.fromisoformat(end).timestamp()
        entries.append(entry)

def inspect(entry):
    if entry["state"] != "RUNNING": return entry
    cmd = ["ssh", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new", entry["node"],
           "ps -u yifanwang -o pid=,ppid=,args="]
    lines = subprocess.check_output(cmd, text=True, timeout=15).splitlines()
    matching = [line.strip().split(None, 2) for line in lines
                if f"/job{entry['job']}/slurm_script" in line]
    if len(matching) != 1: raise RuntimeError((entry, matching))
    entry["batch_pid"] = int(matching[0][0])
    (queue / "allocations" / (entry["job"] + ".json")).write_text(json.dumps(entry, indent=2))
    return entry

with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
    entries = list(pool.map(inspect, entries))
(queue / "migration.json").write_text(json.dumps(entries, indent=2))
print("prepared", sum(e["state"] == "PENDING" for e in entries), "pending packs,",
      len(list((queue / "pending").glob("*.json"))), "configurations,",
      sum(e["state"] == "RUNNING" for e in entries), "allocations")
