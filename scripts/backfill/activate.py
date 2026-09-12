"""Validate the migration, cancel only held source packs, and activate once."""
import json
from pathlib import Path
import subprocess
import sys
import time

queue = Path(sys.argv[1])
assert not (queue / "ACTIVE").exists(), "Queue already active"
receipt = json.loads((queue / "bootstrap.json").read_text())
assert len(receipt["controllers"]) > 0
for entry in receipt["controllers"]:
    job = entry["job"]
    node = json.loads((queue / "nodes" / (job + ".json")).read_text())
    guardian = json.loads((queue / "nodes" / (job + ".guardian.json")).read_text())
    assert node["state"] == "ready" and time.time() - node["heartbeat"] < 30, node
    assert guardian["state"] == "watching", guardian
for job in receipt["held"]:
    detail = subprocess.check_output(["scontrol", "show", "job", "-o", job], text=True)
    assert "UserId=yifanwang(" in detail and "JobState=PENDING" in detail, detail
    assert "Reason=JobHeldUser" in detail, detail
items = [json.loads(p.read_text()) for p in (queue / "pending").glob("*.json")]
assert len(items) == 4 * len(receipt["held"])
assert len({(x["name"], x["seed"]) for x in items}) == len(items), "Duplicate runs"
assert {x["source_job"] for x in items} == set(receipt["held"])
assert not list((queue / "running").glob("*.json"))
activation = dict(at=time.time(), configs=len(items), cancelled=[])
for job in receipt["held"]:
    subprocess.run(["scancel", job], check=True)
    activation["cancelled"].append(job)
    (queue / "activation.json").write_text(json.dumps(activation, indent=2))
for job in receipt["held"]:
    detail = subprocess.check_output(["scontrol", "show", "job", "-o", job], text=True)
    assert "JobState=CANCELLED" in detail, detail
temp = queue / "ACTIVE.tmp"
temp.write_text(json.dumps(activation, indent=2))
temp.replace(queue / "ACTIVE")
print("Activated", len(items), "configurations from", len(receipt["held"]),
      "cancelled pending packs on", len(receipt["controllers"]), "allocations")
