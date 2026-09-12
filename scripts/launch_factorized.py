"""Submit the six-corpus pilot only after matching GPU smoke evidence."""
import hashlib
import json
from pathlib import Path
import subprocess


def main():
    root = Path(__file__).resolve().parents[1]
    stamp = root / "results/factorized_launch.json"
    if stamp.exists():
        raise SystemExit("Launch record already exists; inspect it before resubmitting")
    smoke = json.loads((root / "results/factorized_smoke.json").read_text())
    if smoke.get("ok") is not True:
        raise SystemExit("GPU smoke did not pass")
    for name, digest in smoke["sha256"].items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != digest:
            raise SystemExit("Changed after smoke: " + name)
    datasets = ("tuev", "chbmit", "tusz", "tuar", "sleepedf", "isruc")
    expected = {(ds, arm) for ds in datasets for arm in ("f0", "f1", "f2", "f3")}
    observed = {(r["dataset"], r["arm"]) for r in smoke["timing"]}
    if observed != expected:
        raise SystemExit("Smoke must cover all 24 real configuration/batch pairs")
    for r in smoke["timing"]:
        if r["peak_GiB"] > 55:
            raise SystemExit("Insufficient memory headroom: " + str(r))
    launched = []
    # Exclusive creation prevents an accidental concurrent second launcher.
    with stamp.open("x") as f:
        json.dump({"jobs": [], "smoke_job": smoke["job_id"]}, f)
    for ds in datasets:
        configs = [f"configs/factorized/{ds}_{arm}.yaml:0" for arm in ("f0", "f1", "f2", "f3")]
        result = subprocess.check_output(
            ["sbatch", "--parsable", "-J", "FACT_" + ds,
             "slurm/configs_packed.slurm", *configs], cwd=root, text=True).strip()
        launched.append({"dataset": ds, "job_id": result, "configs": configs})
        stamp.write_text(json.dumps({"jobs": launched, "smoke_job": smoke["job_id"]}, indent=2))
        print(ds, result, flush=True)


if __name__ == "__main__":
    main()
