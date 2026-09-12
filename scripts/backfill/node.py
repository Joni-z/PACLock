"""Fill idle GPUs inside an existing exclusive SLURM allocation.

Run as an srun step. Hold only the batch *shell*, never its training children,
so it cannot exit and destroy added work. Always resume it when draining.
The shared queue contains only explicitly migrated, cancelled pending jobs.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def write_json(path, obj):
    tmp = path.with_name(path.name + ".tmp." + str(os.getpid()))
    tmp.write_text(json.dumps(obj, indent=2))
    tmp.replace(path)


def process_table():
    rows = {}
    text = subprocess.check_output(
        ["ps", "-u", str(os.getuid()), "-o", "pid=,ppid=,args="], text=True)
    for line in text.splitlines():
        fields = line.strip().split(None, 2)
        if len(fields) == 3:
            rows[int(fields[0])] = (int(fields[1]), fields[2])
    return rows


def busy_gpus():
    # Read only the GPU selector, never print process environments.
    busy = set()
    for pid, (_, args) in process_table().items():
        if "python" not in args or pid == os.getpid():
            continue
        try:
            env = Path(f"/proc/{pid}/environ").read_bytes().split(b"\0")
            for field in env:
                if field.startswith(b"HIP_VISIBLE_DEVICES="):
                    busy.update(int(x) for x in field.split(b"=", 1)[1].split(b",") if x.isdigit())
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            pass
    return busy


def claim(queue, remaining_hours):
    for path in sorted((queue / "pending").glob("*.json")):
        try:
            item = json.loads(path.read_text())
            if item["min_remaining_hours"] > remaining_hours:
                continue
            target = queue / "running" / path.name
            path.rename(target)  # one consumer wins; others see ENOENT
        except FileNotFoundError:
            continue
        return target, item
    return None


def has_eligible(queue, remaining):
    for path in (queue / "pending").glob("*.json"):
        try:
            if json.loads(path.read_text())["min_remaining_hours"] <= remaining:
                return True
        except FileNotFoundError:
            continue
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queue", required=True)
    ap.add_argument("--job", required=True)
    ap.add_argument("--probe", action="store_true")
    args = ap.parse_args()
    queue = Path(args.queue)
    if os.environ.get("SLURM_JOB_ID") != args.job:
        raise RuntimeError("Must execute inside the specified SLURM allocation")
    allocation = json.loads((queue / "allocations" / (args.job + ".json")).read_text())
    pid = allocation["batch_pid"]
    rows = process_table()
    if pid not in rows or f"/job{args.job}/slurm_script" not in rows[pid][1]:
        raise RuntimeError("Batch shell PID no longer matches allocation")
    before = Path(f"/proc/{pid}/stat").read_text().split(") ", 1)[1].split()[19]
    lock = queue / "nodes" / (args.job + ".lock")
    lock.mkdir()
    active = {}
    held = False
    status_file = queue / "nodes" / (args.job + ".json")

    def interrupted(signum, frame):
        raise InterruptedError("controller signal " + str(signum))

    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, interrupted)
    try:
        os.kill(pid, signal.SIGSTOP)
        held = True
        # Only this shell is stopped. Verify the state before admitting work.
        time.sleep(.2)
        assert Path(f"/proc/{pid}/stat").read_text().split(") ", 1)[1].split()[0] == "T"
        write_json(status_file, dict(state="ready", controller_pid=os.getpid(),
                                     heartbeat=time.time(), job=args.job,
                                     node=allocation["node"], busy=sorted(busy_gpus())))
        if args.probe:
            time.sleep(2)
            print("BACKFILL_PROBE_OK", flush=True)
            return
        ready_at = time.time()
        while not (queue / "ACTIVE").exists():
            if time.time() - ready_at > 1800:
                raise TimeoutError("Queue activation did not arrive")
            write_json(status_file, dict(state="ready", controller_pid=os.getpid(),
                       heartbeat=time.time(), job=args.job, node=allocation["node"]))
            time.sleep(2)
        while True:
            for gpu, entry in list(active.items()):
                proc, path, item, log = entry
                rc = proc.poll()
                if rc is None:
                    continue
                log.close()
                result = Path(item["repo"]) / "runs" / item["name"] / ("seed" + str(item["seed"])) / "result.json"
                item.update(exit_code=rc, finished_at=time.time(), result=str(result), result_exists=result.exists())
                folder = "done" if rc == 0 and result.exists() else "failed"
                write_json(path, item)
                path.rename(queue / folder / path.name)
                print("finished", item["name"], "gpu", gpu, "rc", rc, flush=True)
                del active[gpu]
            remaining = (allocation["end_epoch"] - time.time()) / 3600
            busy = busy_gpus() | set(active)
            for gpu in sorted(set(range(4)) - busy):
                selected = claim(queue, remaining)
                if selected is None:
                    break
                path, item = selected
                cfg_path = Path(item["repo"]) / item["config"]
                import yaml
                cfg_bytes = cfg_path.read_bytes()
                if hashlib.sha256(cfg_bytes).hexdigest() != item["sha256"]:
                    item["error"] = "Configuration changed since migration"
                    write_json(path, item); path.rename(queue / "failed" / path.name)
                    continue
                cfg = yaml.safe_load(cfg_bytes)
                result = Path(item["repo"]) / "runs" / item["name"] / ("seed" + str(item["seed"])) / "result.json"
                if result.exists():
                    item["skipped"] = "result already exists"
                    write_json(path, item); path.rename(queue / "done" / path.name)
                    continue
                # Preserve epoch/step/early-stop recipe, but make the training
                # loop save its best checkpoint/results before this allocation
                # ends. Budget-limited runs carry stopped_by=time_budget.
                original_max = cfg.get("max_hours", 22.5)
                cfg["max_hours"] = min(float(original_max), remaining - 1.0)
                item.update(allocation=args.job, node=allocation["node"], gpu=gpu,
                            started_at=time.time(), original_max_hours=original_max,
                            max_hours=cfg["max_hours"], slurm_step=os.environ.get("SLURM_STEP_ID"))
                cfg["backfill"] = {k: item[k] for k in ("source_job", "allocation", "gpu", "original_max_hours", "max_hours")}
                runtime = queue / "configs" / (path.stem + ".yaml")
                runtime.write_text(yaml.safe_dump(cfg, sort_keys=False))
                env = dict(os.environ)
                env["HIP_VISIBLE_DEVICES"] = str(gpu)
                env["OMP_NUM_THREADS"] = "8"
                env["MIOPEN_USER_DB_PATH"] = f"/tmp/miopen_bf_{args.job}_{gpu}_{path.stem}"
                env["MIOPEN_CUSTOM_CACHE_DIR"] = env["MIOPEN_USER_DB_PATH"]
                log_path = queue / "logs" / (path.stem + ".out")
                log = log_path.open("w")
                try:
                    proc = subprocess.Popen(
                        [sys.executable, "-u", "-m", "paclock_bench.training.train",
                         "--config", str(runtime), "--seed", str(item["seed"])],
                        cwd=item["repo"], env=env, stdout=log, stderr=subprocess.STDOUT,
                        start_new_session=True)
                except OSError as exc:
                    log.close()
                    item["error"] = "Launch failed: " + str(exc)
                    write_json(path, item)
                    path.rename(queue / "failed" / path.name)
                    continue
                item.update(pid=proc.pid, log=str(log_path))
                write_json(path, item)
                active[gpu] = (proc, path, item, log)
                print("started", item["name"], "gpu", gpu, "pid", proc.pid, flush=True)
            pending = list((queue / "pending").glob("*.json"))
            write_json(status_file, dict(state="running", heartbeat=time.time(), job=args.job,
                       node=allocation["node"], controller_pid=os.getpid(), busy=sorted(busy_gpus()),
                       added={str(g): e[2]["name"] for g, e in active.items()}, remaining_hours=remaining))
            if not active and not has_eligible(queue, remaining):
                break
            time.sleep(15)
    finally:
        for proc, path, item, log in active.values():
            if proc.poll() is None:
                try:
                    os.killpg(proc.pid, signal.SIGTERM)
                    proc.wait(timeout=15)
                except (ProcessLookupError, subprocess.TimeoutExpired):
                    pass
            item.update(error="controller exited before training finished", finished_at=time.time())
            write_json(path, item)
            if path.exists(): path.rename(queue / "failed" / path.name)
            log.close()
        if held:
            try:
                now = Path(f"/proc/{pid}/stat").read_text().split(") ", 1)[1].split()[19]
                if now == before: os.kill(pid, signal.SIGCONT)
            except (FileNotFoundError, ProcessLookupError):
                pass
        write_json(status_file, dict(state="released", at=time.time(), job=args.job))
        lock.rmdir()


if __name__ == "__main__":
    main()
