"""Training loop for one (dataset, model, seed) cell of the matrix.

    python -m paclock_bench.training.train --config configs/experiments/<exp>.yaml \
        [--seed 0] [--out runs/]

Writes ``runs/<name>/seed<k>/result.json`` holding the test metrics, the full
validation curve, the parameter count, and the manifest the data came from. The
result carries its own verdict from the epoch-0-peak check (hard rule 3), so the
table writer never has to re-derive whether a cell is admissible.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from ..data.datasets import build_dataloaders
from ..models.build import build_model, count_params
from .losses import build_loss
from .metrics import compute_metrics, epoch0_peak_check, primary_metric


def set_seed(seed: int) -> None:
    """Seed every RNG that can affect a run.

    ``random`` is seeded too: the reference repo found that Python's module-level
    RNG drove per-batch augmentation choice and was the main source of
    run-to-run variance, because torch/numpy seeding does not touch it.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



def _flatten_seq(logits, y):
    """Collapse a per-epoch sequence prediction to flat (N, K) / (N,).

    CBraMod's ISRUC model (model_for_isruc.py) classifies every epoch in a
    20-epoch sequence, so it returns (B, seq, K) against (B, seq) labels. Every
    other model returns (B, K). Flattening here keeps the loss, the metrics and
    the seed-averaging identical for both, and each epoch stays one prediction,
    which is what the corpus's kappa is defined over.
    """
    if logits.dim() == 3:
        logits = logits.reshape(-1, logits.shape[-1])
        y = y.reshape(-1)
    return logits, y


def amp_context(cfg: dict, device):
    """bf16 autocast when the config asks for it, else a no-op.

    bf16 rather than fp16: no loss scaling needed, and the dynamic range matters
    here because the frontend produces analytic-signal amplitudes that span
    several orders of magnitude across bands.

    The frontend is NOT covered by this. ``analytic.py`` already forces fp32 for
    the Hilbert transform, and the sinc filterbank and the phase estimate it
    feeds are the model's central quantity -- a gauge-invariant phase built from
    a bf16 arctangent is not the model. Autocast wraps the whole forward, but
    torch keeps the explicitly-cast fp32 regions in fp32, so the split is the
    existing one.
    """
    import contextlib
    if not cfg.get("amp"):
        return contextlib.nullcontext()
    dev = "cuda" if str(device).startswith("cuda") else "cpu"
    return torch.autocast(device_type=dev, dtype=torch.bfloat16)


@torch.no_grad()
def evaluate(model, loader: DataLoader, device, criterion, num_classes: int,
             cfg: dict | None = None, return_raw: bool = False):
    model.eval()
    cfg = cfg or {}
    losses, logits_all, y_all = [], [], []
    for X, y in loader:
        X = X.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True).long()
        with amp_context(cfg, device):
            logits = model(X)
        logits, y = _flatten_seq(logits, y)
        losses.append(criterion(logits.float(), y).item())
        logits_all.append(logits.float().cpu().numpy())
        y_all.append(y.cpu().numpy())
    logits_all = np.concatenate(logits_all)
    y_all = np.concatenate(y_all)
    m = compute_metrics(y_all, logits_all, num_classes)
    if return_raw:
        return float(np.mean(losses)), m, logits_all, y_all
    return float(np.mean(losses)), m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--out", default="runs")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    if args.seed is not None:
        cfg["seed"] = args.seed
    seed = cfg.get("seed", 0)
    set_seed(seed)

    device = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    dataset = cfg["dataset"]
    key = primary_metric(dataset)
    # Which metric picks the checkpoint, which is not the same question as which
    # metric gets reported. The protocol fixes the reported one; the model's own
    # repo fixes the selection one, and hard rule 2 covers that too.
    #
    # It matters most where the two disagree in variance. CHB-MIT's validation
    # split holds 150 positives in 21184 windows, so PR-AUC there swings between
    # 0.007 and 0.5 between consecutive evaluations while AUROC barely moves.
    # Selecting -- and early-stopping -- on that noise ended CBraMod's
    # from-scratch runs after one epoch, at chance. CBraMod's own trainer never
    # early-stops and keeps the best checkpoint by `roc_auc > roc_auc_best`.
    #
    # Defaults to the primary metric, so every config that does not set it is
    # bit-identical to before.
    select_key = cfg.get("select_metric", key)

    # Group B runs each repo's own loader and normalisation -- the protocol
    # forbids feeding these models our pipeline's data. `loader` selects it.
    if cfg.get("loader") == "biot":
        from ..data.biot_dataset import build_biot_dataloaders
        train_loader, val_loader, test_loader, info = build_biot_dataloaders(cfg)
    elif cfg.get("loader") == "tfm":
        # TFM normalises with q95 in its Dataset, same as BIOT
        from ..data.biot_dataset import build_biot_dataloaders
        train_loader, val_loader, test_loader, info = build_biot_dataloaders(cfg)
    elif cfg.get("loader") == "labram":
        from ..data.labram_dataset import build_labram_dataloaders
        train_loader, val_loader, test_loader, info = build_labram_dataloaders(cfg)
    else:
        train_loader, val_loader, test_loader, info = build_dataloaders(cfg)
    # Checkpoint selection needs a *dense* validation curve -- hard rule 3 asks
    # whether the peak sits at the first evaluation, which is unanswerable when
    # one epoch is thousands of steps and only one point is recorded. But TUSZ's
    # dev split is 156,395 windows, so validating every 100 steps costs three
    # times what training does, and the run cannot finish.
    #
    # So mid-training selection may use a fixed subset of val. It is fixed
    # (seed 0, not the run's seed) so every seed and every model selects against
    # exactly the same windows, and the reported metrics still come from the
    # full test split -- only the checkpoint-picking signal is subsampled.
    val_cap = int(cfg.get("val_subsample", 0))
    if val_cap and len(val_loader.dataset) > val_cap:
        from torch.utils.data import Subset                      # noqa: PLC0415

        g = np.random.default_rng(0)
        idx = np.sort(g.choice(len(val_loader.dataset), val_cap, replace=False))
        val_loader = DataLoader(
            Subset(val_loader.dataset, idx.tolist()),
            batch_size=val_loader.batch_size, shuffle=False,
            num_workers=cfg.get("num_workers", 16), pin_memory=True)
        print(f"  val subsampled {len(idx)} / {len(val_loader.dataset) if False else val_cap} "
              f"windows for checkpoint selection (test split untouched)", flush=True)

    model = build_model(cfg, info["input_shape"]).to(device)
    if cfg.get("freeze_backbone"):
        # Linear-probe protocol: the (pretrained or control) backbone stays
        # fixed and only the fresh corpus-specific parts train -- head,
        # spatial_pe, montage where configured. The freeze set is exactly the
        # checkpoint-transferable set (_BACKBONE_PREFIXES), so a probe on a
        # random-init model freezes the same tensors and the three-way
        # comparison (v2 / v1 / random) moves a single variable.
        n_frozen = 0
        for pname, prm in model.named_parameters():
            if pname.startswith(("frontend.", "band_pe.", "encoder.")):
                prm.requires_grad_(False)
                n_frozen += 1
        print(f"  freeze_backbone: {n_frozen} tensors frozen; "
              f"training head/spatial_pe only", flush=True)
    if cfg.get("compile"):
        # The tri-axial block is a chain of permute/reshape/LayerNorm around three
        # short attentions, so most of its kernels move memory rather than do
        # arithmetic. Those are exactly what inductor fuses. Opt-in: compilation
        # costs a minute of warm-up and ROCm support is newer than CUDA's, so a
        # run that does not ask for it is untouched.
        model = torch.compile(model)
    criterion = build_loss(cfg)
    n_params = count_params(model)

    print(f"[{cfg['name']}] seed={seed} {n_params:.2f}M params on {device}", flush=True)
    print(f"  input {info['input_shape']} | samples {info['n_samples']}", flush=True)
    print(f"  primary metric: {key} | loss: {cfg['loss']}", flush=True)

    # Group A calibrates against published numbers, so it must train the way
    # those numbers were trained: BIOT uses plain Adam. AdamW decouples weight
    # decay, which at wd=1e-5 is a small but real difference in the objective.
    opt_name = cfg.get("optimizer", "adamw").lower()
    opt_cls = {"adam": torch.optim.Adam, "adamw": torch.optim.AdamW}[opt_name]
    lr = cfg.get("lr", 1e-4)
    wd = cfg.get("weight_decay", 1e-5)
    if cfg.get("layer_decay"):
        # LaBraM scales the LR by depth (--layer_decay 0.65). Without it the
        # pretrained blocks train at the head's rate and the recipe is not
        # LaBraM's any more.
        from .param_groups import layer_decay_param_groups, paclock_layer_decay_param_groups

        if hasattr(model, "encoder") and hasattr(model.encoder, "blocks"):
            groups = paclock_layer_decay_param_groups(model, lr, wd, cfg["layer_decay"])
        else:
            groups = layer_decay_param_groups(model, lr, wd, cfg["layer_decay"])
        optimizer = opt_cls(groups)
        lrs = sorted({round(g["lr"], 8) for g in groups})
        print(f"  layer_decay={cfg['layer_decay']}: {len(groups)} groups, "
              f"lr {lrs[0]:.2e} .. {lrs[-1]:.2e}", flush=True)
    elif cfg.get("multi_lr") and hasattr(model, "backbone_parameters"):
        # CBraMod's finetune_trainer.py gives the pretrained backbone the
        # configured lr and the fresh head a batch-scaled one:
        #   {'params': other_params, 'lr': 0.001*(batch_size/256)**0.5}
        head_lr = 0.001 * (cfg.get("batch_size", 64) / 256) ** 0.5
        optimizer = opt_cls(
            [{"params": list(model.backbone_parameters()), "lr": lr},
             {"params": list(model.head_parameters()), "lr": head_lr}],
            weight_decay=wd,
        )
        print(f"  multi_lr: backbone={lr:g} head={head_lr:g}", flush=True)
    else:
        optimizer = opt_cls(model.parameters(), lr=lr, weight_decay=wd)
    epochs = cfg.get("epochs", 20)
    steps_per_epoch = max(len(train_loader), 1)
    scheduler = None
    if cfg.get("scheduler") == "cosine":
        total_steps = epochs * steps_per_epoch
        warmup_epochs = cfg.get("warmup_epochs", 0)
        if warmup_epochs:
            # LaBraM's --warmup_epochs 5: linear ramp from ~0, then cosine.
            # Starting a pretrained model at full LR undoes the pretraining in
            # the first few hundred steps, which is what warmup exists to avoid.
            warmup_steps = warmup_epochs * steps_per_epoch
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                [torch.optim.lr_scheduler.LinearLR(
                     optimizer, start_factor=1e-3, total_iters=max(warmup_steps, 1)),
                 torch.optim.lr_scheduler.CosineAnnealingLR(
                     optimizer, T_max=max(total_steps - warmup_steps, 1))],
                milestones=[max(warmup_steps, 1)],
            )
            print(f"  warmup {warmup_epochs} epochs ({warmup_steps} steps) "
                  f"then cosine", flush=True)
        else:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=total_steps)
    elif cfg.get("scheduler") == "onecycle":
        # EEGPT's cross-corpus fine-tuning scripts (finetune_BIOT_SleepEDF.py,
        # finetune_LaBraM_SleepEDF.py, linear_probe_*_BCIC2A.py) all use:
        #   OneCycleLR(optimizer, max_lr=max_lr, steps_per_epoch=...,
        #              epochs=max_epochs, pct_start=0.2)
        # Those scripts are the published provenance of the BIOT/LaBraM numbers
        # on the corpora neither repo ships a maker for, so their schedule is
        # used verbatim on those rows rather than each repo's own TUH recipe.
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=cfg.get("max_lr", lr),
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            pct_start=cfg.get("pct_start", 0.2),
        )
        print(f"  onecycle max_lr={cfg.get('max_lr', lr)} pct_start="
              f"{cfg.get('pct_start', 0.2)}", flush=True)

    # Mid-epoch validation: on the large corpora one epoch is thousands of steps
    # and models peak inside epoch 0-1, so validating once per epoch samples the
    # curve too coarsely for best-checkpoint selection to land near the optimum.
    eval_every_steps = cfg.get("eval_every_steps", 0)

    best, best_state, val_curve = -np.inf, None, []
    patience, since_best = cfg.get("patience", 0), 0
    # Hours of training after which the run stops and reports what it has. None
    # keeps the old behaviour exactly.
    max_hours = cfg.get("max_hours")
    stopped_by = "epochs"
    t0 = time.time()

    def validate(tag: str):
        nonlocal best, best_state, since_best
        _, m = evaluate(model, val_loader, device, criterion, cfg["num_classes"], cfg)
        # The curve stays on the reported metric -- it is what rule 3 reads and
        # what the workbook's provenance notes quote -- while selection follows
        # select_key.
        val_curve.append(m[key])
        print(f"  {tag} val " + " ".join(f"{k}={v:.4f}" for k, v in m.items()), flush=True)
        if m[select_key] > best:
            best, since_best = m[select_key], 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            since_best += 1
        model.train()

    # Staged finetune (LP-FT style): for the first N epochs only the tensors
    # that did NOT come from the checkpoint train (fresh tokenizer convs, head,
    # spatial_pe), so a random tokenizer cannot drag the pretrained encoder
    # off its solution before it has learned to speak the encoder's language.
    freeze_loaded_epochs = int(cfg.get("freeze_loaded_epochs", 0))
    loaded_keys = getattr(model, "_loaded_keys", set())
    if freeze_loaded_epochs and loaded_keys:
        n_frz = 0
        for pname, prm in model.named_parameters():
            if pname in loaded_keys:
                prm.requires_grad_(False)
                n_frz += 1
        print(f"  stage 1: {n_frz} checkpoint-loaded tensors frozen for "
              f"{freeze_loaded_epochs} epoch(s); fresh tensors train", flush=True)

    epoch = 0
    for epoch in range(epochs):
        model.train()
        if freeze_loaded_epochs and loaded_keys and epoch == freeze_loaded_epochs:
            for pname, prm in model.named_parameters():
                if pname in loaded_keys:
                    prm.requires_grad_(True)
            since_best = 0          # stage-1 plateaus must not count toward patience
            print(f"  stage 2: all tensors trainable from epoch {epoch}", flush=True)
        running = []
        # Opt-in step breakdown. Three rounds of inferring the bottleneck from
        # aggregate throughput were wrong (kernel launches, then Lustre reads,
        # then process packing), while an isolated profile of the same model and
        # loader measured a 535 samples/s ceiling against 31 observed. The parts
        # have to be timed inside the real loop, not next to it.
        prof_n = int(cfg.get("profile_steps", 0))
        prof = {"wait": 0.0, "h2d": 0.0, "fwd": 0.0, "bwd": 0.0, "opt": 0.0}
        _mark = time.time()
        for step, (X, y) in enumerate(train_loader):
            if prof_n:
                torch.cuda.synchronize(); prof["wait"] += time.time() - _mark
                _mark = time.time()
            X = X.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True).long()
            if prof_n:
                torch.cuda.synchronize(); prof["h2d"] += time.time() - _mark
                _mark = time.time()
            with amp_context(cfg, device):
                logits = model(X)
            # loss in fp32: bf16 cross-entropy over many classes loses enough
            # precision in the log-sum-exp to change the gradient direction
            logits, y = _flatten_seq(logits.float(), y)
            loss = criterion(logits, y)
            if prof_n:
                torch.cuda.synchronize(); prof["fwd"] += time.time() - _mark
                _mark = time.time()
            # Gradient accumulation (accum_steps, default 1 = exact old
            # behaviour). Exists because attention memory scales with
            # batch x tokens^2, so on some corpora the physical batch that
            # fits is smaller than the effective batch the class balance
            # needs -- Siena at 0.95% positives wants an effective 128, but
            # (19,2000) windows OOM the 64GB MI210 above physical 32 (jobs
            # 386667, 387122). Equal-sized micro-batches with loss/accum
            # reproduce the large-batch mean-loss gradient exactly; the
            # scheduler steps once per OPTIMIZER step, not per micro-batch.
            accum = int(cfg.get("accum_steps", 1))
            if accum == 1:
                optimizer.zero_grad(set_to_none=True)
            elif step % accum == 0:
                optimizer.zero_grad(set_to_none=True)
            (loss / accum).backward() if accum > 1 else loss.backward()
            if prof_n:
                torch.cuda.synchronize(); prof["bwd"] += time.time() - _mark
                _mark = time.time()
            if accum == 1 or (step + 1) % accum == 0:
                if cfg.get("grad_clip"):
                    torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                   cfg["grad_clip"])
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
            running.append(loss.item())
            if prof_n:
                torch.cuda.synchronize(); prof["opt"] += time.time() - _mark
                if step + 1 >= prof_n:
                    tot = sum(prof.values()); n = prof_n * X.shape[0]
                    print(f"  [profile] {prof_n} steps, {n} samples, {tot:.2f}s "
                          f"= {n/tot:.1f} samples/s", flush=True)
                    for k, v in prof.items():
                        print(f"    {k:5s} {v:7.2f}s  {100*v/tot:5.1f}%", flush=True)
                    raise SystemExit(0)
                _mark = time.time()
            if eval_every_steps and (step + 1) % eval_every_steps == 0:
                validate(f"epoch {epoch} step {step + 1} |")

        print(f"epoch {epoch:3d} | train_loss {np.mean(running):.4f}", flush=True)
        validate(f"epoch {epoch:3d} |")
        # No early stop while the encoder is still frozen or the LR still warming
        # up: a stage-1 plateau is by construction, not a converged model (TUEP
        # ptS was cut at epoch 1 by exactly this, 2026-09-03).
        no_stop_before = freeze_loaded_epochs + int(cfg.get("warmup_epochs", 0))
        if patience and since_best >= patience and epoch >= no_stop_before:
            print(f"early stop: no val {key} gain for {patience} evals", flush=True)
            stopped_by = "patience"
            break
        # Wall-clock budget. Without it a run that cannot finish its epoch count
        # inside the partition limit produces NOTHING: result.json is written
        # after the test pass, so a TIMEOUT kills the process with the whole run
        # unrecorded -- which is how two TUAB seeds were lost at 12h. PACLock on
        # TUAB at batch 32 needs one epoch per ~14.5h and is configured for 20,
        # so it would always have ended that way.
        #
        # Breaking here instead keeps the best checkpoint, runs the test pass and
        # writes a real result, with `stopped_by` recording that the schedule did
        # not run to completion so the cell can never be read as if it had.
        if max_hours and (time.time() - t0) / 3600.0 >= max_hours:
            print(f"stopping: {max_hours}h wall-clock budget reached at "
                  f"epoch {epoch}", flush=True)
            stopped_by = "time_budget"
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    _, test_m, test_logits, test_y = evaluate(
        model, test_loader, device, criterion, cfg["num_classes"], cfg,
        return_raw=True)
    print("test | " + " ".join(f"{k}={v:.4f}" for k, v in test_m.items()), flush=True)

    # Hard rule 3 is evaluated here so it travels with the result.
    # The check needs to know what "chance" is on this corpus. For PR-AUC that
    # is the val positive rate, not a constant -- CHB-MIT is 0.71% positive.
    val_counts = (info.get("class_counts") or {}).get("val")
    prevalence = None
    if val_counts and len(val_counts) == 2 and sum(val_counts):
        prevalence = val_counts[1] / sum(val_counts)
    verdict = epoch0_peak_check(val_curve, key, cfg["num_classes"],
                                prevalence, test_m)
    if not verdict["ok"]:
        print(f"WARNING mis-configured: {verdict['reason']} -- "
              f"this cell must not be written to the matrix", flush=True)

    out_dir = os.path.join(args.out, cfg["name"], f"seed{seed}")
    os.makedirs(out_dir, exist_ok=True)
    # Per-window test scores beside result.json (diagnosis needs them: the
    # Siena incident, AUROC 0.87 vs AUC-PR 0.07, was unattributable without
    # per-subject scores). Placed AFTER out_dir exists -- the first version
    # of this dump referenced an undefined run_dir and killed SIENA_d2 at
    # the finish line after a completed 2h training.
    np.savez_compressed(os.path.join(out_dir, "test_scores.npz"),
                        logits=test_logits.astype(np.float16), y=test_y)
    result = {
        "name": cfg["name"],
        "dataset": dataset,
        "model": cfg["model"],
        "group": cfg.get("group"),
        "seed": seed,
        "n_params_M": n_params,
        "primary_metric": key,
        "test": test_m,
        "best_val": float(best),
        "val_curve": [float(v) for v in val_curve],
        "verdict": verdict,
        "epochs_run": epoch + 1,
        # "epochs" | "patience" | "time_budget" -- a cell stopped by the budget
        # was not trained to its own schedule and must be reported as such.
        "stopped_by": stopped_by,
        "wall_time_sec": time.time() - t0,
        "config": cfg,
        "data_manifest_created": info["manifest"].get("created_utc"),
        "class_counts": info["class_counts"],
    }
    # Learned coupling gates, reported per corpus in the paper: |beta_b|
    # (fused-row blend, zero-init) and gamma_b (interaction-row gate,
    # unit-init). Finetune runs keep no checkpoint, so this is the only
    # record of what each task retained.
    try:
        fe = getattr(model, "frontend", None)
        gates = {}
        if fe is not None and hasattr(fe, "fusion_beta"):
            gates["beta_norm"] = fe.fusion_beta.detach().norm(dim=-1).cpu().tolist()
        if fe is not None and hasattr(fe, "interaction_gate"):
            gates["gamma"] = fe.interaction_gate.detach().cpu().tolist()
        if gates:
            result["gates"] = gates
    except Exception as e:                                    # noqa: BLE001
        print(f"[gates] not recorded: {type(e).__name__}: {e}", flush=True)
    with open(os.path.join(out_dir, "result.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(f"-> {out_dir}/result.json", flush=True)


if __name__ == "__main__":
    main()
