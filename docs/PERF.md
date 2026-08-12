# Why PACLock trained 10x slower than it needed to

TUEV, `configs/_cand/tuev_base.yaml`, one MI210, batch 32:

| | s / epoch |
|---|---|
| finished runs (`cand_*`, archived in `runs_conv_tokenizer/`) | **1939** |
| current code, same config, same node type (`diag_trainpy`) | **206** |

The model was never the problem. An isolated full training step measured
51.9 ms — 617 samples/s — against 35.3 samples/s observed, and the forward
splits as encoder 10.77 ms / frontend 2.42 ms / head 0.10 ms.

## The cause: an interaction between two things, each harmless alone

`train.py`'s `set_seed()` sets `torch.backends.cudnn.deterministic = True`,
which on ROCm maps onto MIOpen. The frontend tokenisers were
`nn.Conv1d(1, K, kernel_size=patch_len, stride=patch_len)` applied to
`B*C*n_bands = 4096` single-channel signals. Forced determinism makes MIOpen
choose an atomics-free **backward-weights** algorithm, and for
`in_channels=1` with a 200-tap kernel that is a serial reduction.

Measured 2x2 (full step: forward + backward + clip + optimiser):

| tokeniser | `deterministic` | ms/step | s/epoch |
|---|---|---|---|
| `Conv1d` (old) | **True** | **208.56** | 446 |
| GEMM (new) | True | 66.60 | 142 |
| `Conv1d` (old) | False | 55.46 | 119 |
| GEMM (new) | False | 50.79 | 109 |

`Conv1d` is not slow. `Conv1d` **under forced determinism** is slow, by 3.76x.

## The fix

`_patch_project()` in `models/paclock/frontend/triaxial.py`. When
`stride == kernel_size` the patches do not overlap, so the convolution is a
per-patch linear map and `(N, P, patch) @ W` is the same arithmetic as one
GEMM, whose backward is a GEMM too and needs no special deterministic kernel.

The `nn.Conv1d` modules are kept as the parameter holders and only the
computation is replaced, so initialisation, `state_dict` keys and parameter
count are unchanged and old checkpoints still load.

Verified (`scripts/verify_patch_project.py`): operator `max|d| 2.4e-6`
(rel 7.9e-7), whole-frontend tokens rel 5.4e-7, coupling exactly 0, gradient
w.r.t. input rel 3.3e-7. Mathematically identical, **not** bit-identical — a
convolution and a GEMM reduce over `patch` in a different order.

`deterministic = True` is **kept**: with the convolutions gone it costs 1.31x,
and bit-exact reproducibility is worth more than that. It is what let
`D_repeat` show three identical configs producing identical loss curves —
which, read correctly, was the speed bug announcing itself.

## Why this took four wrong diagnoses

Kernel-launch overhead, then Lustre random reads, then process packing, then
"the frontend's `in_channels=1` convolutions are badly shaped for MIOpen". Each
was inferred from aggregate throughput and each was wrong.

The last one was wrong in an instructive way. Every benchmark varied one of the
two factors **while the other was already in its cheap state**:

* the frontend A/B timed the forward only, and the forward was never the cost —
  it measured 1.05x;
* the determinism A/B ran after the convolutions had already been replaced, so
  there was nothing left for the flag to be slow on — it measured 1.29x.

Neither is wrong as a measurement. Both are useless as an explanation, because
a 2-way interaction is invisible to any experiment that holds the other factor
at its cheap level. The 2x2 above was the first design that could see it.

Also worth keeping: `profile_steps` reports 231 samples/s where an uninstrumented
loop does 613, because its five `torch.cuda.synchronize()` calls per step drain
the queue. Its *percentages* are trustworthy; its absolute rate is not, and the
"packing costs 4%" result taken from it was wrong for the same reason — with
every process stalled on a sync, they cannot contend.

## Consequences for results already in hand

`runs_conv_tokenizer/` holds the 28 pre-change candidate cells. They are not
comparable with anything produced after it: an SDPA swap verified at 4e-7 moved
a finished ISRUC run by 0.029 kappa, seven times the seed spread, so a 1e-6
change to the tokeniser has to be assumed to do the same. The sweep is being
re-run rather than reused.

## Addendum: the size of the "mathematically equivalent" shift, measured

The claim above — that results from before and after the tokeniser change are
not comparable — was originally supported only by an anecdote (an SDPA swap
verified at 4e-7 moved an ISRUC run by 0.029) against an *assumed* seed spread.
Wave 1 measured both properly on ISRUC, 3 seeds, same config:

    conv path (archived)   0.6633
    GEMM path (new)        0.6888  0.6910  0.6929   mean 0.6909  sd 0.0021

    shift from the code change: 0.0276  =  13.4 x the seed standard deviation

So a change that is mathematically identical and verified to rel 8.9e-7 moves
the result by thirteen times the run-to-run spread it has to be compared
against. Archiving `runs_conv_tokenizer/` and re-running was not caution, it was
required.

It also means the seed spread itself is dataset-specific and has to be measured
per dataset, not assumed: ISRUCs is sd 0.0021, TUEVs is sd 0.0235, eleven
times larger. The difference is checkpoint selection — 57% of TUEV cells peak at
their first evaluation and 0% of ISRUC cells do — not the model.
