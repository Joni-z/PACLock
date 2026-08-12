"""Every filesystem location the repo needs, resolved in one place.

Before this module the cluster paths were written out longhand in 21 Python
files, 297 configs and 5 Slurm scripts, which made the repo unmovable. Nothing
about the defaults changes: on the machine this was built on, every value below
resolves to exactly the string it replaced, so jobs already queued keep working.

To move the repo, set the two environment variables and copy `vendor/`:

    export PACLOCK_DATA=/new/path/to/raw            # raw corpora, read-only
    export PACLOCK_PROC=/new/path/to/processed_root # the processed_* parents

`REPO` needs no variable: it is derived from this file's own location, so the
vendored baselines are found wherever the checkout lands.

Configs refer to data as ``$PACLOCK_PROC/processed/tuev`` and are expanded at
load time by ``expand``, so a config never has to name a cluster.
"""

from __future__ import annotations

import os

# The checkout itself. Derived, never configured: paths.py is
# <REPO>/paclock_bench/paths.py.
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Where the raw corpora live (TUH, CHB-MIT, ISRUC, ...). Read-only, shared.
DATA = os.environ.get("PACLOCK_DATA", "/work1/chenyuyou/yifanwang/data")

# Parent of processed/, processed_pac/, processed_biot/, processed_labram/.
PROC_ROOT = os.environ.get("PACLOCK_PROC", "/work1/chenyuyou/yifanwang/Zhizhe")

VENDOR = os.path.join(REPO, "vendor")
RUNS = os.path.join(REPO, "runs")
RESULTS = os.path.join(REPO, "results")


def processed(protocol: str = "processed", dataset: str = "") -> str:
    """Path to a preprocessed corpus.

    ``protocol`` is one of processed / processed_pac / processed_biot /
    processed_labram / processed_tfm -- the frozen protocol and the per-model
    ones hard rule 2 requires.
    """
    p = os.path.join(PROC_ROOT, protocol)
    return os.path.join(p, dataset) if dataset else p


def vendored(name: str) -> str:
    """Path to a vendored upstream repo (biot, labram, cbramod, eegpt, tfm)."""
    return os.path.join(VENDOR, name)


def expand(path: str) -> str:
    """Expand $PACLOCK_* (and ~) in a config path.

    Applied to every ``data_root`` and ``checkpoint`` as it is read, so configs
    can be written portably while absolute paths still work unchanged -- an
    absolute path contains nothing to expand and comes back untouched.
    """
    if not path:
        return path
    os.environ.setdefault("PACLOCK_DATA", DATA)
    os.environ.setdefault("PACLOCK_PROC", PROC_ROOT)
    return os.path.expanduser(os.path.expandvars(path))
