# One-off diagnostics

Written to answer a single question and kept only as provenance for the
numbers they produced. They hardcode the cluster paths they were run on and
are **not** expected to work after the move -- nothing imports them, and
`scripts/` proper is portable through `paclock_bench/paths.py`.

Files: `_bstat.py`, `_chkpac.py`, `_collect_b.py`, `_dump.py`, `_rows.py`, `_scale.py`
