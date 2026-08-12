# Superseded TUEV runs

Built on TUEV preprocessing that DROPPED events too close to a recording edge.
The reference implementation wraps instead (tiles the signal 3x), so we were
missing 5.4% of the corpus (106,394 vs the 112,491 the protocol cites) and the
dropped events were not a random subset. Preprocessing fixed to wrap; TUEV
re-run. Kept only as a record.

Superseded 2026-08-05.
