# Superseded: wrong training recipe

Trained with AdamW / lr=1e-4 / cosine / batch 64, which is not how the published
group-A numbers were produced. BIOT uses plain Adam, lr=1e-3, no scheduler,
batch 512. The low LR left SPaRCNet under-trained on TUEV (kappa 0.276 vs a
published 0.423). Re-run with the official recipe; kept only as a record.

Superseded 2026-08-04.
