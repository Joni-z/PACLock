# 作废:缺少 layer-wise LR decay 与 warmup

LaBraM 的官方 recipe 含 --layer_decay 0.65 与 --warmup_epochs 5,但当时训练
循环尚未实现这两项,等于用了一个不同的配方。TUEV kappa 0.4130 vs 发表 0.5067。
实现后重跑。作废于 2026-08-06。
