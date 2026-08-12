"""TFM-Tokenizer for group B: official code, official weights, official recipe.

TFM matters more than the other group-B repos: the xlsx's group-A published
anchors come from its paper, so its pipeline is the one those calibration
numbers were produced with.

Architecture comes from the vendored repo (``vendor/tfm/models/tfm_token.py``)
through its own factory functions, and the weights are the ones shipped in
``pretrained_weigths/``:

    tokenizer = get_tfm_tokenizer_2x2x8(code_book_size=8192, emb_size=64)
    encoder   = get_tfm_token_classifier_64x4(n_classes=..., code_book_size=8192,
                                              emb_size=64)

Unlike the other models this is a **two-stage** pipeline. The tokenizer turns a
single-channel STFT into discrete token ids and is frozen; the transformer
encoder is what fine-tunes on those ids:

    x_stft = get_stft_torch(x, resampling_rate=200)
    _, tokens, _ = tokenizer.tokenize(x_stft_flat, x_temporal_flat)
    logits = encoder(tokens, num_ch=C)

The tokenizer stays in eval mode with gradients off, matching upstream, where it
is a fixed vocabulary rather than something being trained.

``n_freq=100`` in the tokenizer factory is the STFT half-bandwidth at 200 Hz,
which is a third independent confirmation that the model input rate is 200 and
not the 256 that dataset_configs.yaml lists as the corpora's source rate.

Weight layout in ``pretrained_weigths/``:

    single_dataset_settings/<DS>_tfm_tokenizer_2x2x8/tfm_tokenizer_last.pth
    single_dataset_settings/<DS>_tfm_tokenizer_2x2x8/tfm_encoder_best_model.pth
    multiple_dataset_settings/Pretrained_tfm_tokenizer_2x2x8/...   (the 4-corpus row)

The single-dataset weights correspond to the xlsx's "TFM-Tokenizer" row and the
multiple-dataset ones to "TFM-Tokenizer† (4 语料)".
"""

from __future__ import annotations

import os
import sys

import torch
import torch.nn as nn

VENDOR = "/work1/chenyuyou/yifanwang/Zhizhe/paclock-bench/vendor/tfm"
WEIGHTS = os.path.join(VENDOR, "pretrained_weigths")

CODE_BOOK_SIZE = 8192
EMB_SIZE = 64
RESAMPLING_RATE = 200      # --resampling_rate default; get_stft_torch is built on it

# dataset -> the directory name TFM uses for its single-corpus weights
SINGLE_DIRS = {
    "tuab": "TUAB_tfm_tokenizer_2x2x8",
    "tuev": "TUEV_tfm_tokenizer_2x2x8",
    "chbmit": "CHBMIT_tfm_tokenizer_2x2x8",
}


def get_stft_torch(X: torch.Tensor, resampling_rate: int = RESAMPLING_RATE):
    """STFT front end, inlined verbatim from TFM's utils/utils.py.

    Copied rather than imported because that module imports pyhealth at the top
    for its metrics, which we do not use -- our metrics module implements the
    protocol's own definitions, including the trapezoidal PR-AUC. The function
    itself depends only on torch and einops.

    Note ``center=False`` and the ``[:, :resampling_rate//2, :]`` crop: both are
    upstream's and both change the output shape, so neither may be "tidied".
    """
    from einops import rearrange                       # noqa: PLC0415

    B, C, _T = X.shape
    x_temp = rearrange(X, "B C T -> (B C) T")
    window = torch.hann_window(resampling_rate).to(x_temp.device)
    x_stft = torch.abs(
        torch.stft(x_temp, n_fft=resampling_rate,
                   hop_length=resampling_rate // 2,
                   onesided=True, return_complex=True, center=False,
                   window=window)[:, : resampling_rate // 2, :]
    )
    return rearrange(x_stft, "(B C) F T -> B C F T", B=B)


def _import_tfm():
    if VENDOR not in sys.path:
        sys.path.insert(0, VENDOR)
    from models.tfm_token import (                     # noqa: PLC0415
        get_tfm_token_classifier_64x4, get_tfm_tokenizer_2x2x8,
    )
    return get_tfm_tokenizer_2x2x8, get_tfm_token_classifier_64x4, get_stft_torch


class TFMClassifier(nn.Module):
    """Frozen tokenizer + fine-tuned transformer encoder, as upstream runs it.

    The STFT, the flatten-over-channels and the tokenize step all live here so
    the shared training loop keeps handing every model a plain (B, C, T) tensor.
    """

    MAX_CHANNELS = 16          # channel_embed in the released encoder is (16, 64)

    def __init__(self, tokenizer: nn.Module, encoder: nn.Module,
                 stft_fn, resampling_rate: int = RESAMPLING_RATE,
                 n_data_channels: int | None = None):
        super().__init__()
        # The released encoder carries a 16-entry channel embedding, so corpora
        # with more electrodes (PhysioNet-MI 64, FACED 32, BCI-IV-2a 22) get the
        # same 1x1 max-norm projection EEGPT's scripts use to reconcile BIOT
        # with a foreign montage. Corpora with fewer channels pass through.
        self.chan_conv = None
        if n_data_channels is not None and n_data_channels > self.MAX_CHANNELS:
            from .channel_adapt import Conv1dWithConstraint    # noqa: PLC0415
            self.chan_conv = Conv1dWithConstraint(
                n_data_channels, self.MAX_CHANNELS, 1, max_norm=1)
        self.tokenizer = tokenizer
        self.encoder = encoder
        self._stft = stft_fn
        self.resampling_rate = resampling_rate
        # upstream keeps the tokenizer fixed: it is a learned vocabulary, and
        # fine-tuning it would change the token ids the encoder was trained on
        self.tokenizer.eval()
        for p in self.tokenizer.parameters():
            p.requires_grad_(False)

    def train(self, mode: bool = True):
        super().train(mode)
        self.tokenizer.eval()          # never leaves eval, even under model.train()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        from einops import rearrange                   # noqa: PLC0415

        if self.chan_conv is not None:
            x = self.chan_conv(x)
        B, C, _T = x.shape
        x_stft = self._stft(x, resampling_rate=self.resampling_rate)
        x_stft = rearrange(x_stft, "B C F T -> (B C) F T")
        x_flat = rearrange(x, "B C T -> (B C) T")
        with torch.no_grad():
            _, tokens, _ = self.tokenizer.tokenize(x_stft, x_flat)
        tokens = rearrange(tokens, "(B C) T -> B C T", C=C)
        return self.encoder(tokens, num_ch=C)

    def backbone_parameters(self):
        return self.encoder.parameters()

    def head_parameters(self):
        if self.chan_conv is None:
            return self.encoder.parameters()
        import itertools                                   # noqa: PLC0415
        return itertools.chain(self.encoder.parameters(), self.chan_conv.parameters())


def build_tfm(n_classes: int, dataset: str, *, pretrained: bool = True,
              setting: str = "single", n_channels: int | None = None) -> nn.Module:
    """Build TFM-Tokenizer.

    ``setting="single"`` uses the per-corpus weights (the xlsx's "TFM-Tokenizer"
    row); ``"multiple"`` uses the 4-corpus ones ("TFM-Tokenizer† (4 语料)").
    ``pretrained=False`` gives the group-C from-scratch variant.
    """
    mk_tok, mk_enc, stft_fn = _import_tfm()
    tokenizer = mk_tok(code_book_size=CODE_BOOK_SIZE, emb_size=EMB_SIZE)
    encoder = mk_enc(n_classes=n_classes, code_book_size=CODE_BOOK_SIZE,
                     emb_size=EMB_SIZE)

    if pretrained:
        tok_path, enc_path = weight_paths(dataset, setting)
        # The tokenizer must always be the pretrained one: its codebook defines
        # the token ids, and a random codebook makes the encoder weights
        # meaningless. Only the encoder is optionally fresh.
        tok_state = torch.load(tok_path, map_location="cpu", weights_only=False)
        # The single-corpus tokenizers were saved with a one-Linear decoder while
        # the current code builds a two-layer Sequential (decoder.0 / decoder.2).
        # That head only exists for the tokenizer's own reconstruction pretraining
        # -- `tokenize()` runs the patch embeddings, the transformers and the
        # quantiser, and never touches it -- so the difference cannot change a
        # single token id. The multi-corpus tokenizer matches all 191 keys.
        missing, unexpected = tokenizer.load_state_dict(tok_state, strict=False)
        bad = [k for k in list(missing) + list(unexpected)
               if not k.startswith("decoder")]
        if bad:
            raise RuntimeError(f"TFM tokenizer weights mismatch: {bad[:5]}")
        if missing or unexpected:
            print(f"  TFM tokenizer: decoder head not loaded "
                  f"({len(missing)} missing, {len(unexpected)} unused); "
                  f"unused by tokenize()", flush=True)
        if enc_path is not None and os.path.exists(enc_path):
            enc_state = torch.load(enc_path, map_location="cpu", weights_only=False)
            # The released heads are not our classifier: the single-corpus files
            # carry a 1-logit binary head, and the multi-corpus MTP file carries
            # an 8192-way head that predicts codebook indices (its pretraining
            # objective). Both are replaced by a fresh head for the downstream
            # task, exactly as upstream does when fine-tuning, so they are
            # dropped here rather than shape-matched.
            enc_state = {k: v for k, v in enc_state.items()
                         if not k.startswith("classification_head")}
            missing, unexpected = encoder.load_state_dict(enc_state, strict=False)
            bad = [k for k in list(missing) + list(unexpected)
                   if not k.startswith(("classification_head", "classifier"))]
            if bad:
                raise RuntimeError(f"TFM encoder weights mismatch: {bad[:5]}")
            n_loaded = len(enc_state)
            print(f"  TFM encoder: loaded {n_loaded} tensors, "
                  f"classification head re-initialised", flush=True)

    return TFMClassifier(tokenizer, encoder, stft_fn,
                         n_data_channels=n_channels)


def weight_paths(dataset: str, setting: str = "single") -> tuple[str, str | None]:
    """Locate the tokenizer and encoder checkpoints for a dataset."""
    if setting == "single":
        d = SINGLE_DIRS.get(dataset)
        if d is None:
            raise KeyError(
                f"no single-corpus TFM weights for {dataset!r}; "
                f"available: {sorted(SINGLE_DIRS)}")
        base = os.path.join(WEIGHTS, "single_dataset_settings", d)
        tok = os.path.join(base, "tfm_tokenizer_last.pth")
        enc = os.path.join(base, "tfm_encoder_best_model.pth")
    else:
        base = os.path.join(WEIGHTS, "multiple_dataset_settings")
        tok = os.path.join(base, "Pretrained_tfm_tokenizer_2x2x8",
                           "tfm_tokenizer_last.pth")
        enc = os.path.join(base, "MTP_Pretrained_tfm_encoder_64x4",
                           "tfm_encoder_mtp_last.pth")
    if not os.path.exists(tok):
        raise FileNotFoundError(tok)
    return tok, enc


def count_params(model: nn.Module) -> float:
    return sum(p.numel() for p in model.parameters()) / 1e6
