"""config -> assembled PACLock model.

Two architectures, selected by ``cfg['arch']``:

  * "flat" (default, v1): Frontend (band tokens, channels collapsed) -> Encoder
    (single swappable token mixer) -> Head. The original mixer-swap ablation.
  * "triaxial" (v2, AGENT.md sec. 13): TriAxialFrontend (electrode x band x
    time-patch GRID, channels kept) + physics positional encodings ->
    TriAxialEncoder (time/space/freq axis mixers, only the freq mixer is
    swapped: cfg['freq_mixer']) -> Head. The foundation-model backbone.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .frontend import Frontend
from .frontend.conv import ConvFrontend
from .frontend.triaxial import TriAxialFrontend
from .encoder import Encoder
from .triaxial import TriAxialEncoder, BandPE, SpatialPE
from .head import ClassificationHead
from .augment import RandomAugment
from .montage import coords_for


def _spatial_coords(cfg: dict):
    """xyz electrode coords for SpatialPE when cfg opts in, else None (learned
    index embedding). `spatial_pe: xyz` uses the dataset's montage coordinates
    (models/montage.py); anything else keeps the original index embedding so
    existing configs are unchanged (AGENT.md sec. 13.23 A)."""
    if cfg.get("spatial_pe", "index") != "xyz":
        return None
    coords = coords_for(cfg.get("dataset"))
    if coords is not None and coords.shape[0] != cfg["n_channels"]:
        raise ValueError(
            f"spatial_pe=xyz: montage for {cfg.get('dataset')!r} has "
            f"{coords.shape[0]} channels but cfg n_channels={cfg['n_channels']}"
        )
    return coords


class PACLock(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        d = cfg["d_model"]
        self.augment = RandomAugment(cfg.get("augmentations", []))
        if cfg.get("frontend", "sinc") == "conv":
            self.frontend = ConvFrontend(
                n_channels=cfg["n_channels"], hidden_dim=d,
                patch_len=cfg.get("patch_len", 100),
            )
        else:
            self.frontend = Frontend(
                n_bands=cfg["n_bands"], hidden_dim=d, seq_len=cfg["seq_len"],
                sample_rate=cfg["sample_rate"], kernel_size=cfg.get("kernel_size", 101),
                n_channels=cfg["n_channels"], patch_len=cfg.get("patch_len", 200),
            )
        self.encoder = Encoder(
            depth=cfg["depth"], d_model=d, mixer=cfg["mixer"],
            dropout=cfg.get("dropout", 0.1), **cfg.get("mixer_kwargs", {}),
        )
        self.head = ClassificationHead(d, cfg["num_classes"])

    def forward(self, x: torch.Tensor, phase_mode: str = "normal") -> torch.Tensor:
        x = self.augment(x)
        token, phase_unit, amplitude = self.frontend(x)
        h = self.encoder(token, phase_unit=phase_unit, amplitude=amplitude)
        return self.head(h)


class TriAxialPACLock(nn.Module):
    """v2 foundation-model backbone (AGENT.md sec. 13)."""

    def __init__(self, cfg: dict):
        super().__init__()
        d = cfg["d_model"]
        self.freq_mixer = cfg.get("freq_mixer", "coupling")
        self.augment = RandomAugment(cfg.get("augmentations", []))
        # H2 (2026-08-20): learned montage -- a per-corpus channel-mixing
        # matrix W = I + Delta applied to the raw input, Delta ZERO-INIT so
        # the init is exactly the identity (the current model). This is the
        # learnable form of re-referencing / CSP-style spatial filtering,
        # which every strong MI baseline performs early and this model never
        # did: electrodes stayed separate until the encoder spatial attention.
        # Corpus-specific by construction, like spatial_pe and the head -- it
        # is NOT part of the transferable backbone, so the pretraining
        # contract is unchanged (load_pretrained_backbone already excludes
        # everything outside frontend./band_pe./encoder., and this lives
        # outside the frontend).
        self.learned_montage = bool(cfg.get("learned_montage", False))
        if self.learned_montage:
            self.montage_delta = nn.Parameter(
                torch.zeros(cfg["n_channels"], cfg["n_channels"]))
        self.frontend = TriAxialFrontend(
            n_bands=cfg["n_bands"], hidden_dim=d, sample_rate=cfg["sample_rate"],
            kernel_size=cfg.get("kernel_size", 201), patch_len=cfg.get("patch_len", 200),
            pac_patch_len=cfg.get("pac_patch_len"),
            return_pac_vector=self.freq_mixer == "phase",
            tokenizer_mode=cfg.get("tokenizer_mode", "raw"),
            pac_token_mode=cfg.get("pac_token_mode", "measured"),
            interaction_mode=cfg.get("interaction_mode", "product"),
            hybrid_gate=cfg.get("hybrid_gate", "none"),
            fusion_mode=cfg.get("fusion_mode", "blend"),
            raw_stem=cfg.get("raw_stem", "linear"),
            coupling_strength=cfg.get("coupling_strength", False),
        )
        if self.frontend.tokenizer_mode in ("hybrid", "duplex"):
            # The coupling/phase mixers consume an (nb, nb) coupling matrix and
            # would need it lifted to the 2*nb hybrid grid; nothing defines that
            # lift yet, and the deliverable uses attention anyway. Refuse rather
            # than mis-index.
            if self.freq_mixer not in ("attention", "none"):
                raise ValueError(
                    "tokenizer_mode=hybrid/duplex requires freq_mixer=attention or none, "
                    f"got {self.freq_mixer!r}"
                )
            # aux_recon / masked-amplitude pretraining IS supported on these
            # grids: crossfreq_aux_loss masks a band's fused/raw and interaction
            # rows together (paired-row masking), so the visible interaction row
            # cannot hand the reconstruction its own target.
        # BandPE(index) and the band/spatial heads are sized from the GRID's
        # frequency axis, which is 2*n_bands under hybrid -- the frontend owns
        # that fact, so read it rather than re-deriving it here.
        grid_bands = self.frontend.n_token_bands
        self.band_pe = BandPE(d, n_bands=grid_bands, mode=cfg.get("band_pe", "hz"))
        self.spatial_pe = SpatialPE(cfg["n_channels"], d, coords=_spatial_coords(cfg))
        self.encoder = TriAxialEncoder(
            depth=cfg["depth"], d_model=d,
            freq_mixer=self.freq_mixer,
            n_heads=cfg.get("n_heads", 4), dropout=cfg.get("dropout", 0.1),
            space_over_bands=cfg.get("space_over_bands", False),
            # Only FreqMITopology reads this; every other mixer swallows it via **_,
            # so passing it unconditionally leaves existing configs bit-identical.
            mi_k=cfg.get("mi_k", 3),
        )
        self.head = ClassificationHead(d, cfg["num_classes"],
                                      mode=cfg.get("head", "mean"),
                                      n_bands=grid_bands,
                                      n_channels=cfg["n_channels"],
                                      n_patches=cfg["seq_len"] // cfg.get("patch_len", 200))

        # Optional crossfreq-reconstruction auxiliary head (AGENT.md sec. 13.15).
        # When aux_recon_weight > 0, supervised training adds a masked-amplitude
        # reconstruction loss (mask the high-band half, rebuild from visible low
        # bands) alongside the classification loss. Purpose: give the objective an
        # incentive to keep low->high coupling info that plain supervised CE has no
        # reason to preserve (sec. 9.17 Finding 1), and watch whether it stops
        # pac_scale from collapsing. Zero/absent weight => model behaves exactly as
        # before (no new params allocated, no new forward path taken).
        self.aux_recon_weight = cfg.get("aux_recon_weight", 0.0)
        if self.aux_recon_weight > 0:
            self.aux_mask_mode = cfg.get("aux_mask_mode", "crossfreq")
            self.aux_mask_ratio = cfg.get("aux_mask_ratio", 0.5)
            self.mask_token = nn.Parameter(torch.zeros(d))
            self.recon = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))
            # What the masked cells are asked to predict:
            #   amp           -- log mean amplitude, raw scale (the original)
            #   band_norm     -- same target, standardized PER BAND: an l2 loss
            #                    on raw log-amplitudes is dominated by the
            #                    high-variance slow bands (the 1/f aperiodic
            #                    component), which starves the fast oscillatory
            #                    structure downstream tasks need and biases the
            #                    representation toward subject identity
            #                    (arXiv:2605.26434; FAME arXiv:2608.01898)
            #   band_norm_pac -- band_norm PLUS the masked band's coupling
            #                    column: for masked amplitude-band j, predict
            #                    the complex PAC entries Z[i, j] against every
            #                    VISIBLE phase-band i -- the pretraining task
            #                    becomes the tokenizer's native quantity
            self.aux_target = cfg.get("aux_target", "amp")
            if self.aux_target not in ("amp", "band_norm", "band_norm_pac"):
                raise ValueError(f"aux_target must be amp/band_norm/"
                                 f"band_norm_pac, got {self.aux_target!r}")
            self.aux_pac_weight = cfg.get("aux_pac_weight", 1.0)
            if self.aux_target == "band_norm_pac":
                self.recon_pac = nn.Sequential(
                    nn.Linear(d, d), nn.GELU(),
                    nn.Linear(d, 2 * cfg["n_bands"]))

    def crossfreq_aux_loss(self, x: torch.Tensor) -> torch.Tensor:
        """Masked band-amplitude reconstruction as a supervised auxiliary.

        Mirrors models/pretrain.py's MAE crossfreq path (mask token + PEs +
        visible-visible coupling leakage control), but shares the classifier's
        frontend/encoder so the auxiliary gradient shapes the same representation
        the classifier uses. No augmentation here -- the reconstruction target
        must stay clean. Only called during training when aux_recon_weight > 0.
        """
        want_pac = getattr(self, "aux_target", "amp") == "band_norm_pac"
        restore = False
        if want_pac and not self.frontend.return_pac_vector:
            self.frontend.return_pac_vector = True
            restore = True
        frontend_out = self.frontend(x, return_amp_target=True)
        if restore:
            self.frontend.return_pac_vector = False
        pac_target_src = None
        if len(frontend_out) == 5:
            tokens, coupling, band_hz, amp_target, pv = frontend_out
            pac_target_src = pv if want_pac else None
            # the encoder's phase mixer is the only consumer of the complex
            # vector; every other mixer historically received None
            pac_vector = pv if self.freq_mixer == "phase" else None
        else:
            tokens, coupling, band_hz, amp_target = frontend_out
            pac_vector = None
        B, C, nb, P, D = tokens.shape
        # hybrid/duplex grids carry 2*nb_phys rows (fused/raw rows first,
        # interaction rows after -- frontend row-order contract), both derived
        # from the same physical band, and amp_target is per PHYSICAL band. The
        # mask is therefore drawn over physical bands and applied to BOTH of a
        # band's rows -- a visible interaction row would otherwise hand the
        # reconstruction its own target. Residual second-order path: a visible
        # band's interaction row mixes MASKED bands' unit phases via the PAC
        # vectors; phases are amplitude-normalised, so the amplitude target
        # does not travel that path (only per-window coupling statistics do).
        nb_phys = amp_target.shape[2]
        paired = nb == 2 * nb_phys

        if self.aux_mask_mode == "crossfreq":
            pmask = torch.zeros(B, C, nb_phys, P, dtype=torch.bool, device=x.device)
            pmask[:, :, nb_phys // 2:, :] = True            # hide high-frequency half
        else:                                               # "random"
            pmask = torch.rand(B, C, nb_phys, P, device=x.device) < self.aux_mask_ratio
        mask = torch.cat([pmask, pmask], dim=2) if paired else pmask

        tok = torch.where(mask.unsqueeze(-1), self.mask_token.view(1, 1, 1, 1, D), tokens)
        tok = tok + self.band_pe(band_hz).view(1, 1, nb, 1, D)
        tok = tok + self.spatial_pe(C, x.device).view(1, C, 1, 1, D)

        # leakage control: keep coupling only between bands BOTH visible at each
        # (channel, patch); zero every entry touching a masked band (else the
        # reconstruction target leaks through the coupling matrix). coupling is
        # (.., nb_phys, nb_phys) in every mode -- single-scale by contract.
        vis = (~pmask).permute(0, 1, 3, 2)                  # (B,C,P,nb_phys)
        keep = (vis.unsqueeze(-1) & vis.unsqueeze(-2)).to(coupling.dtype)
        cpl = coupling * keep
        pac = None if pac_vector is None else pac_vector * keep
        h = self.encoder(tok, cpl, pac)

        # each masked band's amplitude is predicted from its fused/raw row --
        # the first nb_phys grid rows in every mode.
        pred = self.recon(h[:, :, :nb_phys]).squeeze(-1)    # (B,C,nb_phys,P)
        target = amp_target.detach()
        if getattr(self, "aux_target", "amp") != "amp":
            # per-band standardization with batch statistics -- equal reward
            # for every band's structure instead of an l2 race the slow bands
            # always win (see aux_target comment in __init__)
            mu = target.mean(dim=(0, 1, 3), keepdim=True)
            sd = target.std(dim=(0, 1, 3), keepdim=True).clamp_min(1e-6)
            target = (target - mu) / sd
        loss = F.mse_loss(pred[pmask], target[pmask])
        if pac_target_src is not None:
            # Z is (B, C, P, i, j) complex, i = phase band, j = amplitude band
            # (patch_pac_vector's contract). Supervise entry (i, j) only where
            # band j is MASKED and band i is VISIBLE: a masked phase band
            # makes the entry unpredictable from the input -- noise, not
            # leakage -- so it is excluded rather than averaged in.
            Z = pac_target_src.detach()
            tgt = torch.stack([Z.real, Z.imag], dim=-1)      # (B,C,P,i,j,2)
            tgt = tgt.permute(0, 1, 4, 2, 3, 5)              # (B,C,j,P,i,2)
            pp = self.recon_pac(h[:, :, :nb_phys])           # (B,C,j,P,2nb)
            pp = pp.reshape(*pp.shape[:-1], nb_phys, 2)      # (B,C,j,P,i,2)
            w = (pmask.unsqueeze(-1)
                 & (~pmask).permute(0, 1, 3, 2).unsqueeze(2))  # (B,C,j,P,i)
            w = w.unsqueeze(-1).to(pp.dtype)
            denom = w.sum().clamp_min(1.0)
            loss = loss + self.aux_pac_weight * (
                ((pp - tgt) ** 2) * w).sum() / denom
        return loss

    def forward(self, x: torch.Tensor, phase_mode: str = "normal") -> torch.Tensor:
        x = self.augment(x)
        if self.learned_montage:
            eye = torch.eye(x.shape[1], device=x.device, dtype=x.dtype)
            x = torch.einsum("vc,bct->bvt", eye + self.montage_delta, x)
        frontend_out = self.frontend(x)
        if self.freq_mixer == "phase":
            tokens, coupling, band_hz, pac_vector = frontend_out
            if phase_mode == "magnitude":
                # Preserve every PAC edge magnitude but remove preferred phase.
                pac_vector = torch.complex(pac_vector.abs(), torch.zeros_like(pac_vector.real))
            elif phase_mode == "scramble":
                # Preserve magnitude exactly while independently randomising the
                # measured preferred phase. This is the decisive mechanism test.
                theta = 2.0 * torch.pi * torch.rand_like(pac_vector.real)
                pac_vector = pac_vector * torch.complex(theta.cos(), theta.sin())
            elif phase_mode != "normal":
                raise ValueError(f"unknown phase_mode={phase_mode!r}")
        else:
            tokens, coupling, band_hz = frontend_out
            pac_vector = None
        B, C, nb, P, D = tokens.shape
        # physics positional encodings: band by center-freq, electrode by position
        tokens = tokens + self.band_pe(band_hz).view(1, 1, nb, 1, D)
        tokens = tokens + self.spatial_pe(C, tokens.device).view(1, C, 1, 1, D)
        h = self.encoder(tokens, coupling, pac_vector)   # (B,C,nb,P,D)
        # the grid shape travels with the tokens so a readout can use the axes
        return self.head(h.reshape(B, C * nb * P, D), (C, nb, P))


def build_model(cfg: dict) -> nn.Module:
    if cfg.get("arch", "flat") == "triaxial":
        return TriAxialPACLock(cfg)
    return PACLock(cfg)


# Prefixes carried over from a training/pretrain.py checkpoint into a
# downstream TriAxialPACLock. spatial_pe is deliberately excluded: it was
# sized to the pretraining pool's max channel count and indexes electrodes by
# raw position, which is not a shared identity across corpora (channel 0 in
# TUAB is not channel 0 in FACED), so transferring it would silently paste
# one corpus's electrode geometry onto another's. head/mask_token/recon are
# task-specific (classification head) or pretraining-only (aux reconstruction
# head) and are always reinitialized fresh.
_BACKBONE_PREFIXES = ("frontend.", "band_pe.", "encoder.")


def load_pretrained_backbone(model: nn.Module, checkpoint_path: str,
                             exclude: tuple = ()) -> dict:
    """Load frontend/band_pe/encoder weights from a training/pretrain.py
    checkpoint into `model` in place. Returns
    {"loaded": [...], "skipped_shape": [...], "skipped_excluded": [...]}
    for the caller to log -- a shape mismatch (e.g. n_bands or d_model differs
    from the pretraining config) is silently dropped per-key rather than
    failing the whole load, since the rest of the backbone may still transfer.

    `exclude` is a tuple of key prefixes to leave at their fresh initialisation.
    It exists for one measurement: pretraining runs at patch_len=200 while most
    finetuning configs run at 50, so the tokenizer's Conv1d kernel does not match
    and its weights are dropped by the shape check -- meaning "pretrained" has so
    far meant "pretrained encoder, tokenizer relearned from scratch" on those
    corpora. Matching patch_len fixes that, but it also changes the token grid and
    the PAC estimation window, which docs/FINDINGS.md records as the single
    largest architectural factor. Excluding the tokenizer explicitly gives a third
    arm at the SAME patch_len, so the pretrained tokenizer's contribution can be
    read off without the resolution moving underneath it."""
    # weights_only=False: corrected-schedule checkpoints carry optimizer and
    # scheduler state for --resume, whose numpy scalars PyTorch>=2.6 refuses
    # under the weights_only default. These are our own files.
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    src = ckpt["model"] if "model" in ckpt else ckpt
    dst = model.state_dict()
    loaded, skipped, excluded_keys = [], [], []
    exclude = tuple(exclude or ())
    for k, v in src.items():
        if not k.startswith(_BACKBONE_PREFIXES):
            continue
        if exclude and k.startswith(exclude):
            excluded_keys.append(k)
            continue
        if k not in dst:
            skipped.append(k)
            continue
        if dst[k].shape != v.shape:
            skipped.append(f"{k} (ckpt {tuple(v.shape)} vs model {tuple(dst[k].shape)})")
            continue
        dst[k] = v
        loaded.append(k)
    model.load_state_dict(dst, strict=True)
    # Remembered on the model so the finetune recipe can treat loaded and
    # fresh tensors differently (staged unfreeze, per-depth LR).
    model._loaded_keys = set(loaded)
    return {"loaded": loaded, "skipped_shape": skipped,
            "skipped_excluded": excluded_keys}
