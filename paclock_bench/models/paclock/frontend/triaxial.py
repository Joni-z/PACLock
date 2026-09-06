"""v2 tri-axial frontend: raw EEG -> (electrode x band x time-patch) token GRID.

Unlike the v1 frontend (models/frontend/__init__.py), this one does NOT collapse
the channel axis -- electrodes stay an explicit token dimension so the encoder
can model space and so variable montages are possible (AGENT.md sec. 13.3).

Outputs, for x = (B, C, T):
  * tokens   : (B, C, n_bands, P, d_model)   -- the 3D token grid
  * coupling : (B, C, P, n_bands, n_bands)   -- time-resolved, per-channel MVL
               coupling (AGENT.md sec. 13.6 / 9.17 Finding 2: computed WITHIN
               each patch and per channel, never averaged over time/channels)
  * band_hz  : (n_bands, 2)  center-freq + bandwidth per band, for the band PE

The analytic-signal math (unit complex phase vector, mean-centred amplitude
debiasing, no atan2) is identical to v1 and still validated by
scripts/synth_pac_test.py; only the reduction axes change.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .sinc import SincBandpass
from .analytic import hilbert, phase_amplitude

# Fixed divisor for the MVL normalisation (same rationale as v1's 4D path:
# dividing by a per-patch amplitude std blows up to NaN on flat/dead channels).
NORM_CONST = 100.0


def patch_pac_vector(phase_unit, amplitude, P, normalize=True):
    """Complex, time-resolved directional PAC vector per channel and patch.

    phase_unit, amplitude: (B, C, n_bands, T), T divisible by P.
    Returns complex ``Z`` with shape (B, C, P, n_bands, n_bands), where
    ``Z[..., i, j]`` is low-band-i phase driving band-j amplitude.  Keeping Z
    complex preserves the preferred PAC phase; taking ``abs`` too early was
    exactly what prevented the old mixer from defining a phase geometry.
    """
    B, C, nb, T = phase_unit.shape
    L = T // P
    ph = phase_unit[..., : P * L].reshape(B, C, nb, P, L)
    am = amplitude[..., : P * L].reshape(B, C, nb, P, L)
    am = am - am.mean(dim=-1, keepdim=True)                      # dPAC debiasing
    # Z[b,c,p,i,j] = mean_t phase_i * amp_j   (within patch p)
    Z = torch.einsum("bcipl,bcjpl->bcpij", ph, am.to(ph.dtype)) / L
    if normalize:
        Z = Z / NORM_CONST
    return Z


def patch_coupling(phase_unit, amplitude, P, normalize=True):
    """Backward-compatible MVL magnitude used by the older mixers."""
    return patch_pac_vector(phase_unit, amplitude, P, normalize).abs()


def _patch_project(conv, x):
    """``conv(x)`` for a Conv1d with ``in_channels == 1`` and
    ``stride == kernel_size``, evaluated as a single GEMM.

    Every tokeniser in this frontend has that shape, and it is the worst shape a
    convolution library can be handed: with one input channel the implicit-GEMM
    reduction is only ``kernel_size`` long and the output-channel dimension is 8
    or 64, so MIOpen falls back to a near-naive kernel over 4096 independent
    single-channel signals.

    Measured on the finished runs rather than guessed: ``size_large`` multiplies
    the encoder FLOPs by 5.33 and costs 7% of wall time, which puts the whole
    encoder at 1.8% of a step -- the other 98% is here, sustaining roughly 0.05%
    of the MI210 fp32 peak. The same ladder shows ``patch400`` (4x FEWER tokens)
    running SLOWER than ``patch100``, which only makes sense if the cost tracks
    the convolution kernel size rather than the token count.

    With ``stride == kernel_size`` the patches do not overlap, so the
    convolution IS a per-patch linear map, and ``(N, P, patch) @ W`` is the same
    arithmetic as one well-tuned GEMM. Mathematically identical; NOT
    bit-identical, because the two reduce over ``patch`` in a different order.
    See scripts/verify_patch_project.py.

    Returns ``(N, P, out_channels)`` -- already the layout both call sites
    wanted, so the ``transpose(1, 2)`` they each did afterwards is gone too.
    """
    patch = conv.kernel_size[0]
    x = x.reshape(x.shape[0], -1)                        # (N, T); accepts (N,1,T)
    N, T = x.shape
    P = T // patch
    w = conv.weight.reshape(conv.out_channels, patch).t()         # (patch, K)
    out = x[:, : P * patch].reshape(N, P, patch) @ w              # (N, P, K)
    if conv.bias is not None:
        out = out + conv.bias
    return out


class TriAxialFrontend(nn.Module):
    def __init__(
        self,
        n_bands: int,
        hidden_dim: int,
        sample_rate: int,
        kernel_size: int = 201,
        patch_len: int = 200,
        pac_patch_len: int | None = None,
        normalize: bool = True,
        return_pac_vector: bool = False,
        tokenizer_mode: str = "raw",
        pac_token_mode: str = "measured",
        interaction_mode: str = "product",
        hybrid_gate: str = "none",
        fusion_mode: str = "blend",
        raw_stem: str = "linear",
        coupling_strength: bool = False,
        **_,
    ):
        super().__init__()
        self.n_bands = n_bands
        self.patch_len = patch_len
        # Window(s) the PAC statistic is estimated over; defaults to the token
        # patch so existing configs are unchanged.
        #
        # A LIST turns this multi-scale. The reason is measured, not aesthetic:
        # holding the token count fixed and moving only this window is worth
        # +0.0504 on TUEV and -0.0055 on ISRUC (docs/FINDINGS.md), i.e. the
        # best window is set by the timescale of the phenomenon -- 5 s windows
        # of epileptiform transients want a short one, 30 s sleep epochs want a
        # long one. A backbone that has to commit to one window before it knows
        # the downstream task is committing to the wrong one half the time, so
        # it estimates PAC at every scale and learns the mixture.
        if isinstance(pac_patch_len, (list, tuple)):
            self.pac_patch_lens = [int(v) for v in pac_patch_len]
            if not self.pac_patch_lens:
                raise ValueError("pac_patch_len list must not be empty")
        else:
            self.pac_patch_lens = [int(pac_patch_len or patch_len)]
        # single-scale name kept: error messages, and the coupling/pac_vector
        # outputs still report the first (shortest-indexed) scale
        self.pac_patch_len = self.pac_patch_lens[0]
        self.normalize = normalize
        self.return_pac_vector = return_pac_vector
        if tokenizer_mode not in ("raw", "pac_interaction", "hybrid", "fused", "duplex"):
            raise ValueError(
                "tokenizer_mode must be raw/pac_interaction/hybrid/fused/duplex, got "
                f"{tokenizer_mode!r}"
            )
        if fusion_mode not in ("blend", "gated"):
            raise ValueError(f"fusion_mode must be blend/gated, got {fusion_mode!r}")
        if pac_token_mode not in ("measured", "uniform", "scramble", "magnitude"):
            raise ValueError(
                "pac_token_mode must be measured/uniform/scramble/magnitude, got "
                f"{pac_token_mode!r}"
            )
        if interaction_mode not in ("product", "rotation", "concat"):
            raise ValueError(
                "interaction_mode must be product/rotation/concat, got "
                f"{interaction_mode!r}"
            )
        if hybrid_gate not in ("none", "band"):
            raise ValueError(f"hybrid_gate must be none/band, got {hybrid_gate!r}")
        if hybrid_gate != "none" and tokenizer_mode != "hybrid":
            raise ValueError("hybrid_gate requires tokenizer_mode=hybrid")
        self.tokenizer_mode = tokenizer_mode
        self.pac_token_mode = pac_token_mode
        self.interaction_mode = interaction_mode
        self.hybrid_gate = hybrid_gate
        if hybrid_gate == "band":
            # Per-band learned gate on the INTERACTION rows only. Initialised to
            # one, so at init the gated model is bit-identical to plain hybrid
            # (asserted by verify_hybrid.py). The frequency attention already
            # learns input-dependent mixing between raw and interaction rows;
            # what it cannot fix is head=mean averaging all 2*nb rows uniformly
            # at the readout, where useless interaction rows dilute the pooled
            # representation no matter what attention did upstream. The gate
            # lets the gradient shrink those rows globally -- and its trained
            # value is itself a measurement: alpha_j per corpus reads out how
            # much of band j's interaction the task actually used.
            self.interaction_gate = nn.Parameter(torch.ones(n_bands))
        self.sinc = SincBandpass(n_bands, sample_rate, kernel_size=kernel_size)
        # "hybrid" (2026-08-18): raw band tokens AND PAC interaction tokens,
        # side by side on the frequency axis -- grid (C, 2*nb, P) with rows
        # 0..nb-1 the raw tokens and nb..2nb-1 the interactions. The measured
        # reason: the interaction token REPLACES band j's own phase with a
        # coupling-weighted mixture of lower bands' phases, which wins where
        # cross-band structure is the class signal (TUEV, +0.20 with rotation)
        # and costs up to 0.12 where the discarded within-band phase carried it
        # (CHB-MIT, TUSZ, MI) -- every alternative explanation was measured
        # away first (amplitude readability probe, rotation, concat, uniform;
        # docs/FINDINGS.md). Side-by-side keeps both sources: nothing is
        # replaced, and the encoder's frequency attention decides per task what
        # to read. This retires the constitutive-only ("no free path") design
        # on the strength of that evidence -- the forced version is a bet that
        # loses on 8 of 9 corpora, and the bet was the doctrine, not the PAC.
        self.fusion_mode = fusion_mode
        # CF2 (2026-09-07): explicit coupling-STRENGTH feature on the fused rows.
        # Band j's column |Z_{i,j}| over slower bands i (n_bands values) is mapped
        # by a zero-initialised linear layer to d_model and added to the fused
        # row -- at init the model is unchanged; the magnitudes that the aligned
        # phase normalises away (Eq. rotation) re-enter here as content.
        self.coupling_strength = coupling_strength
        if coupling_strength:
            self.cs_proj = nn.Linear(n_bands, hidden_dim)
            nn.init.zeros_(self.cs_proj.weight); nn.init.zeros_(self.cs_proj.bias)
        if tokenizer_mode == "duplex":
            # duplex = fused rows PLUS separate interaction rows: grid
            # (C, 2*nb, P). Rows 0..nb-1 are fused-blend tokens (r + beta*h,
            # beta init 0 -- the raw worst case), rows nb..2nb-1 are the gated
            # interaction rows hybrid_gate uses (alpha init 1). At init this is
            # bit-identical to hybrid-with-gate (asserted), and if training
            # drives beta and alpha to 0 it degrades to raw plus zero rows,
            # which LayerNorm in the head renormalises away. The measured
            # motivation: TUEV needs the interaction as SEPARATE attendable
            # rows (fused loses 0.15 there), the seizure corpora profit from
            # in-row fusion (fusegate is their family best), and MI needs the
            # raw worst case -- duplex is the only grid holding all three.
            self.fusion_beta = nn.Parameter(torch.zeros(n_bands, hidden_dim))
            self.interaction_gate = nn.Parameter(torch.ones(n_bands))
        if tokenizer_mode == "fused":
            # In-row fusion of the two sources; the grid keeps raw's shape, so
            # the encoder, every head, and the pretraining mask are untouched.
            if fusion_mode == "blend":
                # zero-init: the model IS the raw model at step 0, and PAC
                # content enters only where the gradient earns it. The worst
                # case is raw by construction -- the guarantee hybrid's
                # side-by-side rows could not give on the MI corpora.
                self.fusion_beta = nn.Parameter(torch.zeros(n_bands, hidden_dim))
            else:
                self.fusion_gate = nn.Linear(2 * hidden_dim, hidden_dim)
                # bias +2: sigmoid ~ 0.88, training starts mostly-raw
                nn.init.constant_(self.fusion_gate.bias, 2.0)
        if raw_stem not in ("linear", "deep"):
            raise ValueError(f"raw_stem must be linear/deep, got {raw_stem!r}")
        self.raw_stem = raw_stem
        if tokenizer_mode in ("raw", "hybrid", "fused", "duplex"):
            # Per-(channel, band) raw-waveform patch tokenizer. Shared across
            # all channel/band pairs; retained as the exact legacy baseline.
            self.tokenizer = nn.Conv1d(
                1, hidden_dim, kernel_size=patch_len, stride=patch_len
            )
            if raw_stem == "deep":
                # H1 (2026-08-20): every baseline that beats this model on the
                # motor-imagery corpora enters the signal through a DEEP
                # NONLINEAR conv stem (SPaRCNet: DenseNet-1D; CBraMod: stacked
                # Conv2d+GN+GELU; ContraWR: ResNet over spectrograms), while
                # this frontend was the only one in the suite entering through
                # a single linear map. Where the discriminative feature is a
                # constructed physical quantity (bands, coupling) the linear
                # map suffices and we win; where it must be dug out of the
                # waveform (ERD trajectories, MI/emotion) it starves.
                #
                # The stem refines each patch AFTER the linear projection as a
                # residual: token = linear_patch + refine(patch_waveform),
                # with the refiner's LAST layer ZERO-INIT -- at init the
                # tokenizer is bit-identical to the legacy linear one, and
                # depth must be earned by the gradient (the same worst-case
                # principle as fused beta / duplex alpha / head gamma).
                # The PAC lane is untouched: the phase tokenizer must stay
                # linear and bias-free for gauge invariance, and does.
                d4 = max(hidden_dim // 4, 8)
                self.stem_conv = nn.Sequential(
                    nn.Conv1d(1, d4, kernel_size=11, stride=5, padding=5),
                    nn.GELU(),
                    nn.Conv1d(d4, hidden_dim // 2, kernel_size=7, stride=5,
                              padding=3),
                    nn.GELU(),
                )
                self.stem_out = nn.Conv1d(hidden_dim // 2, hidden_dim,
                                          kernel_size=1)
                nn.init.zeros_(self.stem_out.weight)
                nn.init.zeros_(self.stem_out.bias)
        if tokenizer_mode in ("pac_interaction", "hybrid", "fused", "duplex"):
            if hidden_dim % 2:
                raise ValueError("pac_interaction tokenizer needs an even hidden_dim")
            complex_dim = hidden_dim // 2
            # A single real linear map is shared by real/imaginary unit phase.
            # With no bias it is exactly phase-equivariant:
            # L(e^{i delta} p) = e^{i delta} L(p).
            self.phase_tokenizer = nn.Conv1d(
                1, complex_dim, kernel_size=patch_len,
                stride=patch_len, bias=False,
            )
            self.amplitude_tokenizer = nn.Conv1d(
                1, complex_dim, kernel_size=patch_len,
                stride=patch_len,
            )
            # Diagonal amplitude calibration keeps the PAC tokenizer exactly
            # parameter-matched to the legacy Conv1d tokenizer (whose output
            # bias has hidden_dim rather than complex_dim entries). It lies on
            # the sole token path and cannot bypass the interaction.
            self.amplitude_scale = nn.Parameter(torch.ones(complex_dim))
            # Only allocated for multi-scale, so single-scale configs keep
            # their exact parameter count and their exact RNG consumption.
            if len(self.pac_patch_lens) > 1:
                self.scale_proj = nn.Linear(
                    hidden_dim * len(self.pac_patch_lens), hidden_dim
                )
            if interaction_mode == "concat":
                # SleepPACNet-style fusion control (AGENT.md 13.43-G, "most
                # important missing baseline"): expose the SAME invariant
                # ingredients (a_j, Re/Im of the aligned phase) but let a
                # learned projection combine them instead of forcing a
                # multiplicative interaction. This is the free path §13.18
                # warns about: the network COULD learn to reduce to
                # amplitude-only and ignore phase; the product arms cannot.
                # Adds ~25K params vs the product arms (concat_proj below) --
                # documented, not hidden -- so a product win is conservative
                # (product wins with LESS capacity); a concat win would need
                # this margin controlled before being taken at face value.
                self.concat_proj = nn.Linear(3 * complex_dim, hidden_dim)

    @property
    def n_token_bands(self) -> int:
        """Rows on the token grid's frequency axis. 2*n_bands under "hybrid"
        (raw rows + interaction rows), n_bands otherwise. BandPE(index) and the
        band/spatial heads must be sized from THIS, not from n_bands -- the
        builder reads it so no config has to know the factor."""
        if self.tokenizer_mode in ("hybrid", "duplex"):
            return 2 * self.n_bands
        return self.n_bands

    def token_band_hz(self) -> torch.Tensor:
        """band_hz aligned to the token grid rows: duplicated under "hybrid"
        (an interaction row describes the same target band as its raw row).
        NOTE for band_pe: "hz" -- the duplicate rows get identical PEs, so hz
        mode cannot tell a raw row from its interaction row; "index" gives each
        of the 2*nb rows its own embedding and is what hybrid configs should
        use. band_hz() itself stays n_bands-shaped: the sinc bank has n_bands
        filters and the coupling matrices are (nb, nb) regardless of mode."""
        bh = self.band_hz()
        if self.tokenizer_mode in ("hybrid", "duplex"):
            return torch.cat([bh, bh], dim=0)
        return bh

    def band_hz(self) -> torch.Tensor:
        """(n_bands, 2): [center_freq, bandwidth] in Hz, from the sinc params."""
        low = self.sinc.min_low_hz + self.sinc.low_hz_.abs()
        high = low + self.sinc.min_band_hz + self.sinc.band_hz_.abs()
        center = (low + high) / 2
        width = high - low
        return torch.cat([center, width], dim=1)                # (n_bands, 2)

    def _pac_interaction(self, phase_feat, amplitude_feat, pac_vector):
        """Mandatory gauge-invariant phase-amplitude token interaction.

        ``phase_feat`` is complex (B,C,P,I,K), ``amplitude_feat`` is real
        (B,C,P,J,K), and ``pac_vector[...,I,J]`` is

            Z_ij = E[(A_j - mean A_j) exp(i phi_i)].

        Both fusion modes start from the SAME gauge-invariant ingredient,
        ``aligned_phase`` (the alpha-weighted, preferred-phase-aligned sum of
        slower-band phase features):

            aligned_phase_j = sum_{i<j} alpha_ij exp(-i angle Z_ij) p_i   (j>0)
            aligned_phase_0 = p_0                                        (root)

        where alpha is the row-normalised |Z|. Under an arbitrary phase
        reference shift delta_i, p_i -> exp(i delta_i)p_i and
        Z_ij -> exp(i delta_i)Z_ij, so the two factors cancel exactly and
        aligned_phase_{j>0} is gauge-invariant -- this holds regardless of how
        it is later combined with amplitude, so BOTH fusion modes below
        inherit it. This is the physical gauge invariance the old
        phase-steered mixer lacked.

        ``interaction_mode="product"`` (the original): h_j = a_j *
        aligned_phase_j. There is no raw high-band token beside this
        interaction, so the amplitude and the aligned phase cannot be pulled
        apart again downstream -- the interaction is forced.

        ``interaction_mode="rotation"`` (OURS): h_j = a_j * aligned_phase_j /
        |aligned_phase_j|. The coupling ROTATES the amplitude token instead of
        also rescaling it. Forced exactly as strongly as ``product`` -- the
        token's phase is still determined entirely by the coupling-aligned
        mixture and there is still no raw high-band token beside it -- but
        |h_j| = |a_j| now holds exactly, so band power reaches the encoder
        intact.

        Why ``product`` needed fixing, measured rather than argued
        (scripts/pac_noise_diag.py, scripts/pac_noise_diag2.py): |aligned_phase|
        has a coefficient of variation of ~0.75 across patches on every corpus
        tested, and on BCI-IV-2a its statistics are identical across all four
        classes (mean |Z| 0.0027-0.0031, preferred-phase consistency at the
        surrogate null). So on a band-power task ``product`` multiplies the one
        discriminative quantity, a_j, by a per-patch random gain carrying no
        label information -- and it scores 0.259 on a 4-class problem whose
        chance level is 0.25. The PAC content was never in that modulus: it is
        in the DIRECTION of aligned_phase, which ``rotation`` keeps in full.
        On TUEV, where this tokenizer wins +0.172 kappa, the class signal lives
        in |Z| (the artifact class runs 17x the others), and |Z| enters through
        the mixing weights alpha, i.e. through the direction -- so rotation
        keeps that too.

        A significance gate on the coupling was tried first and rejected by
        measurement, not by taste: with the null level calibrated by
        circular-shift surrogates (scripts/pac_null_calib.py), 36% of edges beat
        their null on BOTH corpora and the fraction is flat across classes
        within TUEV. Coupling in motor imagery is statistically real; it is just
        not class-discriminative, so gating on significance keeps precisely the
        useless coupling. Significance is not discriminativeness.

        ``interaction_mode="concat"`` (SleepPACNet-style control): h_j =
        Linear([a_j, Re(aligned_phase_j), Im(aligned_phase_j)]). The same
        invariant ingredients are exposed, but nothing forces them to
        interact -- the projection could in principle learn to ignore the
        phase columns and pass amplitude through, exactly the free path
        §13.18 says gets optimised away when a prior has one.
        """
        B, C, P, nb, K = phase_feat.shape
        edge = pac_vector.transpose(-2, -1)               # (B,C,P,target,source)
        valid = torch.tril(
            torch.ones(nb, nb, dtype=torch.bool, device=edge.device),
            diagonal=-1,
        )
        mag = edge.abs() * valid
        unit = edge / edge.abs().clamp_min(1e-8)

        if self.pac_token_mode == "uniform":
            count = valid.sum(dim=-1, keepdim=True).clamp_min(1)
            coeff = (valid.to(edge.dtype) / count).view(
                1, 1, 1, nb, nb
            ).expand(B, C, P, -1, -1)
        else:
            if self.pac_token_mode == "scramble":
                # Preserve every |Z| and the batch's exact preferred-phase
                # distribution while breaking which edge owns which phase.
                valid_flat = valid.reshape(nb * nb)
                values = unit.reshape(B, C, P, nb * nb)[..., valid_flat]
                order = torch.rand_like(values.real).argsort(-1)
                shuffled = values.gather(-1, order)
                flat = torch.zeros(
                    B, C, P, nb * nb, dtype=unit.dtype, device=unit.device
                )
                flat[..., valid_flat] = shuffled
                unit = flat.reshape_as(unit)
            if self.pac_token_mode == "magnitude":
                # Deterministic phase-destruction control (AGENT.md 13.43-G6).
                # Keeps the measured magnitude weighting, drops preferred-phase
                # alignment entirely. Unlike `scramble` it injects NO randomness,
                # so `measured - magnitude` isolates the preferred phase without
                # scramble's per-forward permutation noise confound.
                # Like `uniform`/`scramble`, it is NOT gauge-invariant -- only
                # `measured` is, because only there does exp(-i angle Z) cancel
                # the rotation of p_i.
                phase_factor = torch.ones_like(unit)
            else:
                phase_factor = unit.conj()
            denom = mag.sum(dim=-1, keepdim=True)
            measured = (mag / denom.clamp_min(1e-8)) * phase_factor
            count = valid.sum(dim=-1, keepdim=True).clamp_min(1)
            fallback = valid.to(edge.dtype) / count
            coeff = torch.where(denom > 1e-8, measured, fallback)

        aligned_phase = torch.einsum(
            "bcpji,bcpik->bcpjk", coeff, phase_feat
        )
        # The slowest band has no lower-frequency driver. Preserve its own
        # analytic token as the root of the directed hierarchy.
        #
        # Written as a masked select rather than the original in-place
        # `aligned_phase[:, :, :, 0, :] = phase_feat[:, :, :, 0, :]`: an
        # in-place write into an einsum output is an autograd-versioned
        # mutation, which fails under checkpointing/compile on ROCm. The
        # selected values are identical -- band 0 takes its own phase feature,
        # every other band keeps the aligned sum. Verified numerically by
        # tests/test_paclock_equivalence.py.
        nb_idx = torch.arange(nb, device=aligned_phase.device).view(1, 1, 1, nb, 1)
        aligned_phase = torch.where(nb_idx == 0, phase_feat, aligned_phase)

        if self.interaction_mode == "product":
            return amplitude_feat.to(aligned_phase.dtype) * aligned_phase
        if self.interaction_mode == "rotation":
            # Per-component unit modulus: |h_jk| = |a_jk| for every one of the K
            # complex features, so the amplitude token passes through exactly and
            # only the coupling's phase geometry is applied to it. The modulus is
            # gauge-invariant, so normalising by it keeps aligned_phase's gauge
            # invariance. clamp_min guards near-cancellation of the alpha-weighted
            # sum -- phase_feat is an unnormalised learned projection, so its
            # modulus has no lower bound.
            unit_phase = aligned_phase / aligned_phase.abs().clamp_min(1e-6)
            return amplitude_feat.to(unit_phase.dtype) * unit_phase
        # concat: expose the same ingredients, let a learned projection combine
        # them. Real, already at the token width (hidden_dim), no view_as_real
        # needed downstream.
        feat = torch.cat(
            [amplitude_feat, aligned_phase.real, aligned_phase.imag], dim=-1
        )
        return self.concat_proj(feat)

    def _interaction_tokens(self, phase_unit, amplitude, pac_vectors):
        """Analytic phase/amplitude -> real interleaved PAC interaction tokens.

        ``pac_vectors`` is a list, one coupling tensor per PAC window. The
        waveform tokenisation below does not depend on the window, so it runs
        once and only the alignment is repeated per scale -- multi-scale costs
        an einsum and a projection, not another frontend.
        """
        if not isinstance(pac_vectors, (list, tuple)):
            pac_vectors = [pac_vectors]
        B, C, nb, T = phase_unit.shape
        flat_shape = (B * C * nb, T)
        # _patch_project returns (N, P, K), so amplitude_scale now broadcasts
        # over the trailing K axis and no longer needs view(1, -1, 1).
        pr = _patch_project(self.phase_tokenizer, phase_unit.real.reshape(flat_shape))
        pi = _patch_project(self.phase_tokenizer, phase_unit.imag.reshape(flat_shape))
        amp = _patch_project(
            self.amplitude_tokenizer, torch.log1p(amplitude).reshape(flat_shape)
        )
        amp = amp * self.amplitude_scale
        P, K = pr.shape[1], pr.shape[2]
        phase_feat = torch.complex(pr, pi).reshape(
            B, C, nb, P, K
        ).permute(0, 1, 3, 2, 4)
        amplitude_feat = amp.reshape(
            B, C, nb, P, K
        ).permute(0, 1, 3, 2, 4)
        per_scale = []
        for pac_vector in pac_vectors:
            interaction = self._pac_interaction(
                phase_feat, amplitude_feat, pac_vector
            )
            if self.interaction_mode in ("product", "rotation"):
                per_scale.append(torch.view_as_real(interaction).flatten(-2))
            else:
                per_scale.append(interaction)   # concat_proj already real (B,C,P,nb,D)
        if len(per_scale) == 1:
            tokens = per_scale[0]                           # (B,C,P,nb,D)
        else:
            tokens = self.scale_proj(torch.cat(per_scale, dim=-1))
        return tokens.permute(0, 1, 3, 2, 4).contiguous()   # (B,C,nb,P,D)

    def forward(self, x: torch.Tensor, return_amp_target: bool = False):
        # The whole frontend runs in fp32 even under autocast. Point-wise
        # `.float()` calls are not enough: autocast re-casts the operators
        # downstream of them, and the analytic signal is complex, so bf16 reaches
        # the complex kernels and they refuse it outright ("Expected both inputs
        # to be Half, Float or Double tensors but got BFloat16"). Beyond the
        # mechanics, a gauge-invariant phase estimated from a bf16 arctangent is
        # not the quantity the model is about. The cost of excluding it is small:
        # the frontend profiles at roughly 3% of step time, the encoder is where
        # the kernels pile up.
        with torch.autocast(device_type=x.device.type, enabled=False):
            return self._forward_fp32(x.float(), return_amp_target)

    def _forward_fp32(self, x: torch.Tensor, return_amp_target: bool = False):
        B, C, T = x.shape
        filtered = self.sinc(x.reshape(B * C, 1, T)).reshape(B, C, self.n_bands, T)

        # phase / amplitude -> time-resolved per-channel coupling
        z = hilbert(filtered)                                    # (B, C, nb, T)
        phase_unit, amplitude = phase_amplitude(z)
        if self.tokenizer_mode in ("raw", "hybrid", "fused", "duplex"):
            f = filtered.reshape(B * C * self.n_bands, T)
            feat = _patch_project(self.tokenizer, f)             # (B*C*nb, P, D)
            P = feat.shape[1]
            if self.raw_stem == "deep":
                # residual refinement per patch; stride 5*5 then adaptive pool
                # to one vector per patch keeps the token grid shape identical
                h = self.stem_conv(f[:, : P * self.patch_len].unsqueeze(1))
                # (N, D/2, T'): pool T' into P patch bins, then 1x1 to D
                h = torch.nn.functional.adaptive_avg_pool1d(h, P)
                feat = feat + self.stem_out(h).transpose(1, 2)   # (N, P, D)
            tokens = feat.reshape(B, C, self.n_bands, P, -1)
        else:
            P = T // self.patch_len
        # patch_len drives two separable things at once: how many tokens the
        # grid has, and how long a window the PAC statistic is estimated over.
        # A sweep that moves it therefore cannot say which of the two produced
        # the effect -- and for this model they are not the same claim, because
        # the estimation window is where the physics lives and the token count
        # is only resolution. pac_patch_len separates them; when it equals
        # patch_len (the default) the arithmetic below is the original.
        pac_vectors = []
        for win in self.pac_patch_lens:
            P_pac = T // win
            pv = patch_pac_vector(phase_unit, amplitude, P_pac, self.normalize)
            if P_pac != P:
                if P % P_pac:
                    raise ValueError(
                        f"pac_patch_len={win} gives {P_pac} PAC windows, which "
                        f"does not divide the {P} token patches; every "
                        f"pac_patch_len must be a multiple of patch_len"
                    )
                # one PAC window spans several token patches: every token
                # inside it sees the same coupling matrix
                pv = pv.repeat_interleave(P // P_pac, dim=2)
            pac_vectors.append(pv)
        pac_vector = pac_vectors[0]
        if self.tokenizer_mode == "pac_interaction":
            tokens = self._interaction_tokens(
                phase_unit, amplitude, pac_vectors
            )
        elif self.tokenizer_mode == "duplex":
            interaction = self._interaction_tokens(
                phase_unit, amplitude, pac_vectors
            )                                                  # (B,C,nb,P,D)
            beta = self.fusion_beta.view(1, 1, self.n_bands, 1, -1)
            fused_rows = tokens + beta * interaction
            if self.coupling_strength:
                # |Z| is (B,C,P,nb_i,nb_j); band j's column over i -> (B,C,nb_j,P,nb_i)
                cs = pac_vector.abs().permute(0, 1, 4, 2, 3).to(fused_rows.dtype)
                fused_rows = fused_rows + self.cs_proj(cs)
            gated_rows = interaction * self.interaction_gate.view(1, 1, -1, 1, 1)
            tokens = torch.cat([fused_rows, gated_rows], dim=2)  # (B,C,2nb,P,D)
        elif self.tokenizer_mode == "fused":
            interaction = self._interaction_tokens(
                phase_unit, amplitude, pac_vectors
            )                                                  # (B,C,nb,P,D)
            if self.fusion_mode == "blend":
                beta = self.fusion_beta.view(1, 1, self.n_bands, 1, -1)
                tokens = tokens + beta * interaction
            else:
                g = torch.sigmoid(self.fusion_gate(
                    torch.cat([tokens, interaction], dim=-1)))
                tokens = g * tokens + (1.0 - g) * interaction
        elif self.tokenizer_mode == "hybrid":
            # Raw rows first (0..nb-1), interaction rows after (nb..2nb-1).
            # The raw rows here are BIT-IDENTICAL to tokenizer_mode="raw" and
            # the interaction rows to tokenizer_mode="pac_interaction" under
            # the same weights -- asserted by scripts/verify_hybrid.py, not
            # assumed. Row order is a contract: checkpoint surgery and the
            # ablation that deletes interaction rows both index by it.
            interaction = self._interaction_tokens(
                phase_unit, amplitude, pac_vectors
            )
            if self.hybrid_gate == "band":
                interaction = interaction * self.interaction_gate.view(1, 1, -1, 1, 1)
            tokens = torch.cat([tokens, interaction], dim=2)  # (B,C,2nb,P,D)
        # `coupling` and the returned `pac_vector` stay single-scale: they feed
        # the coupling/phase freq-mixers and the phase-ablation modes, whose
        # semantics are defined for one estimation window. The first entry is
        # used, so a single-scale config is unaffected.
        coupling = pac_vector.abs()

        if return_amp_target:
            # hybrid/duplex: the interaction token for band j carries a_j
            # directly, so any masking scheme MUST hide both of a band's rows
            # together -- build.py's crossfreq_aux_loss draws its mask over the
            # nb PHYSICAL bands and applies it to row j and row nb+j jointly
            # (scripts/verify_duplex_pretrain.py). The target below is per
            # physical band -- (B, C, nb, P) -- whatever the grid height.
            # Per-token (electrode, band, patch) log mean amplitude -- a fixed,
            # deterministic regression target for masked-reconstruction pretraining
            # (models/pretrain.py). Deterministic => no representation collapse, no
            # target encoder needed. Predicting a HIGH band's amplitude from a
            # masked grid.  The asymmetric high-band mask is PAC-inspired, but the
            # target remains a statistical amplitude target rather than a claim
            # that biological coupling is uniquely identified.
            L = T // P
            am = amplitude[..., : P * L].reshape(B, C, self.n_bands, P, L)
            amp_target = torch.log(am.mean(dim=-1) + 1e-6)      # (B, C, nb, P)
            if self.return_pac_vector:
                return tokens, coupling, self.token_band_hz(), amp_target, pac_vector
            return tokens, coupling, self.token_band_hz(), amp_target

        if self.return_pac_vector:
            return tokens, coupling, self.token_band_hz(), pac_vector
        return tokens, coupling, self.token_band_hz()
