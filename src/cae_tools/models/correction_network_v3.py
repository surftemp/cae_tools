"""
Correction Network v3 — Flow-based, soft blending, no threshold.

The CN computes p(LST | inputs) where:
- inputs: the SAME 11-channel tensor the UNet receives (all conditioning variables)
- LST: the target variable, enters ONLY through the flow coupling layers

Backbone: StandardEncoder + StandardDecoder (same architecture as main UNet).
The decoder outputs base_channels feature maps (not 1-channel prediction),
which serve as context for the flow coupling layers.

Correction mechanism (soft blending, no threshold):
    z = flow_forward(target, context)
    x_mean = flow_inverse(0, context)
    w = sigmoid(-0.5 * z^2)              # from density ratio derivation
    lst_cn = w * target + (1 - w) * x_mean

    Derivation of w:
        The log-density difference between the observed target and the
        conditional mean (mode) simplifies to -0.5 * z^2, because the
        log-Jacobian-determinant of affine coupling layers depends only
        on context, not on the variable being transformed. Therefore:
            w = p(target|ctx) / (p(target|ctx) + p(x_mean|ctx))
              = sigmoid(log p(target) - log p(x_mean))
              = sigmoid(-0.5 * z^2)

    z is DETACHED in w so UNet gradient cannot manipulate the CN's density
    estimate. x_mean is NOT detached so UNet gradient can improve the CN's
    corrections (cooperative learning). The CN's own NLL loss keeps it
    grounded in density estimation.

Standing instruction: no VGG/perceptual loss code.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod

from cae_tools.models.standard_unet import StandardEncoder, StandardDecoder


# =============================================================================
# Base interface
# =============================================================================

class CorrectionNetworkBase(ABC, nn.Module):

    @abstractmethod
    def forward(self, inputs, lst_raw_norm):
        """
        Args:
            inputs: conditioning variables (B, C_in, H, W) — same tensor UNet receives
            lst_raw_norm: normalised LST target (B, 1, H, W) — variable being modelled
        Returns:
            lst_cn: corrected LST (B, 1, H, W)
        """
        pass

    @abstractmethod
    def get_cn_own_loss(self):
        pass

    @abstractmethod
    def get_diagnostics(self):
        pass


# =============================================================================
# Affine coupling layer
# =============================================================================

class AffineCouplingLayer(nn.Module):
    """
    Scalar affine transform: z_out = z_in * exp(s(context)) + t(context)
    Invertible: z_in = (z_out - t) * exp(-s)

    s and t depend on context ONLY (not on z_in). This means the
    log-Jacobian-determinant (= s) is constant with respect to the
    variable being transformed. This property is essential for the
    soft blending weight derivation.
    """

    def __init__(self, context_channels, hidden_channels=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(context_channels, hidden_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 2, 1),
        )
        # Zero-initialise final layer so initial flow is identity: z_out = z_in
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, z, context):
        st = self.net(context)
        s = st[:, 0:1, :, :].clamp(-3.0, 3.0)
        t = st[:, 1:2, :, :]
        z_out = z * torch.exp(s) + t
        return z_out, s

    def inverse(self, z_out, context):
        st = self.net(context)
        s = st[:, 0:1, :, :].clamp(-3.0, 3.0)
        t = st[:, 1:2, :, :]
        return (z_out - t) * torch.exp(-s)


# =============================================================================
# Flow CN — StandardEncoder + StandardDecoder backbone, soft blending
# =============================================================================

class FlowCorrectionNetwork(CorrectionNetworkBase):
    """
    Conditional normalizing flow for p(LST | inputs).

    Backbone: StandardEncoder + StandardDecoder — same architecture as the
    main UNet, but the decoder outputs base_channels feature maps (not 1
    channel with sigmoid). These feature maps serve as spatial context for
    the flow coupling layers.

    LST enters ONLY through the flow coupling layers as the variable being
    modelled. The backbone never sees LST.

    Correction uses soft blending with weights derived from the flow's own
    density estimate. No fixed threshold. No imposed contamination rate.
    The flow determines both WHICH pixels to correct and HOW MUCH to correct.
    """

    def __init__(self, n_input_channels, base_channels=64,
                 n_coupling_layers=4, coupling_hidden=64,
                 dropout_rate=0.0):
        super().__init__()
        self.n_input_channels = n_input_channels
        self.base_channels = base_channels
        self.apply_corrections = True  # Set by joint trainer per phase

        # Backbone: same architecture as main UNet.
        # Encoder: in_channels -> base -> base*2 -> base*4 -> base*8
        #          with skip connections at each stage.
        # Decoder: base*8 -> base*4 -> base*2 -> base -> base_channels
        #          output_activation='none' because these are internal
        #          features for the coupling layers, not predictions.
        self.encoder = StandardEncoder(
            in_channels=n_input_channels,
            base_channels=base_channels,
            dropout_rate=dropout_rate,
        )
        self.decoder = StandardDecoder(
            out_channels=base_channels,
            base_channels=base_channels,
            dropout_rate=dropout_rate,
            output_activation='none',
        )

        # Flow coupling layers operate on the backbone's feature map.
        # Each layer predicts per-pixel scale (s) and shift (t) from
        # the base_channels context features.
        self.coupling_layers = nn.ModuleList([
            AffineCouplingLayer(base_channels, coupling_hidden)
            for _ in range(n_coupling_layers)
        ])

        self._cn_own_loss = None
        self._diagnostics = {}
        self._last_nll = None
        self._last_weights = None
        self._last_z = None
        self._last_x_mean = None

    def _get_context(self, inputs):
        """
        Run backbone (encoder + decoder) to get spatial context features.

        Args:
            inputs: (B, n_input_channels, H, W)
        Returns:
            context: (B, base_channels, H, W) — feature map for coupling layers
        """
        encoded, skips = self.encoder(inputs)
        context = self.decoder(encoded, skips)
        return context

    def _flow_forward(self, lst, context):
        """
        Data -> latent. Returns z and total log_det.

        Each coupling layer: z_out = z_in * exp(s(context)) + t(context)
        log_det per layer = s (since dz_out/dz_in = exp(s), and log of that = s)
        Total log_det = sum of s across all 4 layers.
        """
        z = lst
        total_log_det = torch.zeros_like(lst)
        for layer in self.coupling_layers:
            z, log_det = layer(z, context)
            total_log_det = total_log_det + log_det
        return z, total_log_det

    def _flow_inverse(self, z, context):
        """
        Latent -> data. Reverses each coupling layer in reverse order.

        Each inverse step: z_in = (z_out - t(context)) * exp(-s(context))
        Starting from z=0 gives the conditional mean (mode of the learned density).
        """
        x = z
        for layer in reversed(self.coupling_layers):
            x = layer.inverse(x, context)
        return x

    def _log_prob(self, lst, context):
        """
        Exact log p(lst | context) via change of variables.

        log p(x) = log p_base(f(x)) + log|det(df/dx)|
                 = [-0.5 * z^2 - 0.5 * log(2*pi)] + sum_k(s_k)

        Returns:
            log_p: (B, 1, H, W) — log-density at each pixel
            z: (B, 1, H, W) — latent value at each pixel
        """
        z, log_det = self._flow_forward(lst, context)
        log_p_base = -0.5 * (z.pow(2) + math.log(2 * math.pi))
        return log_p_base + log_det, z

    def forward(self, inputs, lst_raw_norm):
        """
        Args:
            inputs: (B, n_input_channels, H, W) — conditioning variables only
            lst_raw_norm: (B, 1, H, W) — LST target to evaluate/correct
        Returns:
            lst_cn: (B, 1, H, W) — corrected LST (soft-blended)
        """
        context = self._get_context(inputs)

        # ---- CN own loss: negative log-likelihood ----
        log_p, z = self._log_prob(lst_raw_norm, context)
        self._cn_own_loss = -log_p.mean()

        # ---- Soft blending (when corrections are enabled) ----
        if self.apply_corrections:
            # Conditional mean: inverse flow from z=0 (mode of base N(0,1))
            # This is the LST value the flow considers most probable.
            # NOT detached: UNet gradient can improve x_mean via CN parameters.
            x_mean = self._flow_inverse(
                torch.zeros_like(lst_raw_norm), context)
            # Clamp to data range [0,1]. The flow is trained on [0,1] data,
            # so after convergence x_mean should be in-range, but early in
            # training it might not be.
            x_mean = x_mean.clamp(0.0, 1.0)

            # Blending weight from the density ratio.
            #
            # Derivation:
            #   w = p(target | ctx) / (p(target | ctx) + p(x_mean | ctx))
            #
            # For affine coupling flows, s_k depends on context not on x,
            # so the log-Jacobian-determinant cancels in the ratio:
            #   log p(target) - log p(x_mean) = -0.5 * z_target^2 - (-0.5 * 0^2)
            #                                 = -0.5 * z_target^2
            # Therefore:
            #   w = sigmoid(-0.5 * z_target^2)
            #
            # z is DETACHED so UNet gradient cannot flow through w back to CN.
            w = torch.sigmoid(-0.5 * z.detach().pow(2))

            # Effective target: smooth blend between observed and flow mean.
            # Clean pixel (z~0): w~0.5, target~x_mean, so lst_cn~target.
            # Contaminated pixel (|z|>>0): w~0, lst_cn~x_mean.
            lst_cn = w * lst_raw_norm + (1.0 - w) * x_mean

        else:
            # Parallel phase: no correction, raw targets passed through.
            lst_cn = lst_raw_norm
            w = 0.5 * torch.ones_like(lst_raw_norm)
            x_mean = lst_raw_norm  # No meaningful x_mean in parallel phase

        # Store per-pixel tensors for diagnostic images
        nll = -log_p.detach()
        self._last_nll = nll
        self._last_weights = w.detach()
        self._last_z = z.detach()
        self._last_x_mean = x_mean.detach()

        # ---- Aggregated diagnostics ----
        with torch.no_grad():
            correction = (lst_cn - lst_raw_norm).detach()
            self._diagnostics = {
                'mean_w': w.mean().item(),
                'pct_w_below_0.1': (w < 0.1).float().mean().item() * 100.0,
                'pct_w_below_0.01': (w < 0.01).float().mean().item() * 100.0,
                'mean_correction': correction.abs().mean().item(),
                'max_correction': correction.abs().max().item(),
                'mean_nll': nll.mean().item(),
                'median_nll': nll.median().item(),
                'p99_nll': nll.quantile(0.99).item(),
                'max_nll': nll.max().item(),
                'mean_abs_z': z.abs().mean().item(),
                'max_abs_z': z.abs().max().item(),
            }

        return lst_cn

    def get_cn_own_loss(self):
        return self._cn_own_loss if self._cn_own_loss is not None else torch.tensor(0.0)

    def get_regularisation_loss(self):
        return self.get_cn_own_loss()

    def get_diagnostics(self):
        return self._diagnostics

    def get_last_nll(self):
        """Per-pixel NLL tensor from most recent forward pass."""
        return self._last_nll

    def get_last_weights(self):
        """Per-pixel blending weights from most recent forward pass."""
        return self._last_weights

    def get_last_z(self):
        """Per-pixel latent z values from most recent forward pass."""
        return self._last_z

    def get_last_x_mean(self):
        """Per-pixel conditional mean from most recent forward pass."""
        return self._last_x_mean

    def compute_density_curve(self, inputs_single, pixel_y, pixel_x, n_points=200):
        """
        Compute p(LST | context) at a single pixel across a range of LST values.

        Args:
            inputs_single: (1, n_input_channels, H, W) — single sample
            pixel_y, pixel_x: pixel coordinates
            n_points: number of LST values to evaluate

        Returns:
            lst_values: (n_points,) numpy array in [0, 1]
            densities: (n_points,) numpy array of density values
        """
        context = self._get_context(inputs_single)
        ctx_pixel = context[:, :, pixel_y:pixel_y+1, pixel_x:pixel_x+1]

        lst_values = torch.linspace(0.0, 1.0, n_points, device=inputs_single.device)
        densities = []
        for val in lst_values:
            lst_point = torch.full((1, 1, 1, 1), val.item(), device=inputs_single.device)
            log_p, _ = self._log_prob(lst_point, ctx_pixel)
            densities.append(torch.exp(log_p).item())

        return lst_values.cpu().numpy(), densities
