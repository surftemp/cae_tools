"""
Joint training wrapper: CorrectionNetwork + main UNET.

At training time:
  1. CN corrects raw Landsat LST -> lst_cn
  2. Main UNET trains to predict LST from ERA5/static inputs, using lst_cn as target
  3. Both networks trained jointly via shared loss — full end-to-end backprop,
     no gradient severing

At inference time:
  - CN is completely discarded
  - Main UNET checkpoint is used directly with apply_cae, exactly as before
  - No changes to scoring pipeline, apply_cae.py, or any downstream tooling

Checkpoint design:
  - Main UNET checkpoint saved exactly as before (compatible with apply_cae)
  - Joint state (CN weights + both optimizers) saved separately in joint_state/
  - Continue-training loads both if joint_state/ exists

Loss terms:
  L_main = MSE(pred, lst_cn)
  L_cn   = lambda_sparsity * ||M * Delta||_1   (via CN internal reg loss)
           + lambda_cold * soft_cold_pixel_rate(pred, era5)
  L_total = L_main + L_cn   (single backward pass, full joint graph)

Monitoring printed every test_interval epochs:
  epoch, train_mse, test_mse (vs raw LST), ema_test, ema_ratio
  l_main, l_cn, l_reg, l_cold
  cn_mean_correction   — mean |M*Delta| across batch (is CN doing anything?)
  cn_mean_mask         — mean M across batch (how many pixels touched?)
  cn_pct_mask_active   — fraction of pixels where M > 0.5
  cn_delta_range       — [min, max] of correction Delta
  hard_cold_pct_train  — hard cold pixel rate on train batch (pred < ERA5-10K)
  hard_cold_pct_test   — hard cold pixel rate on test set (main diagnostic)
  lst_cn_mean_delta    — mean(lst_cn - lst_raw): how much did CN shift the target?
  lst_cn_cold_removed  — fraction of cold pixels in lst_raw that CN lifted above threshold
"""

import os
import json
import time
import torch
import torch.nn as nn
import numpy as np

from cae_tools.models.unet import UNET
from cae_tools.models.correction_network import (
    build_correction_network,
    soft_cold_pixel_rate,
)


# Channel index of ERA5 SKT in the input tensor
ERA5_CHANNEL_IDX = 3

# Static channels to feed into CN (no monthly-derived channels)
# Indices: land_cover=0, elevation=2, slope_magnitude=6, slope_direction=7
CN_STATIC_CHANNEL_INDICES = [0, 2, 6, 7]


class JointUNET:
    """
    Wraps a UNET (main network) and a CorrectionNetwork for joint training.

    The UNET inside is a standard UNET instance and can be saved/loaded
    independently for use with apply_cae at inference time.
    """

    def __init__(
        self,
        unet,
        cn_type='conv',
        cn_base_channels=32,
        cn_n_conv_layers=4,
        cn_cold_threshold_norm=0.3,
        lambda_sparsity=0.01,
        lambda_cold=0.1,
        cold_threshold_k=10.0,
        cn_lr=0.0001,
    ):
        self.unet = unet
        self.lambda_cold = lambda_cold
        self.cold_threshold_k = cold_threshold_k

        self.cn = build_correction_network(
            cn_type=cn_type,
            static_channel_indices=CN_STATIC_CHANNEL_INDICES,
            base_channels=cn_base_channels,
            n_conv_layers=cn_n_conv_layers,
            cold_threshold_norm=cn_cold_threshold_norm,
            lambda_sparsity=lambda_sparsity,
        )

        self.cn_lr = cn_lr
        self.cn_optimizer = None
        self._cn_saved_optimizer_path = None

    def _get_era5_channel(self, inputs):
        return inputs[:, ERA5_CHANNEL_IDX:ERA5_CHANNEL_IDX+1, :, :]

    def _hard_cold_pct(self, pred_norm, era5_norm, norm_params):
        """Hard (non-differentiable) cold pixel fraction for monitoring."""
        min_out = norm_params['min_output']
        max_out = norm_params['max_output']
        pred_k = pred_norm.detach() * (max_out - min_out) + min_out

        era5_min = norm_params['inputs']['era5_skt']['min']
        era5_max = norm_params['inputs']['era5_skt']['max']
        era5_k = era5_norm.detach() * (era5_max - era5_min) + era5_min

        cold = (pred_k < (era5_k - self.cold_threshold_k)).float().mean().item()
        return cold * 100.0  # percent

    def _cn_diagnostics(self, lst_raw_norm, lst_cn, era5_norm, norm_params):
        """
        Compute CN-specific diagnostics for monitoring.
        Returns dict of scalar values.
        """
        with torch.no_grad():
            correction = (lst_cn - lst_raw_norm).detach()

            # How much did CN shift the target overall?
            lst_cn_mean_delta = correction.mean().item()

            # Correction magnitude stats
            abs_correction = correction.abs()
            cn_mean_correction = abs_correction.mean().item()
            cn_max_correction = abs_correction.max().item()

            # What fraction of pixels were meaningfully corrected (>0.01 norm units)?
            cn_pct_meaningfully_corrected = (abs_correction > 0.01).float().mean().item() * 100.0

            # How many raw-cold pixels did CN lift above threshold?
            era5_min = norm_params['inputs']['era5_skt']['min']
            era5_max = norm_params['inputs']['era5_skt']['max']
            era5_k = era5_norm.detach() * (era5_max - era5_min) + era5_min

            min_out = norm_params['min_output']
            max_out = norm_params['max_output']
            lst_raw_k = lst_raw_norm.detach() * (max_out - min_out) + min_out
            lst_cn_k  = lst_cn.detach() * (max_out - min_out) + min_out

            raw_cold_mask = (lst_raw_k < (era5_k - self.cold_threshold_k))
            n_raw_cold = raw_cold_mask.float().sum().item()
            if n_raw_cold > 0:
                # Of those cold pixels, how many did CN lift above threshold?
                lifted = (raw_cold_mask & (lst_cn_k >= (era5_k - self.cold_threshold_k)))
                lst_cn_cold_removed_pct = lifted.float().sum().item() / n_raw_cold * 100.0
            else:
                lst_cn_cold_removed_pct = 100.0  # nothing to fix

            # Mean correction in Kelvin (for interpretability)
            correction_k = correction * (max_out - min_out)
            cn_mean_correction_k = correction_k.abs().mean().item()
            cn_max_correction_k  = correction_k.abs().max().item()

        return {
            'lst_cn_mean_delta_norm': lst_cn_mean_delta,
            'cn_mean_correction_norm': cn_mean_correction,
            'cn_max_correction_norm': cn_max_correction,
            'cn_mean_correction_k': cn_mean_correction_k,
            'cn_max_correction_k': cn_max_correction_k,
            'cn_pct_meaningfully_corrected': cn_pct_meaningfully_corrected,
            'lst_cn_cold_removed_pct': lst_cn_cold_removed_pct,
            'n_raw_cold_pixels': n_raw_cold,
        }

    def train_joint(
        self,
        train_dataset,
        test_dataset,
        model_folder,
        nr_epochs=3500,
        batch_size=512,
        checkpoint_interval=500,
        database_path=None,
    ):
        from torch.utils.data import DataLoader

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Device: {device}")

        # Move networks to device
        if self.unet.encoder:
            self.unet.encoder = self.unet.encoder.to(device)
        if self.unet.decoder:
            self.unet.decoder = self.unet.decoder.to(device)
        if self.unet.flow_model:
            self.unet.flow_model = self.unet.flow_model.to(device)
        self.cn = self.cn.to(device)

        os.makedirs(model_folder, exist_ok=True)
        self.unet.save(model_folder)

        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=4, pin_memory=True
        )
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False,
            num_workers=2, pin_memory=True
        )

        norm_params = self.unet.normalisation_parameters

        # Optimizers
        unet_params = []
        if self.unet.encoder:
            unet_params += list(self.unet.encoder.parameters())
        if self.unet.decoder:
            unet_params += list(self.unet.decoder.parameters())
        if self.unet.flow_model:
            unet_params += list(self.unet.flow_model.parameters())

        unet_optimizer = torch.optim.Adam(
            unet_params, lr=self.unet.lr, weight_decay=self.unet.weight_decay
        )
        self.cn_optimizer = torch.optim.Adam(
            self.cn.parameters(), lr=self.cn_lr
        )

        unet_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            unet_optimizer, T_max=nr_epochs
        )
        cn_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.cn_optimizer, T_max=nr_epochs
        )

        # Restore states if continuing
        start_epoch = len(self.unet.history.get('train_mse', []))

        if getattr(self.unet, '_saved_optimizer_state_path', None):
            unet_optimizer.load_state_dict(
                torch.load(self.unet._saved_optimizer_state_path, map_location=device)
            )
            print("Restored UNET optimizer state")

        if getattr(self.cn, '_saved_optimizer_state_path', None):
            self.cn_optimizer.load_state_dict(
                torch.load(self.cn._saved_optimizer_state_path, map_location=device)
            )
            print("Restored CN optimizer state")

        for _ in range(start_epoch):
            unet_scheduler.step()
            cn_scheduler.step()

        print(f"Scheduler: T_max={nr_epochs}, already done={start_epoch}, "
              f"this job={nr_epochs - start_epoch}")

        ema_test = None
        ema_ratio = None
        ema_alpha = 0.3
        best_ema_test = float('inf')
        best_ema_ratio = float('-inf')

        mse_loss = nn.MSELoss()

        for epoch in range(start_epoch, nr_epochs):
            # ================================================================
            # TRAINING PASS
            # ================================================================
            if self.unet.encoder:
                self.unet.encoder.train()
            if self.unet.decoder:
                self.unet.decoder.train()
            if self.unet.flow_model:
                self.unet.flow_model.train()
            self.cn.train()

            train_loss_sum = 0.0
            l_main_sum = 0.0
            l_cn_sum = 0.0
            l_reg_sum = 0.0
            l_cold_sum = 0.0
            hard_cold_train_sum = 0.0
            train_batches = 0

            # For last batch diagnostics (cn diagnostics too slow to accumulate every batch)
            last_lst_raw = None
            last_lst_cn = None
            last_era5 = None
            last_pred = None

            for batch_inputs, batch_targets in train_loader:
                batch_inputs  = batch_inputs.to(device)
                batch_targets = batch_targets.to(device)

                unet_optimizer.zero_grad()
                self.cn_optimizer.zero_grad()

                lst_raw_norm = batch_targets
                era5_norm    = self._get_era5_channel(batch_inputs)

                if self.unet.augment:
                    batch_inputs, lst_raw_norm = self.unet._augment_batch(
                        batch_inputs, lst_raw_norm
                    )
                    era5_norm = self._get_era5_channel(batch_inputs)

                # CN forward (graph alive)
                lst_cn = self.cn(batch_inputs, lst_raw_norm, era5_norm)

                # Main UNET forward
                pred = self._unet_forward(batch_inputs, device)

                # Losses
                l_main = mse_loss(pred, lst_cn)
                l_reg  = self.cn.get_regularisation_loss()
                l_cold = self.lambda_cold * soft_cold_pixel_rate(
                    pred, era5_norm, norm_params,
                    cold_threshold_k=self.cold_threshold_k
                )
                l_cn_total = l_reg + l_cold
                l_total    = l_main + l_cn_total

                l_total.backward()

                torch.nn.utils.clip_grad_norm_(unet_params, max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(self.cn.parameters(), max_norm=1.0)

                unet_optimizer.step()
                self.cn_optimizer.step()

                train_loss_sum  += l_total.item()
                l_main_sum      += l_main.item()
                l_cn_sum        += l_cn_total.item()
                l_reg_sum       += l_reg.item()
                l_cold_sum      += l_cold.item()
                hard_cold_train_sum += self._hard_cold_pct(pred, era5_norm, norm_params)
                train_batches   += 1

                # Keep last batch for CN diagnostics at print time
                last_lst_raw = lst_raw_norm.detach()
                last_lst_cn  = lst_cn.detach()
                last_era5    = era5_norm.detach()
                last_pred    = pred.detach()

            train_mse = l_main_sum / max(train_batches, 1)

            unet_scheduler.step()
            cn_scheduler.step()

            # ================================================================
            # TEST PASS + MONITORING
            # ================================================================
            if epoch % self.unet.test_interval == 0:
                if self.unet.encoder:
                    self.unet.encoder.eval()
                if self.unet.decoder:
                    self.unet.decoder.eval()
                if self.unet.flow_model:
                    self.unet.flow_model.eval()
                self.cn.eval()

                test_mse_sum       = 0.0
                hard_cold_test_sum = 0.0
                test_batches       = 0

                with torch.no_grad():
                    for batch_inputs, batch_targets in test_loader:
                        batch_inputs  = batch_inputs.to(device)
                        batch_targets = batch_targets.to(device)
                        era5_norm_t   = self._get_era5_channel(batch_inputs)
                        pred          = self._unet_forward(batch_inputs, device)
                        # Test MSE against raw LST (NOT lst_cn) — comparable to standalone
                        test_mse_sum       += mse_loss(pred, batch_targets).item()
                        hard_cold_test_sum += self._hard_cold_pct(
                            pred, era5_norm_t, norm_params
                        )
                        test_batches += 1

                test_mse = test_mse_sum / max(test_batches, 1)
                hard_cold_test_pct  = hard_cold_test_sum / max(test_batches, 1)
                hard_cold_train_pct = hard_cold_train_sum / max(train_batches, 1)

                # EMA
                if ema_test is None:
                    ema_test  = test_mse
                    ema_ratio = train_mse / test_mse if test_mse > 0 else 1.0
                else:
                    ema_test  = ema_alpha * test_mse  + (1 - ema_alpha) * ema_test
                    ema_ratio = ema_alpha * (train_mse / test_mse) + \
                                (1 - ema_alpha) * ema_ratio

                # CN diagnostics from last train batch
                cn_diag = self._cn_diagnostics(
                    last_lst_raw, last_lst_cn, last_era5, norm_params
                )

                # Losses (averaged)
                avg_l_main = l_main_sum  / max(train_batches, 1)
                avg_l_cn   = l_cn_sum    / max(train_batches, 1)
                avg_l_reg  = l_reg_sum   / max(train_batches, 1)
                avg_l_cold = l_cold_sum  / max(train_batches, 1)

                lr_unet = unet_optimizer.param_groups[0]['lr']
                lr_cn   = self.cn_optimizer.param_groups[0]['lr']

                # ---- Print block ----
                print(f"\nepoch: {epoch}")
                print(f"  losses:     l_main={avg_l_main:.6f}  l_cn={avg_l_cn:.6f}  "
                      f"l_reg={avg_l_reg:.6f}  l_cold={avg_l_cold:.6f}")
                print(f"  mse:        train={train_mse:.6f}  test={test_mse:.6f}  "
                      f"ema_test={ema_test:.6f}  ema_ratio={ema_ratio:.3f}")
                print(f"  cold%:      train_hard={hard_cold_train_pct:.2f}%  "
                      f"test_hard={hard_cold_test_pct:.2f}%")
                print(f"  CN target:  mean_delta={cn_diag['lst_cn_mean_delta_norm']:+.4f} norm "
                      f"({cn_diag['lst_cn_mean_delta_norm']*(norm_params['max_output']-norm_params['min_output']):+.2f}K)")
                print(f"  CN corr:    mean={cn_diag['cn_mean_correction_k']:.3f}K  "
                      f"max={cn_diag['cn_max_correction_k']:.3f}K  "
                      f"pct_active={cn_diag['cn_pct_meaningfully_corrected']:.1f}%")
                print(f"  CN cold:    raw_cold_px={cn_diag['n_raw_cold_pixels']:.0f}  "
                      f"lifted={cn_diag['lst_cn_cold_removed_pct']:.1f}%")
                print(f"  lr:         unet={lr_unet:.2e}  cn={lr_cn:.2e}")

                # Update history
                h = self.unet.history
                h.setdefault('train_mse',          []).append(float(train_mse))
                h.setdefault('test_mse',            []).append(float(test_mse))
                h.setdefault('l_main',              []).append(float(avg_l_main))
                h.setdefault('l_cn',                []).append(float(avg_l_cn))
                h.setdefault('l_reg',               []).append(float(avg_l_reg))
                h.setdefault('l_cold',              []).append(float(avg_l_cold))
                h.setdefault('hard_cold_test_pct',  []).append(float(hard_cold_test_pct))
                h.setdefault('hard_cold_train_pct', []).append(float(hard_cold_train_pct))
                h.setdefault('cn_mean_correction_k',[]).append(float(cn_diag['cn_mean_correction_k']))
                h.setdefault('cn_pct_active',       []).append(float(cn_diag['cn_pct_meaningfully_corrected']))
                h.setdefault('cn_cold_removed_pct', []).append(float(cn_diag['lst_cn_cold_removed_pct']))
                h['nr_epochs'] = epoch + 1

                if database_path:
                    self._log_to_db(database_path, epoch, {
                        'train_mse':           train_mse,
                        'test_mse':            test_mse,
                        'ema_test':            ema_test,
                        'ema_ratio':           ema_ratio,
                        'l_main':              avg_l_main,
                        'l_cn':                avg_l_cn,
                        'l_reg':               avg_l_reg,
                        'l_cold':              avg_l_cold,
                        'hard_cold_test_pct':  hard_cold_test_pct,
                        'hard_cold_train_pct': hard_cold_train_pct,
                        'cn_mean_correction_k':cn_diag['cn_mean_correction_k'],
                        'cn_max_correction_k': cn_diag['cn_max_correction_k'],
                        'cn_pct_active':       cn_diag['cn_pct_meaningfully_corrected'],
                        'cn_cold_removed_pct': cn_diag['lst_cn_cold_removed_pct'],
                        'n_raw_cold_pixels':   cn_diag['n_raw_cold_pixels'],
                        'lr_unet':             lr_unet,
                        'lr_cn':               lr_cn,
                    })

                # Best checkpoints
                if ema_test < best_ema_test:
                    best_ema_test = ema_test
                    ckpt_path = os.path.join(model_folder, 'checkpoint_best_test_mse')
                    print(f"  ★ New best ema_test {ema_test:.6f} → saving {ckpt_path}")
                    self._save_checkpoint(
                        ckpt_path, unet_optimizer, self.cn_optimizer,
                        unet_scheduler, cn_scheduler, epoch
                    )

                if ema_ratio > best_ema_ratio:
                    best_ema_ratio = ema_ratio
                    ckpt_path = os.path.join(model_folder, 'checkpoint_best_ratio')
                    print(f"  ★ New best ema_ratio {ema_ratio:.3f} → saving {ckpt_path}")
                    self._save_checkpoint(
                        ckpt_path, unet_optimizer, self.cn_optimizer,
                        unet_scheduler, cn_scheduler, epoch
                    )

                # Save root state every test_interval
                self.unet.history['nr_epochs'] = epoch + 1
                self.unet.save(model_folder)
                self._save_joint_state(
                    os.path.join(model_folder, 'joint_state'),
                    unet_optimizer, self.cn_optimizer,
                    unet_scheduler, cn_scheduler
                )

            # Periodic epoch checkpoint
            if checkpoint_interval and (epoch + 1) % checkpoint_interval == 0:
                ckpt_path = os.path.join(model_folder, f'checkpoint_epoch_{epoch+1}')
                print(f"Saving checkpoint to {ckpt_path}...")
                self._save_checkpoint(
                    ckpt_path, unet_optimizer, self.cn_optimizer,
                    unet_scheduler, cn_scheduler, epoch
                )

        print(f"\nJoint training complete. Model saved to {model_folder}")

    def _unet_forward(self, batch_inputs, device):
        if self.unet.architecture == 'flow_matching':
            B = batch_inputs.shape[0]
            t = torch.ones(B, device=device)
            target_channels = self.unet.output_shape[0]
            x_t = torch.zeros(
                B, target_channels,
                batch_inputs.shape[2], batch_inputs.shape[3],
                device=device
            )
            return self.unet.flow_model(x_t, batch_inputs, t)
        else:
            enc_out, skips = self.unet.encoder(batch_inputs)
            return self.unet.decoder(enc_out, skips)

    def _save_checkpoint(self, ckpt_path, unet_optimizer, cn_optimizer,
                         unet_scheduler, cn_scheduler, epoch):
        self.unet.save(ckpt_path)
        torch.save(unet_optimizer.state_dict(),
                   os.path.join(ckpt_path, 'optimizer.state'))
        torch.save(unet_scheduler.state_dict(),
                   os.path.join(ckpt_path, 'scheduler.state'))
        joint_dir = os.path.join(ckpt_path, 'joint_state')
        self._save_joint_state(joint_dir, unet_optimizer, cn_optimizer,
                               unet_scheduler, cn_scheduler)
        cn_config = {
            'cn_type': 'conv',
            'static_channel_indices': self.cn.static_channel_indices,
            'base_channels': self.cn.net[0].out_channels
                if hasattr(self.cn.net[0], 'out_channels') else 32,
            'cold_threshold_norm': self.cn.cold_threshold_norm,
            'lambda_sparsity': self.cn.lambda_sparsity,
            'lambda_cold': self.lambda_cold,
            'cold_threshold_k': self.cold_threshold_k,
            'cn_lr': self.cn_lr,
            'epoch': epoch,
        }
        with open(os.path.join(ckpt_path, 'joint_config.json'), 'w') as f:
            json.dump(cn_config, f, indent=2)

    def _save_joint_state(self, joint_dir, unet_optimizer, cn_optimizer,
                          unet_scheduler, cn_scheduler):
        os.makedirs(joint_dir, exist_ok=True)
        torch.save(self.cn.state_dict(),
                   os.path.join(joint_dir, 'cn.weights'))
        torch.save(unet_optimizer.state_dict(),
                   os.path.join(joint_dir, 'unet_optimizer.state'))
        torch.save(cn_optimizer.state_dict(),
                   os.path.join(joint_dir, 'cn_optimizer.state'))
        torch.save(unet_scheduler.state_dict(),
                   os.path.join(joint_dir, 'unet_scheduler.state'))
        torch.save(cn_scheduler.state_dict(),
                   os.path.join(joint_dir, 'cn_scheduler.state'))

    def _log_to_db(self, database_path, epoch, metrics):
        import sqlite3
        conn = sqlite3.connect(database_path)
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS joint_training (
            epoch INTEGER,
            train_mse REAL, test_mse REAL, ema_test REAL, ema_ratio REAL,
            l_main REAL, l_cn REAL, l_reg REAL, l_cold REAL,
            hard_cold_test_pct REAL, hard_cold_train_pct REAL,
            cn_mean_correction_k REAL, cn_max_correction_k REAL,
            cn_pct_active REAL, cn_cold_removed_pct REAL,
            n_raw_cold_pixels REAL, lr_unet REAL, lr_cn REAL,
            model_id TEXT, timestamp TEXT
        )''')
        model_id = self.unet.get_model_id() \
            if hasattr(self.unet, 'get_model_id') else 'unknown'
        c.execute('''INSERT INTO joint_training VALUES
            (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)''', (
            epoch,
            metrics['train_mse'],       metrics['test_mse'],
            metrics['ema_test'],        metrics['ema_ratio'],
            metrics['l_main'],          metrics['l_cn'],
            metrics['l_reg'],           metrics['l_cold'],
            metrics['hard_cold_test_pct'], metrics['hard_cold_train_pct'],
            metrics['cn_mean_correction_k'], metrics['cn_max_correction_k'],
            metrics['cn_pct_active'],   metrics['cn_cold_removed_pct'],
            metrics['n_raw_cold_pixels'],
            metrics['lr_unet'],         metrics['lr_cn'],
            model_id,
            time.strftime('%Y-%m-%dT%H:%M:%S')
        ))
        conn.commit()
        conn.close()


# =============================================================================
# Factory functions
# =============================================================================

def build_joint_unet_fresh(unet_kwargs, joint_kwargs):
    unet = UNET(**unet_kwargs)
    return JointUNET(unet, **joint_kwargs)


def load_joint_unet_for_continue(model_folder, checkpoint_subdir=None):
    load_from = os.path.join(model_folder, checkpoint_subdir) \
        if checkpoint_subdir else model_folder

    unet = UNET()
    unet.load(load_from)

    joint_config_path = os.path.join(load_from, 'joint_config.json')
    if not os.path.exists(joint_config_path):
        raise FileNotFoundError(
            f"joint_config.json not found in {load_from}. "
            f"Was this checkpoint saved by joint training?"
        )
    with open(joint_config_path) as f:
        jc = json.load(f)

    joint = JointUNET(
        unet=unet,
        cn_type=jc.get('cn_type', 'conv'),
        cn_base_channels=jc.get('base_channels', 32),
        cn_cold_threshold_norm=jc.get('cold_threshold_norm', 0.3),
        lambda_sparsity=jc.get('lambda_sparsity', 0.01),
        lambda_cold=jc.get('lambda_cold', 0.1),
        cold_threshold_k=jc.get('cold_threshold_k', 10.0),
        cn_lr=jc.get('cn_lr', 0.0001),
    )

    joint_state_dir = os.path.join(load_from, 'joint_state')
    cn_weights_path = os.path.join(joint_state_dir, 'cn.weights')
    if os.path.exists(cn_weights_path):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        joint.cn.load_state_dict(
            torch.load(cn_weights_path, map_location=device)
        )
        print(f"Loaded CN weights from {cn_weights_path}")

    cn_opt_path = os.path.join(joint_state_dir, 'cn_optimizer.state')
    if os.path.exists(cn_opt_path):
        joint.cn._saved_optimizer_state_path = cn_opt_path

    return joint
