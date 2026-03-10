"""
Augment winter training data with multiple strategies:

1. CONDITIONING NOISE: Small Gaussian noise on conditioning, LST unchanged.
   Teaches smoothness in conditioning response.

2. COHERENT SHIFT: Shift temperature conditioning AND LST target by same
   offset. Extends conditioning coverage to colder/warmer regimes.

3. DOY PERTURBATION: Applied within both strategies above. Shifts sin_doy
   and cos_doy by ±N days to fill gaps in the DOY coverage.

4. DONOR MONTH TRANSFER: Take boxes from adjacent months (Nov→Dec, Feb→Jan),
   re-encode their DOY into the target month range, and adjust temperature
   conditioning slightly. Brings genuinely different spatial patterns from
   similar winter conditions.

Usage:
  python augment_winter_boxes.py \
      --input-pt /path/to/train.pt \
      --output-pt /path/to/train_augmented.pt \
      --months 12 1 \
      --noise-copies 2 --noise-std 0.015 \
      --shift-copies 2 --shift-down-k 5.0 --shift-up-k 2.0 \
      --doy-jitter-days 15 \
      --donor-copies 1 --donor-temp-shift-std 2.0 \
      --seed 42
"""

import argparse
import torch
import numpy as np
import os
import sys


# ── DOY / month encoding helpers ────────────────────────────────────────

def recover_doy_and_months(cond, cond_vars, norm_params, normalised):
    """Recover DOY and month from sin_doy/cos_doy encoding."""
    sin_idx = cond_vars.index('sin_doy')
    cos_idx = cond_vars.index('cos_doy')

    sin_v = cond[:, sin_idx].numpy().copy()
    cos_v = cond[:, cos_idx].numpy().copy()

    if normalised:
        for name, vals in [('sin_doy', sin_v), ('cos_doy', cos_v)]:
            mn = norm_params['min_inputs'][name]
            mx = norm_params['max_inputs'][name]
            vals[:] = vals * (mx - mn) + mn

    d1 = np.arcsin(np.clip(sin_v, 0, 1)) * 366 / np.pi
    d2 = 366 - d1
    c1 = np.cos((d1 / 366) * 2 * np.pi)
    c2 = np.cos((d2 / 366) * 2 * np.pi)
    doy = np.where(np.abs(c1 - cos_v) < np.abs(c2 - cos_v), d1, d2)
    doy = np.clip(doy, 1, 366).astype(int)

    boundaries = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 366]
    months = np.zeros(len(doy), dtype=int)
    for m in range(12):
        mask = (doy > boundaries[m]) & (doy <= boundaries[m + 1])
        months[mask] = m + 1
    months[months == 0] = 1
    return doy, months


def encode_doy(doy_values, norm_params, normalised):
    """Encode DOY values back to normalised sin_doy, cos_doy."""
    # DatasetLoader encoding:
    #   sin_doy = sin((doy/366) * pi)
    #   cos_doy = cos((doy/366) * 2 * pi)
    sin_phys = np.sin((doy_values / 366.0) * np.pi)
    cos_phys = np.cos((doy_values / 366.0) * 2 * np.pi)

    if normalised:
        sin_mn = norm_params['min_inputs']['sin_doy']
        sin_mx = norm_params['max_inputs']['sin_doy']
        cos_mn = norm_params['min_inputs']['cos_doy']
        cos_mx = norm_params['max_inputs']['cos_doy']
        sin_norm = (sin_phys - sin_mn) / (sin_mx - sin_mn)
        cos_norm = (cos_phys - cos_mn) / (cos_mx - cos_mn)
        return sin_norm.astype(np.float32), cos_norm.astype(np.float32)
    else:
        return sin_phys.astype(np.float32), cos_phys.astype(np.float32)


# Month DOY ranges
MONTH_DOY_RANGES = {
    1:  (1, 31),
    2:  (32, 59),
    3:  (60, 90),
    4:  (91, 120),
    5:  (121, 151),
    6:  (152, 181),
    7:  (182, 212),
    8:  (213, 243),
    9:  (244, 273),
    10: (274, 304),
    11: (305, 334),
    12: (335, 365),
}

# Donor month mapping: which month donates to which target
DONOR_MAP = {
    12: 11,  # November donates to December
    1:  2,   # February donates to January
}

MONTH_NAMES = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']


def main():
    parser = argparse.ArgumentParser(
        description='Augment winter boxes: noise + shift + DOY jitter + donor transfer')
    parser.add_argument('--input-pt', required=True)
    parser.add_argument('--output-pt', required=True)
    parser.add_argument('--months', nargs='+', type=int, default=[12, 1],
                        help='Months to augment (default: 12 1)')

    # Strategy 1: Conditioning noise
    parser.add_argument('--noise-copies', type=int, default=2)
    parser.add_argument('--noise-std', type=float, default=0.015,
                        help='Noise std on normalised conditioning (default: 0.015)')

    # Strategy 2: Coherent temperature shift
    parser.add_argument('--shift-copies', type=int, default=2)
    parser.add_argument('--shift-down-k', type=float, default=5.0,
                        help='Max downward (colder) temperature shift in K (default: 5.0)')
    parser.add_argument('--shift-up-k', type=float, default=2.0,
                        help='Max upward (warmer) temperature shift in K (default: 2.0)')

    # DOY jitter (applied in both strategies 1 and 2)
    parser.add_argument('--doy-jitter-days', type=int, default=15,
                        help='Max DOY jitter in days (default: 15)')

    # Strategy 3: Donor month transfer
    parser.add_argument('--donor-copies', type=int, default=1,
                        help='Copies from donor month per target month (default: 1)')
    parser.add_argument('--donor-temp-shift-std', type=float, default=2.0,
                        help='Std of temperature shift for donors in K (default: 2.0)')
    parser.add_argument('--donor-fraction', type=float, default=1.0,
                        help='Fraction of donor boxes to use (default: 1.0)')

    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # ── Load ────────────────────────────────────────────────────────────
    print(f"Loading {args.input_pt}...")
    data = torch.load(args.input_pt, map_location='cpu')

    spatial = data['spatial_inputs']
    cond = data['conditioning']
    outputs = data['outputs']
    cond_vars = data['cond_variables']
    norm_params = data['normalisation_parameters']
    normalised = data.get('normalized', True)
    output_activation = norm_params.get('output_activation', 'sigmoid')

    N = spatial.shape[0]
    print(f"Total samples: {N}")
    print(f"Cond vars: {cond_vars}")

    # ── Index conditioning channels ─────────────────────────────────────
    temp_vars = {'era5_skt', 'era5_d2m', 'era5_stl1'}
    doy_vars = {'sin_doy', 'cos_doy'}
    other_vars = set(cond_vars) - temp_vars - doy_vars

    temp_indices = [i for i, v in enumerate(cond_vars) if v in temp_vars]
    doy_indices = [i for i, v in enumerate(cond_vars) if v in doy_vars]
    sin_doy_idx = cond_vars.index('sin_doy')
    cos_doy_idx = cond_vars.index('cos_doy')
    other_indices = [i for i, v in enumerate(cond_vars) if v in other_vars]
    noise_indices = [i for i, v in enumerate(cond_vars) if v not in doy_vars]

    # Temperature scales: K per normalised unit
    temp_scales = {}
    for v in temp_vars:
        mn = norm_params['min_inputs'][v]
        mx = norm_params['max_inputs'][v]
        temp_scales[v] = mx - mn

    out_mn = norm_params['min_output']
    out_mx = norm_params['max_output']
    out_scale = out_mx - out_mn

    print(f"\nTemp scales (K/norm unit): "
          + ", ".join(f"{v}={temp_scales[v]:.1f}" for v in temp_vars))
    print(f"Output scale: {out_scale:.1f} K/norm unit")

    # ── Recover DOY and months ──────────────────────────────────────────
    doy, months = recover_doy_and_months(
        cond.clone(), cond_vars, norm_params, normalised)

    print(f"\nOriginal month counts:")
    for m in range(1, 13):
        count = int(np.sum(months == m))
        marker = " <-- target" if m in args.months else ""
        marker = " <-- donor" if m in DONOR_MAP.values() and DONOR_MAP.get(
            {v: k for k, v in DONOR_MAP.items()}.get(m, -1), -1) in args.months else marker
        # simpler:
        is_target = m in args.months
        is_donor = m in [DONOR_MAP[t] for t in args.months if t in DONOR_MAP]
        tag = " <-- target" if is_target else (" <-- donor" if is_donor else "")
        print(f"  {MONTH_NAMES[m-1]:>4s}: {count:6d}{tag}")

    # ── Helper: jitter DOY and re-encode ────────────────────────────────
    def jitter_doy_in_cond(cd, src_doy, target_month, max_jitter):
        """Jitter DOY within target month range and update sin/cos encoding."""
        doy_lo, doy_hi = MONTH_DOY_RANGES[target_month]
        n = cd.shape[0]
        jitter = np.random.randint(-max_jitter, max_jitter + 1, size=n)
        new_doy = np.clip(src_doy + jitter, doy_lo, doy_hi).astype(float)
        sin_new, cos_new = encode_doy(new_doy, norm_params, normalised)
        cd[:, sin_doy_idx] = torch.from_numpy(sin_new)
        cd[:, cos_doy_idx] = torch.from_numpy(cos_new)
        return cd

    # Collect all augmented data
    aug_spatial = []
    aug_cond = []
    aug_outputs = []
    total_added = 0

    for target_month in args.months:
        target_mask = months == target_month
        target_idx = np.where(target_mask)[0]
        n_target = len(target_idx)
        target_doy = doy[target_idx]

        print(f"\n{'='*60}")
        print(f"Augmenting {MONTH_NAMES[target_month-1]}: "
              f"{n_target} source boxes, DOY range "
              f"{target_doy.min()}-{target_doy.max()}")

        # ── Strategy 1: Conditioning noise + DOY jitter ─────────────────
        print(f"\n  Strategy 1: Noise (copies={args.noise_copies}, "
              f"std={args.noise_std}, doy_jitter=±{args.doy_jitter_days}d)")
        for i in range(args.noise_copies):
            sp = spatial[target_idx]
            cd = cond[target_idx].clone()
            out = outputs[target_idx]

            # Noise on non-DOY conditioning
            noise = torch.zeros_like(cd)
            noise[:, noise_indices] = (
                torch.randn(n_target, len(noise_indices)) * args.noise_std)
            cd = cd + noise

            # DOY jitter
            cd = jitter_doy_in_cond(cd, target_doy, target_month,
                                     args.doy_jitter_days)

            cd = cd.clamp(0.0, 1.0)

            aug_spatial.append(sp)
            aug_cond.append(cd)
            aug_outputs.append(out)
            total_added += n_target
            print(f"    Copy {i+1}: +{n_target} boxes")

        # ── Strategy 2: Coherent shift + DOY jitter ─────────────────────
        print(f"\n  Strategy 2: Shift (copies={args.shift_copies}, "
              f"down={args.shift_down_k}K, up={args.shift_up_k}K, "
              f"doy_jitter=±{args.doy_jitter_days}d)")
        for i in range(args.shift_copies):
            sp = spatial[target_idx]
            cd = cond[target_idx].clone()
            out = outputs[target_idx].clone()

            # Random temperature shift per box: uniform in [-down, +up]
            shift_k = (torch.rand(n_target) * (args.shift_down_k + args.shift_up_k)
                       - args.shift_down_k)

            # Shift temperature conditioning channels
            for idx in temp_indices:
                v = cond_vars[idx]
                cd[:, idx] = cd[:, idx] + shift_k / temp_scales[v]

            # Shift output
            if output_activation == 'tanh':
                shift_out = 2 * shift_k / out_scale
            else:
                shift_out = shift_k / out_scale
            out = out + shift_out.reshape(-1, 1, 1, 1)

            # Small noise on non-temperature, non-DOY channels
            if other_indices:
                other_noise = (
                    torch.randn(n_target, len(other_indices)) * args.noise_std * 0.5)
                cd[:, other_indices] = cd[:, other_indices] + other_noise

            # DOY jitter
            cd = jitter_doy_in_cond(cd, target_doy, target_month,
                                     args.doy_jitter_days)

            cd = cd.clamp(0.0, 1.0)
            if output_activation == 'tanh':
                out = out.clamp(-1.0, 1.0)
            else:
                out = out.clamp(0.0, 1.0)

            aug_spatial.append(sp)
            aug_cond.append(cd)
            aug_outputs.append(out)
            total_added += n_target

            sk = shift_k.numpy()
            print(f"    Copy {i+1}: +{n_target} boxes, "
                  f"shift [{sk.min():.1f}, {sk.max():.1f}] K")

        # ── Strategy 3: Donor month transfer ────────────────────────────
        if target_month in DONOR_MAP:
            donor_month = DONOR_MAP[target_month]
            donor_mask = months == donor_month
            donor_idx = np.where(donor_mask)[0]
            n_donor_all = len(donor_idx)

            # Subsample if requested
            n_donor_use = int(n_donor_all * args.donor_fraction)
            if n_donor_use < n_donor_all:
                donor_idx = np.random.choice(
                    donor_idx, size=n_donor_use, replace=False)

            donor_doy = doy[donor_idx]
            n_donor = len(donor_idx)

            print(f"\n  Strategy 3: Donor transfer "
                  f"({MONTH_NAMES[donor_month-1]}→{MONTH_NAMES[target_month-1]}, "
                  f"{n_donor} donors, copies={args.donor_copies})")

            target_doy_lo, target_doy_hi = MONTH_DOY_RANGES[target_month]

            for i in range(args.donor_copies):
                sp = spatial[donor_idx]
                cd = cond[donor_idx].clone()
                out = outputs[donor_idx].clone()

                # Re-encode DOY: map donor DOY into target month range
                # Uniform random within target month
                new_doy = np.random.randint(
                    target_doy_lo, target_doy_hi + 1, size=n_donor).astype(float)
                sin_new, cos_new = encode_doy(new_doy, norm_params, normalised)
                cd[:, sin_doy_idx] = torch.from_numpy(sin_new)
                cd[:, cos_doy_idx] = torch.from_numpy(cos_new)

                # Small temperature shift (donor month is close but not identical)
                temp_shift = (torch.randn(n_donor) *
                              args.donor_temp_shift_std)

                for idx in temp_indices:
                    v = cond_vars[idx]
                    cd[:, idx] = cd[:, idx] + temp_shift / temp_scales[v]

                # Shift output by same amount
                if output_activation == 'tanh':
                    shift_out = 2 * temp_shift / out_scale
                else:
                    shift_out = temp_shift / out_scale
                out = out + shift_out.reshape(-1, 1, 1, 1)

                # Small noise on non-temperature, non-DOY channels
                if other_indices:
                    other_noise = (
                        torch.randn(n_donor, len(other_indices)) *
                        args.noise_std * 0.5)
                    cd[:, other_indices] = cd[:, other_indices] + other_noise

                cd = cd.clamp(0.0, 1.0)
                if output_activation == 'tanh':
                    out = out.clamp(-1.0, 1.0)
                else:
                    out = out.clamp(0.0, 1.0)

                aug_spatial.append(sp)
                aug_cond.append(cd)
                aug_outputs.append(out)
                total_added += n_donor

                ts = temp_shift.numpy()
                print(f"    Copy {i+1}: +{n_donor} boxes from "
                      f"{MONTH_NAMES[donor_month-1]}, "
                      f"temp shift [{ts.min():.1f}, {ts.max():.1f}] K, "
                      f"DOY remapped to [{target_doy_lo}, {target_doy_hi}]")

    # ── Concatenate ─────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    all_spatial = torch.cat([spatial] + aug_spatial, dim=0)
    all_cond = torch.cat([cond] + aug_cond, dim=0)
    all_outputs = torch.cat([outputs] + aug_outputs, dim=0)

    N_new = all_spatial.shape[0]

    print(f"Original:  {N}")
    print(f"Added:     {total_added}")
    print(f"New total: {N_new}")

    # ── Shuffle ─────────────────────────────────────────────────────────
    print("Shuffling...")
    perm = torch.randperm(N_new)
    all_spatial = all_spatial[perm]
    all_cond = all_cond[perm]
    all_outputs = all_outputs[perm]

    # ── Save ────────────────────────────────────────────────────────────
    out_data = {}
    for k, v in data.items():
        if k == 'spatial_inputs':
            out_data[k] = all_spatial
        elif k == 'conditioning':
            out_data[k] = all_cond
        elif k == 'outputs':
            out_data[k] = all_outputs
        elif k == 'n_samples':
            out_data[k] = N_new
        else:
            out_data[k] = v

    out_data['augmentation_meta'] = {
        'method': 'noise + coherent_shift + doy_jitter + donor_transfer',
        'target_months': args.months,
        'noise_copies': args.noise_copies,
        'noise_std': args.noise_std,
        'shift_copies': args.shift_copies,
        'shift_down_k': args.shift_down_k,
        'shift_up_k': args.shift_up_k,
        'doy_jitter_days': args.doy_jitter_days,
        'donor_copies': args.donor_copies,
        'donor_temp_shift_std': args.donor_temp_shift_std,
        'donor_fraction': args.donor_fraction,
        'donor_map': {str(k): v for k, v in DONOR_MAP.items()},
        'n_original': N,
        'n_added': total_added,
        'n_total': N_new,
        'seed': args.seed,
    }

    print(f"\nSaving to {args.output_pt}...")
    os.makedirs(os.path.dirname(args.output_pt) or '.', exist_ok=True)
    torch.save(out_data, args.output_pt)

    size_gb = os.path.getsize(args.output_pt) / 1e9
    print(f"Saved: {N_new} samples, {size_gb:.2f} GB")

    # ── Verify month counts ─────────────────────────────────────────────
    print("\nMonth counts after augmentation:")
    _, new_months = recover_doy_and_months(
        all_cond.clone(), cond_vars, norm_params, normalised)
    for m in range(1, 13):
        count_new = int(np.sum(new_months == m))
        count_old = int(np.sum(months == m))
        diff = count_new - count_old
        print(f"  {MONTH_NAMES[m-1]:>4s}: {count_new:6d} (was {count_old:6d}, +{diff})")

    print("\nDone!")


if __name__ == '__main__':
    main()
