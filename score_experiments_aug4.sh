#!/bin/bash
# =============================================================================
# Score experimental models on Aug 4 debug inputs and count cold boxes
# Run from anywhere on JASMIN. Requires pyt_cae_tools conda env.
#
# Usage: bash score_experiments_aug4.sh
# =============================================================================

set -e

IN_DIR="/gws/nopw/j04/eocis_chuk/shaerdan/models/scores/debug_inputs/2023-08-04"
BASE_OUT="/gws/nopw/j04/eocis_chuk/shaerdan/models/scores/debug_outputs/2023-08-04"

# Models to test
declare -A MODELS
MODELS["baseline_fc_attn"]="/gws/nopw/j04/eocis_chuk/shaerdan/models/model_pB_BxW3tfkE/checkpoint_epoch_2500"
MODELS["conv_attn"]="/gws/nopw/j04/eocis_chuk/shaerdan/models/model_pB_conv_Qj2cbAOW/checkpoint_epoch_1000"
MODELS["conv_noattn"]="/gws/nopw/j04/eocis_chuk/shaerdan/models/model_pB_conv_noattn_UqsksJ2l/checkpoint_epoch_1000"

for name in conv_attn conv_noattn; do
    model_path="${MODELS[$name]}"
    out_dir="${BASE_OUT}_${name}"

    echo "============================================"
    echo "Scoring: ${name}"
    echo "Model:   ${model_path}"
    echo "Output:  ${out_dir}"
    echo "============================================"

    mkdir -p "${out_dir}"

    for path in ${IN_DIR}/*.nc; do
        file=$(basename "$path")
        if [ ! -f "${out_dir}/${file}" ]; then
            echo "  Scoring ${file}..."
            python /home/users/shaerdan/cae_tools_pB/src/cae_tools/cli/apply_cae.py "${path}" "${out_dir}/${file}" --model-folder "${model_path}"
        else
            echo "  Skipping ${file} (already exists)"
        fi
    done

    echo "Done scoring ${name}"
    echo ""
done

echo "============================================"
echo "All scoring complete. Running cold box count..."
echo "============================================"

python3 -c "
import xarray as xr
import numpy as np
import os

in_dir = '${IN_DIR}'

models = {
    'baseline (fc+attn, ep2500)': '${BASE_OUT}',
    'conv+attn (ep1000)':         '${BASE_OUT}_conv_attn',
    'conv+noattn (ep1000)':       '${BASE_OUT}_conv_noattn',
}

for label, out_dir in models.items():
    if not os.path.isdir(out_dir):
        print(f'{label}: OUTPUT DIR NOT FOUND - skipping')
        continue

    total = 0
    cold = 0
    min_est = 999
    max_delta = 0
    est_vals = []

    for f in sorted(os.listdir(out_dir)):
        if not f.endswith('.nc'):
            continue
        try:
            di = xr.open_dataset(os.path.join(in_dir, f))
            do = xr.open_dataset(os.path.join(out_dir, f))
        except:
            continue
        est = do['model_output'].values
        skt = di['era5_skt'].values
        n = est.shape[0]
        for i in range(n):
            m = float(np.nanmean(est[i]))
            total += 1
            est_vals.append(m)
            if m < 280:
                cold += 1
                delta = float(skt[i]) - m
                if delta > max_delta:
                    max_delta = delta
            if m < min_est:
                min_est = m
        di.close(); do.close()

    est_arr = np.array(est_vals)
    print(f'')
    print(f'  {label}')
    print(f'    Total boxes:  {total}')
    print(f'    Cold (<280K): {cold}  ({100*cold/total:.1f}%)')
    print(f'    Est range:    [{min_est:.1f}K, {est_arr.max():.1f}K]')
    print(f'    Est mean:     {est_arr.mean():.1f}K')
    print(f'    Worst delta:  -{max_delta:.1f}K')
    # Temperature distribution buckets
    bins = [0, 260, 270, 280, 290, 300, 310, 320, 330, 400]
    hist, _ = np.histogram(est_arr, bins=bins)
    print(f'    Distribution:')
    for j in range(len(bins)-1):
        if hist[j] > 0:
            print(f'      {bins[j]:3d}-{bins[j+1]:3d}K: {hist[j]:6d}  ({100*hist[j]/total:.1f}%)')
"
