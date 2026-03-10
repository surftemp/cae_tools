"""Check how many boxes survive at threshold=0 vs threshold=0.5."""
import torch
import numpy as np
import sys

paths = sys.argv[1:] if len(sys.argv) > 1 else [
    "/gws/nopw/j04/eocis_chuk/shaerdan/preprocessed_v9/train_v9_conditioned.pt",
    "/gws/nopw/j04/eocis_chuk/shaerdan/preprocessed_v9/val_v9_conditioned.pt",
]

for path in paths:
    print(f"Loading {path.split('/')[-1]}...")
    data = torch.load(path, map_location='cpu')
    norm = data['normalisation_parameters']

    if 'spatial_variables' in data:
        sp_vars = data['spatial_variables']
        lc_idx = sp_vars.index('land_cover')
        lc = data['spatial_inputs'][:, lc_idx]
    else:
        iv = data['input_variables']
        lc_idx = iv.index('land_cover')
        lc = data['inputs'][:, lc_idx]

    lc = lc * (norm['max_inputs']['land_cover'] - norm['min_inputs']['land_cover']) + norm['min_inputs']['land_cover']
    lc_int = lc.round().long()
    N = lc_int.shape[0]

    has_any_zero = ((lc_int == 0).reshape(N, -1).sum(dim=1) > 0).numpy()
    has_majority_zero = ((lc_int == 0).float().reshape(N, -1).mean(dim=1) > 0.5).numpy()
    all_zero = ((lc_int == 0).reshape(N, -1).all(dim=1)).numpy()
    pure_uk = (~has_any_zero)

    print(f"  Total:                {N:,}")
    print(f"  Pure UK (0% lc=0):    {pure_uk.sum():,} ({100*pure_uk.sum()/N:.1f}%)")
    print(f"  Any lc=0:             {has_any_zero.sum():,} ({100*has_any_zero.sum()/N:.1f}%)")
    print(f"  >50% lc=0:            {has_majority_zero.sum():,}")
    print(f"  100% lc=0:            {all_zero.sum():,}")
    print()
