"""
Smooth seam lines on a no-overlap (stride=100) stitched LST .nc file.

Identifies tile boundaries at every 100th row/column and applies a narrow
Gaussian blur ONLY along those seam lines. All other pixels are untouched.

Usage:
    python smooth_seams.py input.nc output.nc [--variable lst_model_estimate]
                                               [--stride 100]
                                               [--seam-width 3]
                                               [--sigma 1.5]
"""

import argparse
import numpy as np
import xarray as xr
from scipy.ndimage import gaussian_filter


def smooth_seams(data, stride=100, seam_width=3, sigma=1.5):
    """
    Smooth seam lines in a 2D array.

    Args:
        data: 2D numpy array (y, x)
        stride: tile stride in pixels (seams at every stride-th row/col)
        seam_width: half-width of the seam strip in pixels (total = 2*width+1)
        sigma: Gaussian sigma for smoothing

    Returns:
        2D array with seams smoothed, all other pixels unchanged
    """
    H, W = data.shape
    result = data.copy()

    # Full Gaussian-smoothed version
    # Only compute where needed but easier to smooth the whole thing
    # and selectively copy back
    smoothed = gaussian_filter(data.astype(np.float64), sigma=sigma)

    # Build seam mask: True where we want to apply smoothing
    seam_mask = np.zeros((H, W), dtype=bool)

    # Horizontal seams (row boundaries)
    for row in range(stride, H, stride):
        r_lo = max(0, row - seam_width)
        r_hi = min(H, row + seam_width + 1)
        seam_mask[r_lo:r_hi, :] = True

    # Vertical seams (column boundaries)
    for col in range(stride, W, stride):
        c_lo = max(0, col - seam_width)
        c_hi = min(W, col + seam_width + 1)
        seam_mask[:, c_lo:c_hi] = True

    # Only replace seam pixels, preserve NaN
    valid = ~np.isnan(data)
    replace = seam_mask & valid
    result[replace] = smoothed[replace]

    n_seam = int(replace.sum())
    n_total = int(valid.sum())
    print(f"  Smoothed {n_seam:,} seam pixels out of {n_total:,} valid "
          f"({100*n_seam/max(n_total,1):.1f}%)")

    return result


def main():
    parser = argparse.ArgumentParser(
        description='Smooth seam lines on no-overlap stitched LST')
    parser.add_argument('input_nc', help='Input .nc file (stride=100 stitched)')
    parser.add_argument('output_nc', help='Output .nc file with smoothed seams')
    parser.add_argument('--variable', default='lst_model_estimate',
                        help='Variable to smooth (default: lst_model_estimate)')
    parser.add_argument('--stride', type=int, default=100,
                        help='Tile stride in pixels (default: 100)')
    parser.add_argument('--seam-width', type=int, default=3,
                        help='Half-width of seam strip in pixels (default: 3)')
    parser.add_argument('--sigma', type=float, default=1.5,
                        help='Gaussian sigma for smoothing (default: 1.5)')

    args = parser.parse_args()

    print(f"Loading {args.input_nc}...")
    ds = xr.open_dataset(args.input_nc, drop_variables=['time_bnds'])

    if args.variable not in ds:
        print(f"ERROR: Variable '{args.variable}' not found. "
              f"Available: {list(ds.data_vars)}")
        return

    data = ds[args.variable].values
    print(f"  Shape: {data.shape}")
    print(f"  Stride: {args.stride}, seam_width: {args.seam_width}, "
          f"sigma: {args.sigma}")

    # Handle different dimensionalities: (y, x) or (1, y, x) etc
    squeeze_dims = []
    arr = data
    while arr.ndim > 2:
        squeeze_dims.append(arr.shape[0])
        arr = arr[0]

    print(f"  Processing 2D array of shape {arr.shape}...")
    smoothed = smooth_seams(arr, stride=args.stride,
                            seam_width=args.seam_width, sigma=args.sigma)

    # Reconstruct original shape
    result = smoothed
    for dim_size in reversed(squeeze_dims):
        result = result[np.newaxis]

    # Write back
    ds_out = ds.copy()
    ds_out[args.variable].values = result

    print(f"Saving to {args.output_nc}...")
    ds_out.to_netcdf(args.output_nc)
    print("Done.")


if __name__ == '__main__':
    main()
