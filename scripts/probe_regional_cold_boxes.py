"""
Probe cold boxes (<280K) across UK regions on Aug 4, 2023.
For each region that contains cold boxes, pick one cold box and one nearby warm box,
print their inputs side-by-side, and flag any inputs outside training range.
"""
import xarray as xr
import numpy as np
import os

# Training normalisation ranges (from model_pB_BxW3tfkE)
TRAIN_MIN = {
    'land_cover': 0.0, 'albedo_monthly_climatology_means': -0.0128,
    'elevation': -45.04, 'era5_skt': 269.325, 'sin_doy': 0.0257,
    'cos_doy': -1.0, 'slope_magnitude': 0.0, 'slope_direction': -180.0,
    'urban_area': 0.0, 'suburban_area': 0.0,
    'pixel_st_hot_pattern': 272.74, 'pixel_st_cold_pattern': 251.38
}
TRAIN_MAX = {
    'land_cover': 21.0, 'albedo_monthly_climatology_means': 3.0675,
    'elevation': 1263.76, 'era5_skt': 312.968, 'sin_doy': 1.0,
    'cos_doy': 0.999, 'slope_magnitude': 1.928, 'slope_direction': 180.0,
    'urban_area': 1.0, 'suburban_area': 1.0,
    'pixel_st_hot_pattern': 341.23, 'pixel_st_cold_pattern': 309.96
}

model_vars = [
    'land_cover', 'albedo_monthly_climatology_means', 'elevation',
    'era5_skt', 'sin_doy', 'cos_doy', 'slope_magnitude', 'slope_direction',
    'urban_area', 'suburban_area', 'pixel_st_hot_pattern', 'pixel_st_cold_pattern'
]

# UK regions in OSGB (approximate)
REGIONS = {
    'SE England':      {'x': (400000, 700000), 'y': (  50000, 250000)},
    'East Anglia':     {'x': (500000, 700000), 'y': (250000, 400000)},
    'Midlands':        {'x': (300000, 500000), 'y': (250000, 400000)},
    'SW England':      {'x': ( 50000, 350000), 'y': (  0,    200000)},
    'Wales':           {'x': (150000, 350000), 'y': (200000, 400000)},
    'N England':       {'x': (300000, 550000), 'y': (400000, 620000)},
    'S Scotland':      {'x': (200000, 450000), 'y': (620000, 800000)},
    'N Scotland':      {'x': (100000, 450000), 'y': (800000,1200000)},
}

in_dir = '/gws/nopw/j04/eocis_chuk/shaerdan/models/scores/debug_inputs/2023-08-04'
out_dir = '/gws/nopw/j04/eocis_chuk/shaerdan/models/scores/debug_outputs/2023-08-04'

# --- Pass 1: Collect ALL boxes with their metadata ---
print("=== PASS 1: Collecting all boxes ===")
all_boxes = []  # list of dicts
for f in sorted(os.listdir(out_dir)):
    di = xr.open_dataset(os.path.join(in_dir, f))
    do = xr.open_dataset(os.path.join(out_dir, f))
    est = do['model_output'].values
    for i in range(est.shape[0]):
        m = float(np.nanmean(est[i]))
        all_boxes.append({
            'file': f, 'idx': i,
            'est': m,
            'skt': float(di.era5_skt.values[i]),
            'x': float(di.SW_corner_x.values[i]),
            'y': float(di.SW_corner_y.values[i]),
        })
    di.close(); do.close()

total = len(all_boxes)
cold = [b for b in all_boxes if b['est'] < 280]
warm = [b for b in all_boxes if b['est'] >= 290]
print(f"Total boxes: {total}, Cold (<280K): {len(cold)}, Warm (>=290K): {len(warm)}")

# --- Pass 2: Bin cold boxes by region ---
print("\n=== COLD BOX DISTRIBUTION BY REGION ===")
region_cold = {}
region_warm = {}
for name, bounds in REGIONS.items():
    xlo, xhi = bounds['x']
    ylo, yhi = bounds['y']
    rc = [b for b in cold if xlo <= b['x'] < xhi and ylo <= b['y'] < yhi]
    rw = [b for b in warm if xlo <= b['x'] < xhi and ylo <= b['y'] < yhi]
    region_cold[name] = rc
    region_warm[name] = rw
    pct = f"({100*len(rc)/len(cold):.1f}%)" if cold else ""
    # count total boxes in region for rate
    rt = [b for b in all_boxes if xlo <= b['x'] < xhi and ylo <= b['y'] < yhi]
    rate = f"{100*len(rc)/len(rt):.1f}%" if rt else "N/A"
    print(f"  {name:15s}: {len(rc):4d} cold / {len(rt):5d} total boxes  (cold rate: {rate})  warm: {len(rw)}")

# --- Pass 3: For each region with cold boxes, pick one cold + one nearby warm, print details ---
print("\n" + "="*100)
print("=== REGIONAL COLD vs WARM BOX COMPARISON ===")
print("="*100)

def get_region(x, y):
    for name, bounds in REGIONS.items():
        xlo, xhi = bounds['x']
        ylo, yhi = bounds['y']
        if xlo <= x < xhi and ylo <= y < yhi:
            return name
    return 'Other'

def print_box_detail(label, box_info, in_dir, out_dir):
    """Print input details for a single box and flag out-of-range inputs."""
    f = box_info['file']
    idx = box_info['idx']
    di = xr.open_dataset(os.path.join(in_dir, f))
    do = xr.open_dataset(os.path.join(out_dir, f))

    print(f"  {label}")
    print(f"    File: {f}, box idx: {idx}")
    print(f"    Location: ({box_info['x']:.0f}, {box_info['y']:.0f})  Region: {get_region(box_info['x'], box_info['y'])}")
    print(f"    Model est mean: {box_info['est']:.1f}K   ERA5 skt: {box_info['skt']:.1f}K   Delta: {box_info['est']-box_info['skt']:.1f}K")

    oor_count = 0
    for v in model_vars:
        d = di[v].values
        if len(d.shape) == 1:
            val = d[idx]
            lo, hi = TRAIN_MIN[v], TRAIN_MAX[v]
            flag = ""
            if val < lo: flag = f" *** BELOW TRAIN MIN ({lo:.2f}) ***"; oor_count += 1
            if val > hi: flag = f" *** ABOVE TRAIN MAX ({hi:.2f}) ***"; oor_count += 1
            print(f"    {v:45s}  value={val:.4f}{flag}")
        else:
            arr = d[idx]
            vmin, vmax, vmean = float(np.nanmin(arr)), float(np.nanmax(arr)), float(np.nanmean(arr))
            lo, hi = TRAIN_MIN[v], TRAIN_MAX[v]
            flag = ""
            if vmin < lo: flag += f" *** MIN BELOW TRAIN ({lo:.2f}) ***"; oor_count += 1
            if vmax > hi: flag += f" *** MAX ABOVE TRAIN ({hi:.2f}) ***"; oor_count += 1
            # Count out-of-range pixels
            below = int(np.sum(arr < lo))
            above = int(np.sum(arr > hi))
            oor_pix = ""
            if below > 0 or above > 0:
                oor_pix = f"  [OOR pixels: {below} below, {above} above]"
            print(f"    {v:45s}  min={vmin:.4f}  max={vmax:.4f}  mean={vmean:.4f}{flag}{oor_pix}")

    # Also print land cover class distribution for this box
    lc = di['land_cover'].values[idx].flatten() if len(di['land_cover'].values.shape) > 1 else di['land_cover'].values[idx]
    if hasattr(lc, 'flatten'):
        lc = lc.flatten()
    lc_int = lc.astype(int)
    classes, counts = np.unique(lc_int, return_counts=True)
    top5 = sorted(zip(classes, counts), key=lambda x: -x[1])[:5]
    dist_str = ", ".join([f"class {c}: {n}" for c, n in top5])
    print(f"    {'LC class distribution (top 5)':45s}  {dist_str}")
    if oor_count > 0:
        print(f"    >>> {oor_count} OUT-OF-RANGE inputs detected <<<")

    di.close(); do.close()

for name in REGIONS:
    rc = region_cold[name]
    rw = region_warm[name]
    if not rc:
        continue

    print(f"\n--- {name} ---")
    # Pick the median cold box (not the worst, more representative)
    rc_sorted = sorted(rc, key=lambda b: b['est'])
    cold_pick = rc_sorted[len(rc_sorted)//2]

    # Find closest warm box in same region
    if rw:
        # Pick warm box closest geographically to the cold one
        def dist(b):
            return ((b['x']-cold_pick['x'])**2 + (b['y']-cold_pick['y'])**2)**0.5
        warm_pick = min(rw, key=dist)
    else:
        # No warm box in region - find nearest warm box anywhere
        warm_pick = min(warm, key=lambda b: ((b['x']-cold_pick['x'])**2 + (b['y']-cold_pick['y'])**2)**0.5)
        print(f"  (no warm boxes in region, using nearest from {get_region(warm_pick['x'], warm_pick['y'])})")

    print_box_detail("COLD BOX (median for region)", cold_pick, in_dir, out_dir)
    print_box_detail("WARM BOX (nearest in region)", warm_pick, in_dir, out_dir)

# --- Summary: out-of-range statistics across ALL cold boxes ---
print("\n" + "="*100)
print("=== OUT-OF-RANGE INPUT ANALYSIS (all cold boxes) ===")
print("="*100)
# Sample up to 100 cold boxes for efficiency
sample_cold = cold if len(cold) <= 100 else [cold[i] for i in np.linspace(0, len(cold)-1, 100, dtype=int)]
oor_stats = {v: {'below': 0, 'above': 0, 'total_pixels': 0} for v in model_vars}

for b in sample_cold:
    di = xr.open_dataset(os.path.join(in_dir, b['file']))
    for v in model_vars:
        d = di[v].values
        lo, hi = TRAIN_MIN[v], TRAIN_MAX[v]
        if len(d.shape) == 1:
            val = d[b['idx']]
            oor_stats[v]['total_pixels'] += 1
            if val < lo: oor_stats[v]['below'] += 1
            if val > hi: oor_stats[v]['above'] += 1
        else:
            arr = d[b['idx']]
            n_pix = arr.size
            oor_stats[v]['total_pixels'] += n_pix
            oor_stats[v]['below'] += int(np.sum(arr < lo))
            oor_stats[v]['above'] += int(np.sum(arr > hi))
    di.close()

print(f"Sampled {len(sample_cold)} cold boxes:")
for v in model_vars:
    s = oor_stats[v]
    tot = s['total_pixels']
    below_pct = 100*s['below']/tot if tot else 0
    above_pct = 100*s['above']/tot if tot else 0
    flag = " <<<" if (below_pct > 0.1 or above_pct > 0.1) else ""
    print(f"  {v:45s}  below_min: {s['below']:6d} ({below_pct:.2f}%)  above_max: {s['above']:6d} ({above_pct:.2f}%){flag}")
