import torch, numpy as np

d = torch.load('/gws/nopw/j04/eocis_chuk/shaerdan/preprocessed_v8/train_v8.pt', map_location='cpu')
inputs = d['inputs']
outputs = d['outputs']
norm = d['normalisation_parameters']

era5_min = norm['min_inputs']['era5_skt']
era5_max = norm['max_inputs']['era5_skt']
out_min = norm['min_output']
out_max = norm['max_output']

N = len(inputs)
print("Original training set: %d boxes" % N)

keep = []
removed_summer = 0
removed_winter = 0
for i in range(N):
    era5_phys = era5_min + inputs[i, 3, :, :].mean().item() * (era5_max - era5_min)
    lst_phys = out_min + outputs[i, 0, :, :].mean().item() * (out_max - out_min)
    delta = lst_phys - era5_phys

    if era5_phys > 288 and delta < -10:
        removed_summer += 1
    elif era5_phys <= 288 and delta < -15:
        removed_winter += 1
    else:
        keep.append(i)

    if (i+1) % 20000 == 0:
        print("  Processed %d/%d (removed %d summer, %d winter)" % (
            i+1, N, removed_summer, removed_winter))

total_removed = removed_summer + removed_winter
print("\nRemoved summer cloud boxes (ERA5>288K, delta<-10K): %d" % removed_summer)
print("Removed winter extreme boxes (ERA5<=288K, delta<-15K): %d" % removed_winter)
print("Total removed: %d (%.2f%%)" % (total_removed, 100*total_removed/N))
print("Keeping: %d boxes" % len(keep))

keep = torch.tensor(keep, dtype=torch.long)

clean_inputs = inputs[keep]
del inputs
clean_outputs = outputs[keep]
del outputs
del d

out_path = '/gws/nopw/j04/eocis_chuk/shaerdan/preprocessed_v8/train_v8_clean.pt'
torch.save({
    'inputs': clean_inputs,
    'outputs': clean_outputs,
    'normalisation_parameters': norm,
}, out_path)
print("Saved to %s" % out_path)

# Clean test set too
import os
test_path = '/gws/nopw/j04/eocis_chuk/shaerdan/preprocessed_v8/test_v8.pt'
if os.path.exists(test_path):
    print("\n=== CLEANING TEST SET ===")
    td = torch.load(test_path, map_location='cpu')
    t_in = td['inputs']
    t_out = td['outputs']
    t_norm = td.get('normalisation_parameters', norm)
    tN = len(t_in)

    t_keep = []
    t_rs = 0
    t_rw = 0
    for i in range(tN):
        era5_phys = era5_min + t_in[i, 3, :, :].mean().item() * (era5_max - era5_min)
        lst_phys = out_min + t_out[i, 0, :, :].mean().item() * (out_max - out_min)
        delta = lst_phys - era5_phys
        if era5_phys > 288 and delta < -10:
            t_rs += 1
        elif era5_phys <= 288 and delta < -15:
            t_rw += 1
        else:
            t_keep.append(i)
        if (i+1) % 5000 == 0:
            print("  Processed %d/%d" % (i+1, tN))

    print("Test: removed %d summer, %d winter, keeping %d" % (t_rs, t_rw, len(t_keep)))

    t_keep = torch.tensor(t_keep, dtype=torch.long)
    clean_t_in = t_in[t_keep]
    del t_in
    clean_t_out = t_out[t_keep]
    del t_out
    del td

    torch.save({
        'inputs': clean_t_in,
        'outputs': clean_t_out,
        'normalisation_parameters': t_norm,
    }, '/gws/nopw/j04/eocis_chuk/shaerdan/preprocessed_v8/test_v8_clean.pt')
    print("Saved cleaned test set")
