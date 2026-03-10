import torch
import sys

p1 = sys.argv[1] if len(sys.argv) > 1 else "/gws/nopw/j04/eocis_chuk/shaerdan/preprocessed_v9/train_v9_conditioned.pt"
p2 = sys.argv[2] if len(sys.argv) > 2 else "/gws/nopw/j04/eocis_chuk/shaerdan/preprocessed_v9/train_v9_conditioned_augmented_v2_uk.pt"

print(f"File 1: {p1.split('/')[-1]}")
print(f"File 2: {p2.split('/')[-1]}")

t1 = torch.load(p1, map_location='cpu')
t2 = torch.load(p2, map_location='cpu')
n1 = t1['normalisation_parameters']
n2 = t2['normalisation_parameters']

print(f"Samples: {t1['n_samples']} vs {t2['n_samples']}")
print()

for k in sorted(n1['min_inputs'].keys()):
    m1, x1 = n1['min_inputs'][k], n1['max_inputs'][k]
    m2, x2 = n2['min_inputs'][k], n2['max_inputs'][k]
    flag = " ***" if abs(m1 - m2) > 1e-4 or abs(x1 - x2) > 1e-4 else ""
    print(f"  {k:>35s}: [{m1:.4f}, {x1:.4f}] vs [{m2:.4f}, {x2:.4f}]{flag}")

m1o, x1o = n1['min_output'], n1['max_output']
m2o, x2o = n2['min_output'], n2['max_output']
flag = " ***" if abs(m1o - m2o) > 1e-4 or abs(x1o - x2o) > 1e-4 else ""
print(f"  {'output':>35s}: [{m1o:.4f}, {x1o:.4f}] vs [{m2o:.4f}, {x2o:.4f}]{flag}")