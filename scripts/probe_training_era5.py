import torch, numpy as np

print("Loading training data (just metadata, not spatial arrays)...", flush=True)
data = torch.load('/gws/nopw/j04/eocis_chuk/shaerdan/preprocessed_v8/train_v8.pt', 
                   map_location='cpu', weights_only=False)

print(f"Type: {type(data)}", flush=True)
if isinstance(data, dict):
    print(f"Keys: {list(data.keys())}", flush=True)
    for k, v in data.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: shape={v.shape}, dtype={v.dtype}", flush=True)
        elif isinstance(v, np.ndarray):
            print(f"  {k}: shape={v.shape}, dtype={v.dtype}", flush=True)
        else:
            print(f"  {k}: type={type(v)}, len={len(v) if hasattr(v,'__len__') else 'N/A'}", flush=True)
elif isinstance(data, (list, tuple)):
    print(f"Length: {len(data)}", flush=True)
    for i, item in enumerate(data[:5]):
        if isinstance(item, torch.Tensor):
            print(f"  [{i}]: shape={item.shape}, dtype={item.dtype}", flush=True)
        else:
            print(f"  [{i}]: type={type(item)}", flush=True)
elif isinstance(data, torch.Tensor):
    print(f"Shape: {data.shape}, dtype={data.dtype}", flush=True)

print("Done.", flush=True)
