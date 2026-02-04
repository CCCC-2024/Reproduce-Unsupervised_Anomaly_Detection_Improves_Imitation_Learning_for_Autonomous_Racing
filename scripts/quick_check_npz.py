import numpy as np

path = "data/processed/raindrop_mix.npz"
##path = "data/splits/raindrop_mix_idx.npz"
d = np.load(path, allow_pickle=True)

print("keys:", d.files)
for k in d.files:
    v = d[k]
    try:
        print(k, v.shape, v.dtype)
    except Exception:
        print(k, type(v))
