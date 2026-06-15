import numpy as np
import uproot

TREE_NAME = "Events"


def _load_dataset_auto(path, tree_name):
    with uproot.open(f"{path}:{tree_name}") as tree:
        scalar_branches = []
        for name in tree.keys():
            try:
                interp = tree[name].interpretation
                dtype  = interp.numpy_dtype
                if dtype.shape == ():
                    scalar_branches.append(name)
            except Exception:
                pass
        df = tree.arrays(scalar_branches, library="pd")

    float_features = df.select_dtypes(include=["float32", "float64"]).columns.tolist()
    int_features   = df.select_dtypes(include=["int32",  "int64",
                                                "uint32", "uint64"]).columns.tolist()

    df[float_features] = df[float_features].astype(np.float64)
    df[int_features]   = df[int_features].astype(np.int64)

    print(f"  Auto-detected {len(float_features)} float features, "
          f"{len(int_features)} int features")
    return df, float_features, int_features


# ── load ──────────────────────────────────────────────────────────────────────
df, float_features, int_features = _load_dataset_auto(
    "../TrainingDataset/training.root", TREE_NAME
)
print(f"    {df.shape[1]} features x {df.shape[0]} events")
print(f"    Number of float features: {len(float_features)}")
print(f"    Number of int features:   {len(int_features)}")

# ── subsample ─────────────────────────────────────────────────────────────────
N_SAMPLE = 100000
if len(df) > N_SAMPLE:
    df = df.sample(n=N_SAMPLE, random_state=42).reset_index(drop=True)
    print(f"    Subsampled to {len(df)} events")

print(f"    {df.shape[1]} features x {df.shape[0]} events")
print(f"    Number of float features: {len(float_features)}")
print(f"    Number of int features:   {len(int_features)}")

# ── write .root ───────────────────────────────────────────────────────────────
out_path = "../TrainingDataset/training_subsampled.root"

# uproot.recreate expects a dict {branch_name: np.ndarray}
branch_data = {col: df[col].to_numpy() for col in df.columns}

with uproot.recreate(out_path) as f:
    f[TREE_NAME] = branch_data

print(f"    Saved subsampled .root → {out_path}")