# Working with OlmoEarth embeddings in Roger Studio

Roger Studio's **Export embeddings as COG** button (visible when a base
encoder is selected in the OlmoEarth Import panel) produces a multi-band
int8 GeoTIFF compatible with Ai2's published embedding format: one band
per embedding dimension, signed 8-bit integers from -127 to +127, with
-128 reserved for nodata.

This document shows how to consume the exported COG for the four
canonical downstream workflows. All recipes run in pure Python
(rasterio + numpy + scikit-learn) against the `.tif` you downloaded.

## The COG format

| Attribute | Value |
|---|---|
| Driver | `COG` (falls back to tiled `GTiff` on older rasterio) |
| Dtype | `int8` |
| Nodata | `-128` |
| Bands | One per embedding dimension (128 = Nano, 192 = Tiny, 768 = Base, 1024 = Large) |
| Pixel size | `target_gsd_m × patch_size` (default 40 m with GSD=10 and patch=4) |
| Tags | `model_repo_id`, `embedding_dim`, `patch_size`, `n_periods`, `period_days`, `quantization` |

To recover float vectors from int8 (required for every downstream task
below), use the AlphaEarth-compatible dequantizer that Ai2 ships:

```python
import rasterio
import torch
import numpy as np
from olmoearth_pretrain.evals.embedding_transforms import dequantize_embeddings

with rasterio.open("allenai_OlmoEarth-v1-Tiny_embedding_*.tif") as ds:
    q = ds.read()             # (D, H, W) int8
    transform = ds.transform  # rasterio.Affine
    crs = ds.crs              # e.g. EPSG:32618 UTM

# Transpose to (H, W, D) and dequantize — produces float32 in the
# approximate range the encoder originally emitted.
q_hwd = np.transpose(q, (1, 2, 0))
emb = dequantize_embeddings(torch.from_numpy(q_hwd)).numpy()  # (H, W, D)

# Mask pixels the backend couldn't cover (edge chunks, skipped periods).
nodata_mask = (q == -128).all(axis=0)  # (H, W) bool
```

Everything below assumes `emb` and `nodata_mask` as produced above.

## Workflow 1: similarity search

"Where else on the map looks like this pixel?"

```python
from numpy.linalg import norm

# Pick a query. Integer row/col in the raster; use rasterio.transform
# to convert from lon/lat if needed.
query_row, query_col = 120, 88
q_vec = emb[query_row, query_col]          # (D,)

# Cosine similarity per pixel.
H, W, D = emb.shape
flat = emb.reshape(-1, D)                  # (H*W, D)
flat_norm = flat / (norm(flat, axis=-1, keepdims=True) + 1e-9)
q_norm = q_vec / (norm(q_vec) + 1e-9)
sim = (flat_norm @ q_norm).reshape(H, W)   # (H, W) in [-1, 1]
sim[nodata_mask] = np.nan

# Save as a single-band float32 raster — open in QGIS, overlay in
# Roger Studio via Import GeoTIFF, or render with matplotlib.
import rasterio
with rasterio.open(
    "similarity.tif", "w",
    driver="GTiff", height=H, width=W, count=1, dtype="float32",
    crs=crs, transform=transform, nodata=np.nan,
) as dst:
    dst.write(sim.astype(np.float32), 1)
```

A **region query** (average embedding over a window) often beats a
single-pixel query because one pixel may be noisy:

```python
win = emb[80:100, 60:80]                   # a 20×20 patch
q_vec = win.reshape(-1, D).mean(axis=0)    # mean-pooled query
# ...compute `sim` the same way as above
```

## Workflow 2: few-shot segmentation

"Train a per-pixel classifier from a handful of labels."

Ai2's Ca Mau mangrove example achieved F1 = 0.84 on three classes with
only 60 total labels (20 per class). A logistic regression over the
192-dim Tiny embeddings is enough — the encoder already organized the
ecological distinctions during pretraining.

```python
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

H, W, D = emb.shape
X_all = emb.reshape(-1, D)

# Your labels. Shape (N, 3) = (row, col, class_id). Collected by
# clicking on the map, reading a reference raster, or any other method.
# Example — 20 points per class, 3 classes:
train_points = np.array([
    [100, 45, 0],   # class 0 pixels
    [102, 48, 0],
    # ... 18 more class-0 rows ...
    [50, 120, 1],   # class 1 pixels
    # ...
    [200, 30, 2],   # class 2 pixels
    # ...
])
train_idx = train_points[:, 0] * W + train_points[:, 1]
train_labels = train_points[:, 2]

clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
clf.fit(X_all[train_idx], train_labels)
prediction = clf.predict(X_all).reshape(H, W).astype(np.int16)
prediction[nodata_mask] = -1               # preserve nodata

# Save as a single-band int16 class raster.
with rasterio.open(
    "segmentation.tif", "w",
    driver="GTiff", height=H, width=W, count=1, dtype="int16",
    crs=crs, transform=transform, nodata=-1,
) as dst:
    dst.write(prediction, 1)
```

**Sample labels from an existing raster** (e.g. ESA WorldCover) for
automatic training without manual clicking:

```python
# Load a reference label raster aligned to the same CRS / grid.
with rasterio.open("worldcover_aligned.tif") as ref:
    labels_raster = ref.read(1)            # (H, W) class ids

# Pick 20 random pixels per class.
rng = np.random.default_rng(42)
train_idx = []
train_labels = []
for class_id in np.unique(labels_raster[~nodata_mask]):
    candidates = np.argwhere(labels_raster == class_id)
    pick = rng.choice(len(candidates), size=20, replace=False)
    for r, c in candidates[pick]:
        train_idx.append(r * W + c)
        train_labels.append(class_id)
# ... fit/predict as above
```

## Workflow 3: change detection

"Where did the landscape shift between two dates?"

Export embeddings twice with different `date_range` values, then
compute per-pixel cosine distance. The Park Fire burn scar example from
the Ai2 article is this exact recipe.

```python
import rasterio
from numpy.linalg import norm
from olmoearth_pretrain.evals.embedding_transforms import dequantize_embeddings
import torch

def load_emb(path: str):
    with rasterio.open(path) as ds:
        q = ds.read()  # (D, H, W) int8
    q_hwd = np.transpose(q, (1, 2, 0))
    emb = dequantize_embeddings(torch.from_numpy(q_hwd)).numpy()
    mask = (q == -128).all(axis=0)
    return emb, mask

emb_t1, mask_t1 = load_emb("park_fire_2023_09.tif")    # before
emb_t2, mask_t2 = load_emb("park_fire_2024_09.tif")    # after

# Cosine distance per pixel.
n1 = emb_t1 / (norm(emb_t1, axis=-1, keepdims=True) + 1e-9)
n2 = emb_t2 / (norm(emb_t2, axis=-1, keepdims=True) + 1e-9)
cos_sim = (n1 * n2).sum(axis=-1)                       # (H, W) in [-1, 1]
change = 1.0 - cos_sim                                 # 0 = identical, 2 = opposite
change[mask_t1 | mask_t2] = np.nan                     # nodata in either

with rasterio.open(
    "change.tif", "w",
    driver="GTiff", height=emb_t1.shape[0], width=emb_t1.shape[1],
    count=1, dtype="float32", nodata=np.nan,
    crs=crs, transform=transform,
) as dst:
    dst.write(change.astype(np.float32), 1)
```

**Threshold for a binary change mask**:

```python
# Pick a threshold empirically — typically 0.05–0.2 captures real change
# while rejecting seasonal / illumination noise.
change_mask = (change > 0.10).astype(np.uint8)
```

## Workflow 4: PCA false-color

"What structure is in the embedding? Show me."

Reduce the D-dim embedding to 3 dimensions, map to R/G/B, render.
Similar embeddings get similar colors automatically — agricultural
parcels, urban cores, water bodies each pick up distinct hues without
any labels.

```python
from sklearn.decomposition import PCA

H, W, D = emb.shape
flat = emb[~nodata_mask]                   # (N_valid, D)

pca = PCA(n_components=3)
pcs = pca.fit_transform(flat)              # (N_valid, 3)

# Rescale each component to [0, 255] independently.
lo = pcs.min(axis=0)
hi = pcs.max(axis=0)
pcs_u8 = ((pcs - lo) / (hi - lo + 1e-9) * 255).astype(np.uint8)

# Scatter back into an (H, W, 3) canvas.
rgb = np.zeros((H, W, 3), dtype=np.uint8)
rgb[~nodata_mask] = pcs_u8

# Save as a 3-band uint8 GeoTIFF — opens as RGB in any viewer.
with rasterio.open(
    "pca_rgb.tif", "w",
    driver="GTiff", height=H, width=W, count=3, dtype="uint8",
    crs=crs, transform=transform,
    photometric="rgb",
) as dst:
    for b in range(3):
        dst.write(rgb[:, :, b], b + 1)

print("Explained variance:", pca.explained_variance_ratio_.sum())
```

The Flevoland polder grid in Ai2's article was produced with exactly
this recipe on a Tiny-dim embedding at 40 m/pixel.

## Tips and gotchas

**Embedding dim vs encoder size**: larger embeddings capture finer
distinctions but cost more disk + more compute in downstream tasks. Tiny
(192) is enough for most land-cover / land-use work; Base (768) helps
when you need to separate many closely-related classes (crop types,
forest sub-types).

**Pixel size trade-off**: the embedding COG's pixel size is
`target_gsd_m × patch_size`. Default is 10 m GSD × 4 patch = **40 m**.
Use `target_gsd_m=10, patch_size=1` for a ~10 m embedding (much bigger
file, slower compute, usually unnecessary for regional analysis).

**Temporal context**: `n_periods=12` with `period_days=30` gives a
yearly temporal stack — good for stable features like land cover. Use
fewer periods (1–3) for event detection (fires, floods) where you care
about recent conditions.

**OOD regions**: the embeddings work anywhere on Earth — the encoder
pretraining was global — but downstream classifiers trained on one
region may not transfer. Always validate on held-out pixels from the
same region.

**nodata propagation**: edge pixels of your AOI and periods that lacked
usable S2 scenes show up as `-128` across all bands. Always mask before
computing similarity / PCA / classification, and set the output raster's
nodata value accordingly so GIS tools render them transparent.

## References

- [Ai2 blog post](https://allenai.org/blog/olmoearth-embeddings) — the feature in Studio
- `olmoearth_pretrain.evals.embedding_transforms` — quantization source
- [Ai2 embeddings tutorial notebook](https://github.com/allenai/olmoearth_pretrain) — reference implementations
- Roger Studio `README.md` → "Performance and limitations" — timing and region notes
