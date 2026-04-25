# FT classification → GeoJSON polygon export

Roger Studio's `Download as GeoJSON` button (visible on the **Run
inference** tab in the OlmoEarth Import panel for any classification or
segmentation FT head) vectorises the predicted class raster into
polygons readable by Google Earth Pro, QGIS, ArcGIS, leaflet, and any
other tool that speaks `application/geo+json`.

This document covers what the endpoint does, what the output looks
like, when it errors, and how to consume the result.

## Endpoint

`POST /api/olmoearth/ft-classification/geojson`

```json
{
  "bbox": {"west": -81.43, "south": 25.13, "east": -81.40, "north": 25.16},
  "model_repo_id": "allenai/OlmoEarth-v1-FT-Mangrove-Base",
  "date_range": "2024-04-01/2024-10-01",        // optional
  "event_date": "2022-08-15",                    // pre/post heads only
  "min_pixels": 4,                                // default 4 (~160 m² @ 10 m GSD)
  "simplify_tolerance_m": 5.0                     // default 5 m, set 0 to disable
}
```

Response: `application/geo+json` with `Content-Disposition: attachment;
filename="<head-tag>_<job-id-prefix>.geojson"` so the browser triggers
a download.

The response also carries:

* `X-Feature-Count` — number of polygons in the FeatureCollection
* `X-Job-Id` — the underlying inference job (handy for stitching back
  to the same map tile)

## What gets produced

```jsonc
{
  "type": "FeatureCollection",
  "features": [
    {
      "type": "Feature",
      "geometry": {
        "type": "Polygon",
        "coordinates": [[
          [-81.4262, 25.1418],     // WGS-84, RFC 7946 order: [lon, lat]
          [-81.4259, 25.1418],
          …
        ]]
      },
      "properties": {
        "class_id": 1,             // matches legend.classes[*].index
        "class_name": "mangrove",
        "color": "#94eb63",        // published rslearn colour
        "area_m2": 4823.5,         // measured in pinned-CRS metres
        "pixel_count": 48          // raw class-raster cells
      }
    },
    …
  ],
  "properties": {                  // top-level provenance
    "model_repo_id": "allenai/OlmoEarth-v1-FT-Mangrove-Base",
    "scene_id": "S2A_MSIL2A_20240801T155831_R097_T17RMM_…",
    "scene_datetime": "2024-08-01T15:58:31Z",
    "task_type": "segmentation",
    "job_id": "f4a2…",
    "min_pixels": 4,
    "simplify_tolerance_m": 5.0
  }
}
```

* **One polygon per contiguous class region.** Adjacent same-class
  pixels merge; class-boundary pixels become polygon edges.
* **Coordinates are WGS-84** (`[lon, lat]`), RFC 7946 compliant. The
  pinned CRS is reprojected via `rasterio.warp.transform_geom` before
  serialisation.
* **`area_m2`** is measured in the inference job's pinned-CRS metres
  (typically a UTM zone), which is what you'd want for actual area
  computations — `(lon, lat)` coordinates would mis-measure by the
  cosine of the latitude.
* **`color`** is the head's *published* class colour from rslearn's
  `olmoearth_run.yaml` legend. For few-shot classifications the
  colour comes from the user's class palette.

## Which heads are supported

Any head whose `task_type` is `classification` or `segmentation`:

| Head | Task type | Class count | Notes |
|---|---|---|---|
| `OlmoEarth-v1-FT-Mangrove-Base` | segmentation | 4 | Per-pixel: nodata / mangrove / water / other |
| `OlmoEarth-v1-FT-AWF-Base` | segmentation | 10 | Southern-Kenya land use |
| `OlmoEarth-v1-FT-EcosystemTypeMapping-Base` | segmentation | 110 | IUCN ecosystem level 3 |
| `OlmoEarth-v1-FT-ForestLossDriver-Base` | classification | 10 | Pre/post — pass `event_date` |
| Few-shot (`/embedding-tools/few-shot`) | classification | user-defined | Pipes through `/ft-classification/geojson` since the result has the same shape |
| `OlmoEarth-v1-FT-LFMC-Base` | regression | n/a | **Rejected with 400** — see below |

LFMC is regression — there's no class raster to vectorise. The
endpoint returns `400` with an actionable detail:

```
Model 'allenai/OlmoEarth-v1-FT-LFMC-Base' returned task_type='regression',
not a classification. GeoJSON vectorisation only supports classification
heads (Mangrove, AWF, Ecosystem, etc.). For regression (LFMC) or
embeddings, use Export-as-COG.
```

## Tuning knobs

### `min_pixels` — drop speckle

Default `4` ≈ 160 m² at 10 m GSD. Below this the polygon count
balloons and the file becomes hard to view. Set `0` to keep every
single-pixel region (useful for debugging the model's edge behaviour).

```bash
# Strict — keep only contiguous regions of ≥1 ha
... -d '{"min_pixels": 100, ...}'
# Permissive — keep speckle for diagnostic plots
... -d '{"min_pixels": 0, ...}'
```

### `simplify_tolerance_m` — Douglas–Peucker

Default `5.0` m ≈ half the S2 GSD. Visually identical to the raw
polygon, often 5–10× smaller GeoJSON. Set `0` to disable simplification
when you need every vertex (e.g. for binary diff against a ground-truth
GeoJSON).

```bash
# Disable simplification — keep every original vertex
... -d '{"simplify_tolerance_m": 0, ...}'
# Aggressive simplification — smooth, smaller file, lossy edges
... -d '{"simplify_tolerance_m": 25, ...}'
```

## Error envelopes

| Status | When | What to do |
|---|---|---|
| `200` | Vectorisation succeeded. Body is the FeatureCollection. | — |
| `400` | Model is regression (LFMC) or unknown task type | Use `/olmoearth/download/<job_id>.tif` (COG) instead |
| `404` | Inference produced no class raster (every chunk failed) | Retry — typically transient PC outage; partial scene cache survives so the retry resumes faster |
| `413` | AOI exceeds the deployment's chunk-count ceiling | Shrink the bbox |
| `499` | Client closed the connection mid-job | (logged at info level, no retry needed — the in-flight chunks were cancelled) |
| `503` (Retry-After) | Inference fell back to a stub OR memory / breaker tripped | Wait the suggested seconds + retry. The detail body inlines `stub_reason` so you know what failed |

The `503 + stub_reason` envelope is new (audit fix `cdf4ac6`,
2026-04-25): the endpoint used to return `400 task_type=None` whenever
inference produced a stub, which was a confusing leak of an internal
state name. Now it explicitly says "fell back to a stub — underlying
reason: …".

## Workflow tip: reuse the cached job

The endpoint calls `start_inference(bbox, model_repo_id, date_range,
event_date)` internally. If you already ran the **Run + add to map**
button on the same `(bbox, model_repo_id, date_range, event_date)`,
the GeoJSON download skips the forward pass entirely and just
vectorises the cached raster (~1-2 s instead of multi-minute).

Conversely, calling the GeoJSON endpoint first warms the inference
cache, so a subsequent `Run + add to map` is also instant.

## Consuming the output

### Google Earth Pro

`File → Open → pick the .geojson` — Google Earth renders polygons
with their `color` property, hover shows `class_name` + `area_m2`.

### QGIS

`Layer → Add Layer → Add Vector Layer → pick .geojson`. The Symbology
tab can drive fill colour from the `color` property:
`Categorized → Value: color → Classify`.

### Python / pandas

```python
import geopandas as gpd
gdf = gpd.read_file("mangrove_f4a26abc.geojson")
print(gdf.dtypes)
# class_id          int64
# class_name       object
# color            object
# area_m2         float64
# pixel_count       int64
# geometry      geometry

# Total area per class
gdf.groupby("class_name")["area_m2"].sum().sort_values(ascending=False)
```

### Leaflet / web

```js
import L from "leaflet";
fetch("/api/olmoearth/ft-classification/geojson", { method: "POST", body: ... })
  .then(r => r.json())
  .then(fc => {
    L.geoJSON(fc, {
      style: f => ({
        color: f.properties.color,
        fillColor: f.properties.color,
        weight: 1, fillOpacity: 0.6,
      }),
      onEachFeature: (f, layer) => {
        layer.bindPopup(
          `<b>${f.properties.class_name}</b><br>` +
          `${f.properties.area_m2.toFixed(0)} m² · ` +
          `${f.properties.pixel_count} px`
        );
      },
    }).addTo(map);
  });
```

## Provenance + reproducibility

Every FeatureCollection carries the `job_id` it was vectorised from.
Re-running the same `(bbox, model_repo_id, date_range, event_date)`
produces the same `job_id` (deterministic SHA hash of the spec), so
GeoJSON outputs are bit-for-bit reproducible across hosts as long as
the underlying S2 scenes haven't been reprocessed by Microsoft
Planetary Computer.

The `scene_id` + `scene_datetime` fields in the top-level `properties`
let downstream consumers verify which Sentinel-2 capture the
classification was run against — important for any
publication-quality output where reviewers will want to re-fetch the
exact scenes.
