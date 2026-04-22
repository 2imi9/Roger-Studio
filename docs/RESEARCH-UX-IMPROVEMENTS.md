# Roger Studio — Research UX Improvement Backlog

Written 2026-04-21 after a full-session audit. Roger Studio today is a capable
single-user, single-session workbench: draw a bbox, run a FT model, inspect a
raster, chat with an LLM about it. To serve actual **scientific research**,
where work is multi-session, reviewable, and publishable, the following
capabilities are still missing or partial. Grouped by theme and roughly
ordered by leverage-per-effort.

---

## 1. Reproducibility & citation

Researchers need to re-run and cite what they did. The Studio currently has
**no persistent session model** — close the tab, lose the work. Labels are the
one exception (GeoJSON download).

| Gap | Proposed fix | Blast radius |
|---|---|---|
| No saved session | Persist the full scene state (bbox, imagery layers, label set, chat history, analysis results) as a single JSON blob on disk. `File → Save Session` / `Open Session`. Keyed by user-provided session name. | Medium (2-3 days) |
| No citation string for FT outputs | Every inference result should carry a citation block: `{ model_repo_id, commit_sha, s2_scene_id, s2_datetime, bbox, date_range, patch_size, sliding_window }`. Surface as a "Cite this layer" button in the Added Layer popover that copies a formatted BibTeX or CSL-JSON entry. | Small (1 day) |
| No DOI / permalink | Opt-in "publish session" that uploads the JSON blob to a local-first store (SQLite today, object storage later) and returns a short URL. Read-only view for collaborators. | Medium |
| Stale caches across restarts | Inference job cache is in-memory (`_jobs` dict). Persist to disk (sqlite or filesystem) so restarting the backend doesn't invalidate work the user hasn't saved yet. | Small — but touches concurrency |

---

## 2. Data provenance

Scientific work requires knowing where every pixel came from and what
transformations were applied.

| Gap | Proposed fix |
|---|---|
| Raster layers lack a "where did this come from?" panel | Extend the `ImageryLayer` type with a `provenance: { source, fetched_at, transforms: [...] }` field. Surface in Added Layer popover as a hoverable chip. For OE inference: model, scene, composite date. For uploads: filename, upload time. For STAC: collection, asset, mosaic date. |
| Labels lose ground-truth context | `LabelFeature` already has `created_at` + `source: "manual"`. Add `confidence` (optional float 0–1, defaults to 1 for manual), `reviewer` (optional str), `notes`. Surface in GeoJSON export. |
| No way to mark a layer as "derived from" another | Chain relationships: `layer.derived_from: layer_id`. Lets users trace "this classification came from this composite which came from this scene". Visualize as a small graph in a new "Lineage" tab. |

---

## 3. Uncertainty representation

Researchers don't just want class labels — they want confidence intervals,
per-pixel entropy, and out-of-distribution flags. Today the UI shows argmax
class only.

| Gap | Proposed fix |
|---|---|
| Classification outputs hide per-class probabilities | FT inference already returns `class_probs` for scene-level; sliding-window doesn't. Add a `probability_raster` output that stores the full `(H, W, num_classes)` tensor server-side, and expose a "Switch to uncertainty view" toggle that paints per-pixel `1 - max(probs)` as a heatmap. |
| No OOD flagging | Add a simple Mahalanobis distance check on the encoder embedding vs. the FT training set distribution; flag pixels > 3σ as OOD. Render as a checkerboard overlay. |
| Regression heads have no error bars | Most regression heads were trained with MSE loss; no native uncertainty. Add MC-dropout inference (N forward passes with dropout enabled, return mean + stddev). One-line toggle on the `run_olmoearth_inference` tool. |

---

## 4. Batch & scripted workflows

Right now every inference is one-bbox-at-a-time via the sidebar or chat. Real
research needs batching.

| Gap | Proposed fix |
|---|---|
| No way to run a model over N bboxes | Accept a GeoJSON FeatureCollection as input to `run_olmoearth_inference`. Fan out N jobs, return a collection of results. The LLM already does this serially in the SF Parks example; make it a first-class backend feature with dedup + parallelism. |
| No time-series over one bbox | Expose `run_olmoearth_timeseries(bbox, date_ranges: list[str])` that loops over dates and returns a stack. Frontend: a "play scrubber" on the Added Layer row so the user can animate the class map over time. |
| No programmatic export | Add a `/api/export/{layer_id}` endpoint that serves the full-resolution GeoTIFF for an inference result, not just the 256×256 PNG tiles. |

---

## 5. Comparative analysis

The Compare mode (SplitMap) is vertical side-by-side only. Researchers compare
many things: years, models, parameters.

| Gap | Proposed fix |
|---|---|
| Only 2-way compare | Add a "grid compare" mode: 2×2 or 3×3 synced map panels, each showing a different layer. Useful for year-over-year or model-A-vs-B-vs-WorldCover. |
| No statistical summary when comparing | Pop a side panel: class confusion matrix between two segmentation layers over the intersected bbox, or Pearson r between two regression layers. |
| No difference map | "Diff layer A against B" produces a new derived layer showing pixel-wise class agreement/disagreement (for segmentation) or A-B (for regression). |

---

## 6. LLM grounding & trust

The LLM agent is powerful but currently trusts itself too much on edge cases
(e.g., hallucinating tool responses when a tool output is unfamiliar).

| Gap | Proposed fix |
|---|---|
| No cited-output mode | Add a system-prompt flag that forces the LLM to cite `job_id` and `tool_name` for every numeric claim. Show a "📎 sources" panel under each assistant message with clickable links into the actual tool-call JSON. |
| Auto-retry opacity | Backend now auto-retries stubbed inference (landed in this session) with a chip `auto_retry_applied`. Surface that in the chat UI as a visible pill on the assistant message so the user knows the tool retried internally. |
| No "I don't know" affordance | When tool output has `stub_reason`, the LLM should refuse to fabricate results. Add a system-prompt example demonstrating the refusal. |
| Local Gemma 4 hallucinates faster than it calls tools | Add a phantom-tool-call detection pass (the repo already has `_phantom_tool_detect.py`) more aggressively for the smallest local models. Current behavior: detect + retry; target: also warn the user when the local model's phantom rate is > 1-in-5. |

---

## 7. Annotation quality & review workflow

Build Labels is MVP. Serious labeling projects need review, QA, and class
schemas.

| Gap | Proposed fix |
|---|---|
| No class schema file | Let a researcher upload a class schema (YAML / JSON) defining the allowed tag list, colors, and hierarchical parent→child relationships. LabelPanel reads this instead of the current 8 hardcoded defaults. |
| No review status | Every label should track `review_status: "unreviewed" \| "approved" \| "rejected" \| "needs_discussion"`. Filter/color the map by status. Export only approved labels. |
| No multi-label | Can't assign multiple tags to one feature. Common for research where a polygon is "forest" AND "owned by X". Change `tag: str` → `tags: str[]`. |
| No label history | When a tag changes on an existing feature, record the prior tag + timestamp. Surface as a hover tooltip on the feature. |

---

## 8. Export & sharing

| Gap | Proposed fix |
|---|---|
| Only GeoJSON export | Also export: Shapefile (for ArcGIS users), GeoPackage (for QGIS), COCO JSON (for ML folks), raster mask as GeoTIFF. |
| No report generation | "Generate Report" button that PDF-renders the scene bbox + loaded layers + legend + chat transcript + citation block. Matplotlib-based server-side, fetched by the frontend. |
| No share link | Same as session persistence (Theme 1). Short URL + read-only viewer mode. |

---

## 9. Collaboration

Single-user today. Nothing stops the backend from being multi-tenant.

| Gap | Proposed fix |
|---|---|
| No user accounts | Add a minimal auth layer (OAuth / magic link) + per-user session storage. Existing sessionStorage becomes per-user. |
| No shared project | A project = a session + a member list. Members can open read-only or edit. |
| No real-time cursors | Out of scope for MVP, but worth noting: WebSocket cursor sync is small once auth lands. |

---

## 10. Low-hanging paper cuts (< 1 day each)

Found during the audit, listed so they don't get lost.

- **Stale project memory** — claims 7 LLM tools; reality is 11. Update.
- **NEON CHM sample** shipped with CRS bbox misinterpretation (UTM meters leaked as lat/lon). Fixed indirectly by dropping it from Sample Rasters; but the underlying **backend `/api/upload` needs to reproject raster bbox to WGS84** before returning.
- **Terra-draw selection fill amber `#f59e0b`** aliases with the viridis top color on rasters. Change selection fill to a cool blue/gray so data + selection never read as one layer.
- **`mapCanvas.clientWidth = 400`** in preview — cosmetic only, but on narrow windows (< 480 px) the sidebar swallows the map. Add a responsive breakpoint that collapses the sidebar below a threshold.
- **`map.on("load", apply)` pattern** in MapView — fires only once; any imagery layer added mid-flyTo is stranded. Currently worked around in SampleRasters (explicit off-on cycle after fetch). Root-fix the reconciler to use `styledata` / `once("idle", apply)` with proper re-arming.
- **4 separate chat client modules** (gemma/cloud/claude/gemini/openai) duplicate ~80% of the same code (tool loop, status polling, history persistence, markdown rendering). Extract a `useChatClient(config)` hook that takes provider-specific config (base URL, auth header builder, model-preset list). Would cut ~1500 lines.
- **No test coverage for frontend** — backend has 76 tests; frontend has 0. Add Playwright covering the three Sample Rasters cards, the LLM popout last-chat routing, and the Map-tab submenu.

---

## Summary — highest leverage for researchers

If I had to pick three:

1. **Session save/load** (Theme 1) — unlocks multi-day research, not just multi-click.
2. **Per-pixel uncertainty** (Theme 3) — the gap between "a classification map" and "a publishable classification map."
3. **Cited-output LLM mode** (Theme 6) — turns the chat agent from a demo into a tool researchers can trust in a methods section.

Everything else is incremental on top of those three.
