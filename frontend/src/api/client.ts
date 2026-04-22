import type {
  AnalysisRequest,
  AnalysisResult,
  BBox,
  EnvDataResult,
  ElevationResult,
  DatasetInfo,
} from "../types";

const BASE = "/api";

/** Default per-request timeout. Generous enough for most reads (health
 * checks, dataset list, analyze / env-data / stats) but bounded so a
 * hung backend doesn't lock up the UI forever. Pass ``timeoutMs`` in
 * ``RequestOptions`` to override — e.g. OlmoEarth inference endpoints
 * that legitimately run for minutes pass a longer value. */
const DEFAULT_TIMEOUT_MS = 30_000;

export interface RequestOptions extends RequestInit {
  /** Override the default 30 s request timeout. Set to ``0`` to disable
   * the timeout entirely (useful when the caller already owns an
   * AbortSignal, like Cloud Chat's cancel button via
   * ``useCancellableSend``). */
  timeoutMs?: number;
}

async function request<T>(path: string, init?: RequestOptions): Promise<T> {
  const { timeoutMs: overrideTimeoutMs, signal: callerSignal, ...rest } = init ?? {};
  const timeoutMs = overrideTimeoutMs ?? DEFAULT_TIMEOUT_MS;

  // Merge the caller's signal (if any) with our timeout. Both the timeout
  // firing AND the caller aborting should short-circuit the fetch.
  // ``AbortSignal.any`` is the clean way to compose; fall back to a
  // manual controller for environments where it isn't implemented yet.
  let signal: AbortSignal | undefined;
  let timeoutId: ReturnType<typeof setTimeout> | undefined;
  if (timeoutMs > 0) {
    const timeoutSignal = AbortSignal.timeout(timeoutMs);
    if (callerSignal && typeof (AbortSignal as { any?: unknown }).any === "function") {
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      signal = (AbortSignal as any).any([callerSignal, timeoutSignal]);
    } else if (callerSignal) {
      // No AbortSignal.any available — roll a composite controller.
      const ctrl = new AbortController();
      const abort = () => ctrl.abort();
      timeoutSignal.addEventListener("abort", abort, { once: true });
      callerSignal.addEventListener("abort", abort, { once: true });
      signal = ctrl.signal;
    } else {
      signal = timeoutSignal;
    }
  } else {
    // ``RequestInit.signal`` allows ``null`` (clears a default); coalesce
    // so the downstream fetch call sees ``undefined`` consistently.
    signal = callerSignal ?? undefined;
  }

  try {
    const res = await fetch(`${BASE}${path}`, {
      headers: { "Content-Type": "application/json" },
      ...rest,
      signal,
    });
    if (!res.ok) {
      const text = await res.text();
      throw new Error(`API ${res.status}: ${text}`);
    }
    return await res.json();
  } catch (e) {
    // Distinguish timeout from a user-triggered abort so UI code can
    // render "request timed out" vs "you cancelled that" differently.
    // ``AbortSignal.timeout`` rejects with a TimeoutError DOMException;
    // manual cancellation via the caller's signal rejects with a plain
    // AbortError.
    if (e instanceof DOMException && e.name === "TimeoutError") {
      throw new Error(`API timeout after ${timeoutMs} ms: ${path}`);
    }
    throw e;
  } finally {
    if (timeoutId) clearTimeout(timeoutId);
  }
}

export async function analyze(req: AnalysisRequest): Promise<AnalysisResult> {
  // /analyze hits Planetary Computer STAC twice (WorldCover COG reads +
  // OlmoEarth catalog lookup) plus HuggingFace for dataset metadata. Even
  // with the backend parallelizing those calls, cold-cache worst-case can
  // reach ~60 s when PC is slow. The default 30 s timeout surfaced as a
  // visible "API timeout after 30000 ms: /analyze" in the UI. Give it a
  // 90 s budget — long enough for cold cache, short enough that a truly
  // hung backend still releases the UI within a minute and a half.
  return request("/analyze", {
    method: "POST",
    timeoutMs: 90_000,
    body: JSON.stringify(req),
  });
}

export async function getEnvData(
  bbox: BBox,
  variables: string[] = ["wind", "temperature", "solar", "humidity"]
): Promise<EnvDataResult> {
  const params = new URLSearchParams({
    west: String(bbox.west),
    south: String(bbox.south),
    east: String(bbox.east),
    north: String(bbox.north),
    variables: variables.join(","),
  });
  return request(`/env-data?${params}`);
}

export async function getElevation(
  bbox: BBox,
  resolution: number = 20
): Promise<ElevationResult> {
  // ``POST /api/reconstruct`` now accepts a single body with bbox + the
  // (optional) resolution field; before this change the route silently
  // ignored any resolution the caller passed because it took a naked
  // ``BBox`` as the body. Aligned in the architecture audit pass.
  // Backend response is ``ReconstructResponse`` (pydantic-validated) —
  // we unwrap ``terrain`` so callers see the raw grid shape they
  // declare in the return type.
  const resp = await request<{ status: string; terrain: ElevationResult }>(
    `/reconstruct`,
    {
      method: "POST",
      body: JSON.stringify({ ...bbox, resolution }),
    },
  );
  return resp.terrain;
}

export async function uploadFile(file: File): Promise<DatasetInfo> {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(`${BASE}/upload`, { method: "POST", body: form });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`Upload failed (${res.status}): ${text}`);
  }
  return res.json();
}

export async function listDatasets(): Promise<DatasetInfo[]> {
  return request("/datasets");
}

export async function deleteDataset(filename: string): Promise<void> {
  await request(`/datasets/${encodeURIComponent(filename)}`, {
    method: "DELETE",
  });
}

export async function autoLabel(
  filename: string,
  nClasses: number = 6,
  method: "auto" | "tipsv2" | "spectral" | "samgeo" = "auto"
): Promise<GeoJSON.FeatureCollection & { properties?: Record<string, unknown> }> {
  const params = new URLSearchParams({
    n_classes: String(nClasses),
    method,
  });
  return request(`/auto-label/${encodeURIComponent(filename)}?${params}`, {
    method: "POST",
  });
}

export async function validateLabels(
  filename: string,
  geojson: GeoJSON.FeatureCollection,
  opts: {
    pipeline?: "tipsv2" | "samgeo" | "spectral";
    onlyLowConfidence?: boolean;
    lowConfThreshold?: number;
    maxConcurrent?: number;
  } = {}
): Promise<GeoJSON.FeatureCollection & { properties?: Record<string, unknown> }> {
  const params = new URLSearchParams({
    pipeline: opts.pipeline || "tipsv2",
    only_low_confidence: String(opts.onlyLowConfidence ?? true),
    low_conf_threshold: String(opts.lowConfThreshold ?? 0.6),
    max_concurrent: String(opts.maxConcurrent ?? 4),
  });
  return request(`/auto-label/${encodeURIComponent(filename)}/validate?${params}`, {
    method: "POST",
    body: JSON.stringify(geojson),
  });
}

export interface LLMHealth {
  reachable: boolean;
  base_url: string;
  model: string;
  docker_available: boolean;
  container_name: string;
  container_status: string; // 'running' | 'missing' | 'exited' | 'docker_not_installed' | ...
  image: string;
  image_built?: boolean;
  building_image?: boolean;
  port: number;
  hf_cache: string;
  hf_token_set?: boolean;
  last_start_error?: string | null;
  provider_mode?: "local" | "cloud";
  api_key_set?: boolean;
  runtime?: "ollama" | "vllm" | "cloud";
}

export async function gemmaHealth(): Promise<LLMHealth> {
  return request("/auto-label/gemma/health");
}

export async function gemmaSetModel(model: string): Promise<{ ok: boolean; model: string }> {
  return request("/auto-label/gemma/model", {
    method: "POST",
    body: JSON.stringify({ model }),
  });
}

export async function gemmaStart(hfToken?: string): Promise<{
  started: boolean;
  already_running: boolean;
  container_name?: string;
  container_id?: string;
  model?: string;
  note?: string;
}> {
  const body = hfToken ? { hf_token: hfToken } : {};
  return request("/auto-label/gemma/start", {
    method: "POST",
    body: JSON.stringify(body),
  });
}

export async function gemmaStop(): Promise<{
  stopped: boolean;
  was_status?: string;
  container_name?: string;
}> {
  return request("/auto-label/gemma/stop", { method: "POST" });
}

export async function gemmaLogs(tail: number = 200): Promise<{ logs: string }> {
  return request(`/auto-label/gemma/logs?tail=${tail}`);
}

export interface AgentArtifact {
  id: string;
  filename: string;
  content_type: string;
  size_bytes: number;
  summary: string;
  created_at: number;
  created_at_iso: string;
  download_url: string;
}

export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
  // Optional assistant-only fields that the UI may render as extra blocks.
  // Populated by the backend's tool-loop response shape; harmless to round-
  // trip through chat history because the server rebuilds them per turn.
  reasoning_content?: string;
  phantom_tool_call_fixed?: boolean;
  empty_retry_fixed?: boolean;
  stopped_reason?: string;
  artifacts?: AgentArtifact[];
}

export async function gemmaChat(
  messages: ChatMessage[],
  sceneContext?: Record<string, unknown>,
  opts: { signal?: AbortSignal } = {},
): Promise<ChatMessage & {
  usage?: Record<string, number>;
  reasoning_content?: string;
  phantom_tool_call_fixed?: boolean;
  empty_retry_fixed?: boolean;
  stopped_reason?: string;
  artifacts?: AgentArtifact[];
}> {
  // See note on ``cloudChat`` — LLM tool loops can run long; disable the
  // global 30 s request timeout and rely on the caller's Stop button
  // (via ``opts.signal``) for cancellation.
  return request("/auto-label/gemma/chat", {
    method: "POST",
    timeoutMs: 0,
    body: JSON.stringify({ messages, scene_context: sceneContext }),
    signal: opts.signal,
  });
}

// ---------------------------------------------------------------------------
// Cloud Chat — NVIDIA NIM (mirrors the gemma* shape so the UI just swaps
// the API call). API key is sessionStorage-only and forwarded per-request;
// the backend never persists it.
// ---------------------------------------------------------------------------

export interface CloudHealth {
  reachable: boolean;
  auth_ok: boolean;
  model: string;
  base_url: string;
  api_key_set: boolean;
  error: string | null;
  available_models?: string[];
}

export async function cloudHealth(apiKey?: string): Promise<CloudHealth> {
  const qs = apiKey ? `?api_key=${encodeURIComponent(apiKey)}` : "";
  return request(`/cloud/health${qs}`);
}

export async function cloudSetModel(
  model: string,
): Promise<{ ok: boolean; model: string }> {
  return request("/cloud/model", {
    method: "POST",
    body: JSON.stringify({ model }),
  });
}

export async function cloudChat(
  messages: ChatMessage[],
  sceneContext?: Record<string, unknown>,
  opts: { apiKey?: string; model?: string; signal?: AbortSignal } = {},
): Promise<
  ChatMessage & {
    usage?: Record<string, number>;
    model?: string;
    provider?: string;
  }
> {
  // LLM chat runs a tool loop that can legitimately take minutes (S2
  // fetch → FT inference → summarization, etc.). Disable the global
  // 30 s timeout; the user already owns cancellation via the Stop
  // button wired through ``useCancellableSend`` (``opts.signal``).
  return request("/cloud/chat", {
    method: "POST",
    timeoutMs: 0,
    body: JSON.stringify({
      messages,
      scene_context: sceneContext,
      api_key: opts.apiKey,
      model: opts.model,
    }),
    signal: opts.signal,
  });
}

// ---------------------------------------------------------------------------
// Claude Chat — Anthropic Messages API. Same shape as cloudChat — only the
// transport differs. API key sessionStorage-only, never persisted server-side.
// ---------------------------------------------------------------------------

export interface ClaudeHealth {
  reachable: boolean;
  auth_ok: boolean;
  model: string;
  api_key_set: boolean;
  error: string | null;
  display_name?: string | null;
  max_input_tokens?: number | null;
  max_output_tokens?: number | null;
}

export async function claudeHealth(apiKey?: string): Promise<ClaudeHealth> {
  const qs = apiKey ? `?api_key=${encodeURIComponent(apiKey)}` : "";
  return request(`/claude/health${qs}`);
}

export async function claudeSetModel(
  model: string,
): Promise<{ ok: boolean; model: string }> {
  return request("/claude/model", {
    method: "POST",
    body: JSON.stringify({ model }),
  });
}

export async function claudeChat(
  messages: ChatMessage[],
  sceneContext?: Record<string, unknown>,
  opts: { apiKey?: string; model?: string; signal?: AbortSignal } = {},
): Promise<
  ChatMessage & {
    usage?: Record<string, number>;
    model?: string;
    provider?: string;
  }
> {
  // Tool-loop chat; caller owns cancellation via ``opts.signal``.
  return request("/claude/chat", {
    method: "POST",
    timeoutMs: 0,
    body: JSON.stringify({
      messages,
      scene_context: sceneContext,
      api_key: opts.apiKey,
      model: opts.model,
    }),
    signal: opts.signal,
  });
}

// ---------------------------------------------------------------------------
// Gemini Chat — Google AI Studio via their OpenAI-compatible endpoint. Same
// shape as cloudChat / claudeChat — the Cloud tab swaps transports by
// provider without changing the message contract.
// ---------------------------------------------------------------------------

export interface GeminiHealth {
  reachable: boolean;
  auth_ok: boolean;
  model: string;
  base_url: string;
  api_key_set: boolean;
  error: string | null;
  available_models?: string[];
}

export async function geminiHealth(apiKey?: string): Promise<GeminiHealth> {
  const qs = apiKey ? `?api_key=${encodeURIComponent(apiKey)}` : "";
  return request(`/gemini/health${qs}`);
}

export async function geminiSetModel(
  model: string,
): Promise<{ ok: boolean; model: string }> {
  return request("/gemini/model", {
    method: "POST",
    body: JSON.stringify({ model }),
  });
}

export async function geminiChat(
  messages: ChatMessage[],
  sceneContext?: Record<string, unknown>,
  opts: { apiKey?: string; model?: string; signal?: AbortSignal } = {},
): Promise<
  ChatMessage & {
    usage?: Record<string, number>;
    model?: string;
    provider?: string;
    artifacts?: AgentArtifact[];
  }
> {
  // Tool-loop chat; caller owns cancellation via ``opts.signal``.
  return request("/gemini/chat", {
    method: "POST",
    timeoutMs: 0,
    body: JSON.stringify({
      messages,
      scene_context: sceneContext,
      api_key: opts.apiKey,
      model: opts.model,
    }),
    signal: opts.signal,
  });
}


// ---------------------------------------------------------------------------
// OpenAI Chat — native /v1/chat/completions (the spec every other cloud
// provider above emulates). Same request/response shape as cloudChat /
// claudeChat / geminiChat — the CloudHub swaps transports by provider
// without changing the message contract. API key stays client-side and
// is forwarded per-request; server-owned deployments can export
// OPENAI_API_KEY instead.
// ---------------------------------------------------------------------------

export interface OpenAIHealth {
  reachable: boolean;
  auth_ok: boolean;
  model: string;
  base_url: string;
  api_key_set: boolean;
  error: string | null;
  available_models?: string[];
}

export async function openaiHealth(apiKey?: string): Promise<OpenAIHealth> {
  const qs = apiKey ? `?api_key=${encodeURIComponent(apiKey)}` : "";
  return request(`/openai/health${qs}`);
}

export async function openaiSetModel(
  model: string,
): Promise<{ ok: boolean; model: string }> {
  return request("/openai/model", {
    method: "POST",
    body: JSON.stringify({ model }),
  });
}

export async function openaiChat(
  messages: ChatMessage[],
  sceneContext?: Record<string, unknown>,
  opts: { apiKey?: string; model?: string; signal?: AbortSignal } = {},
): Promise<
  ChatMessage & {
    usage?: Record<string, number>;
    model?: string;
    provider?: string;
    artifacts?: AgentArtifact[];
  }
> {
  // Tool-loop chat; caller owns cancellation via ``opts.signal``.
  return request("/openai/chat", {
    method: "POST",
    timeoutMs: 0,
    body: JSON.stringify({
      messages,
      scene_context: sceneContext,
      api_key: opts.apiKey,
      model: opts.model,
    }),
    signal: opts.signal,
  });
}


export async function healthCheck(): Promise<{ status: string }> {
  return request("/health");
}

export interface OlmoEarthModel {
  repo_id: string;
  type?: "encoder" | "fine-tuned" | "unknown";
  size_tier?: "smallest" | "small" | "medium" | "largest";
  base?: string;
  task?: string;
  task_key?: string;
  downloads?: number;
  likes?: number;
  last_modified?: string;
  reason?: string;
}

export interface OlmoEarthDatasetLive {
  repo_id: string;
  description?: string;
  license?: string;
  task?: string;
  coverage?: string;
  size?: string;
  downloads?: number;
  likes?: number;
  last_modified?: string;
  docs?: string;
}

export interface OlmoEarthCatalog {
  models: OlmoEarthModel[];
  datasets: OlmoEarthDatasetLive[];
  project_coverage: { repo_id: string; region: BBox; dataset: OlmoEarthDatasetLive }[];
  recommended_model?: OlmoEarthModel;
  notes: string[];
}

export async function getOlmoEarthCatalog(
  opts: { bbox?: BBox; force?: boolean } = {},
): Promise<OlmoEarthCatalog> {
  const params = new URLSearchParams();
  if (opts.bbox) {
    params.set("west", String(opts.bbox.west));
    params.set("south", String(opts.bbox.south));
    params.set("east", String(opts.bbox.east));
    params.set("north", String(opts.bbox.north));
  }
  if (opts.force) params.set("force", "true");
  const qs = params.toString();
  // Cold-cache path (first call after a backend restart) hits HuggingFace's
  // /api/models and /api/datasets with 10 s timeout + 2 retries each. Even
  // with the backend now parallelizing them, a flaky HF round can still
  // run ~30 s. 60 s client timeout lets the retry path complete instead of
  // surfacing as "API timeout after 30000 ms" in the UI.
  return request(`/olmoearth/catalog${qs ? `?${qs}` : ""}`, { timeoutMs: 60_000 });
}

export interface OlmoEarthRepoStatus {
  status?: "loading" | "cached" | "error";
  size_bytes?: number;
  path?: string;
  error?: string | null;
  started_ts?: number;
  finished_ts?: number;
  repo_type?: string;
}

export interface OlmoEarthCacheStatus {
  repos: Record<string, OlmoEarthRepoStatus>;
}

export async function getOlmoEarthCacheStatus(): Promise<OlmoEarthCacheStatus> {
  return request("/olmoearth/cache-status");
}

/** In-memory warm-cache snapshot. Distinct from ``getOlmoEarthCacheStatus``
 * which is about DISK cache. A repo_id appearing in ``loaded`` means the
 * next inference call on that repo skips the 2–10 s safetensors re-read
 * — used by SplitMap to badge compare-demo buttons "warm (~3 s)" vs
 * "cold (~30 s)" so users can predict click cost. */
export interface OlmoEarthLoadedModels {
  loaded: string[];
}

export async function getLoadedOlmoEarthModels(): Promise<OlmoEarthLoadedModels> {
  return request("/olmoearth/loaded-models");
}

/** Pull the user's sessionStorage-stored provider keys so the
 * explain-raster endpoint can succeed even when the server has no
 * env-var keys. Mirrors the keys the chat components write: NIM under
 * ``geoenv.cloud.apiKey``, Claude under ``geoenv.claude.apiKey``. */
function readProviderKey(storageKey: string): string {
  try {
    const raw = sessionStorage.getItem(storageKey);
    if (!raw) return "";
    const parsed = JSON.parse(raw);
    return typeof parsed === "string" ? parsed : "";
  } catch {
    return "";
  }
}

export interface ExplainRasterRequest {
  model_repo_id: string;
  task_type?: string | null;
  colormap?: string | null;
  bbox?: { west: number; south: number; east: number; north: number } | null;
  scene_id?: string | null;
  scene_datetime?: string | null;
  scene_cloud_cover?: number | null;
  class_names?: string[] | null;
  top_classes?: { index: number; name: string; score?: number }[] | null;
  prediction_value?: number | null;
  units?: string | null;
  stub_reason?: string | null;
  /** Inference job_id. The server plumbs this into the agent's
   * scene_context so the raster-histogram / scalar-stats tools know
   * which job's raster to read when the LLM calls them without
   * explicit args. */
  job_id?: string | null;
}

export interface ExplainRasterToolCall {
  name: string;
  ok: boolean;
  summary: string | null;
}

export interface ExplainRasterResponse {
  explanation: string;
  source: "nim" | "claude" | "gemma" | "fallback";
  model: string | null;
  tool_calls: ExplainRasterToolCall[];
}

/** Ask the backend's ``/api/explain-raster`` endpoint for an LLM-generated
 * explanation of an inference raster. The pill-click UX in MapView calls
 * this when the user expands a raster to learn what the colors mean —
 * swaps the old 110-row class-list dump for a 2–3 paragraph summary. */
export async function explainRaster(
  req: ExplainRasterRequest,
): Promise<ExplainRasterResponse> {
  const nimKey = readProviderKey("geoenv.cloud.apiKey");
  const claudeKey = readProviderKey("geoenv.claude.apiKey");
  // 180 s timeout — NIM free-tier (minimax-m2.7) under load can take
  // 90–120 s on its own; Claude fallback then adds another ~15 s if
  // NIM fails. Earlier 60 s client-side cap was shorter than NIM's own
  // server-side timeout (120 s), so the browser gave up while the
  // server was still waiting for the LLM — user saw "API timeout after
  // 60000 ms" even when the request was still in flight.
  return request("/explain-raster", {
    method: "POST",
    timeoutMs: 180_000,
    body: JSON.stringify({
      ...req,
      nim_api_key: nimKey || undefined,
      claude_api_key: claudeKey || undefined,
    }),
  });
}

export async function loadOlmoEarthRepo(args: {
  repoId: string;
  repoType: "model" | "dataset";
  hfToken?: string;
}): Promise<OlmoEarthRepoStatus> {
  return request("/olmoearth/load", {
    method: "POST",
    body: JSON.stringify({
      repo_id: args.repoId,
      repo_type: args.repoType,
      hf_token: args.hfToken,
    }),
  });
}

export async function unloadOlmoEarthRepo(
  repoId: string,
): Promise<{ removed: boolean; bytes_freed?: number; error?: string }> {
  return request("/olmoearth/unload", {
    method: "POST",
    body: JSON.stringify({ repo_id: repoId }),
  });
}

// Per-task legend block. Regression legends carry the colormap gradient +
// predicted value; classification / segmentation legends carry per-class
// entries that the frontend renders as color swatches.
export type OlmoEarthLegend =
  | {
      kind: "regression";
      label: string;
      stops: [string, number][];
      units: string | null;
      value: number | null;
      /** Optional honesty note — e.g. "regression output — predicted
       *  moisture percentage" — rendered as small italic text under
       *  the gradient so users don't over-interpret the number. */
      note?: string;
    }
  | {
      kind: "classification" | "segmentation";
      classes: { index: number; name: string; color: string }[];
      names_tentative: boolean;
      colors_source?: "published" | "colormap_gradient";
      /** Calibration-honesty note. The scores behind these class ids
       *  are raw uncalibrated softmax — do NOT read them as
       *  probabilities. Surfaced in the legend panel. */
      note?: string;
    }
  | {
      label: string;
      stops: [string, number][];
      note?: string;
    };

export interface OlmoEarthInferenceResult {
  job_id: string;
  tile_url: string;
  legend?: OlmoEarthLegend;
  colormap: string;
  kind: "stub" | "pytorch";
  status: "ready" | "running";
  model_repo_id: string;
  bbox: BBox;
  notes?: string[];
  // FT + pytorch paths fill these in.
  task_type?: "classification" | "segmentation" | "regression" | "embedding";
  num_classes?: number;
  class_names?: string[];
  class_names_tentative?: boolean;
  class_probs?: number[];
  /** Class indices that actually appear in the rendered raster — used by
   * the map color-strip to show only present-class swatches instead of
   * the full catalog (e.g. 4 Bay Area ecosystems out of 110). Absent or
   * null on older responses; when unset the UI falls back to the full
   * class list. */
  present_class_ids?: number[] | null;
  prediction_value?: number | null;
  units?: string | null;
  decoder_key?: string;
  embedding_dim?: number;
  patch_size?: number;
  sliding_window?: boolean;
  window_size?: number | null;
  scene_id?: string;
  scene_datetime?: string;
  scene_cloud_cover?: number | null;
  stub_reason?: string;
}

export async function startOlmoEarthInference(args: {
  bbox: BBox;
  modelRepoId: string;
  dateRange?: string;
  slidingWindow?: boolean;
  windowSize?: number;
}): Promise<OlmoEarthInferenceResult> {
  // Inference can legitimately take minutes (S2 STAC search + multi-band
  // download + encoder forward + FT head + raster caching). Raise the
  // request timeout accordingly so the default 30 s doesn't kill a
  // real run. The backend itself has its own ``OPENAI_TIMEOUT`` /
  // ``NVIDIA_TIMEOUT`` etc. for upstream calls; this is just the
  // browser→backend leg.
  return request("/olmoearth/infer", {
    method: "POST",
    timeoutMs: 5 * 60 * 1000, // 5 min
    body: JSON.stringify({
      bbox: args.bbox,
      model_repo_id: args.modelRepoId,
      date_range: args.dateRange,
      sliding_window: args.slidingWindow,
      window_size: args.windowSize,
    }),
  });
}

// Shape of one demo side returned from /api/olmoearth/demo-pairs. The
// frontend uses this directly as an ImageryLayer (id, label, tileUrl).
// `spec` is the full inference payload — passed back to the prebake
// endpoint on first click to trigger tile generation if the user is
// impatient and doesn't want to wait on lazy-on-tile-request rendering.
/** Static legend hint resolvable before inference runs. Carries the
 * colormap key, a human axis label, an honesty note about what the
 * numeric values actually mean, and the gradient stops the frontend
 * paints. Lets SplitMap show a legend the moment a demo loads instead
 * of waiting ~30 s for the post-inference legend. */
export interface OlmoEarthLegendHint {
  colormap: string;
  label?: string | null;
  note?: string | null;
  stops: [string, number][];
  /** Semantic anchor for the left/low end of the gradient (e.g.
   * "non-mangrove" for mangrove softmax, "cropland" for AWF landuse).
   * Falls back to "low" in the UI when absent. */
  low_label?: string | null;
  /** Semantic anchor for the right/high end. Falls back to "high". */
  high_label?: string | null;
}

export interface OlmoEarthDemoSide {
  id: string;
  label: string;
  tile_url: string;
  job_id: string;
  spec: {
    bbox: { west: number; south: number; east: number; north: number };
    model_repo_id: string;
    date_range: string;
    max_size_px: number;
    sliding_window: boolean;
    window_size: number | null;
  };
  legend_hint?: OlmoEarthLegendHint | null;
}

export interface OlmoEarthDemoPair {
  id: string;
  title: string;
  blurb: string;
  fit_bbox: { west: number; south: number; east: number; north: number };
  a: OlmoEarthDemoSide;
  b: OlmoEarthDemoSide;
}

export async function getOlmoEarthDemoPairs(): Promise<{ pairs: OlmoEarthDemoPair[] }> {
  return request("/olmoearth/demo-pairs");
}

/** Kick background inference for a single demo pair so its tiles are ready
 * when the user actually drags the compare divider. Fire-and-forget. */
export async function prebakeOlmoEarthDemo(pairId: string): Promise<unknown> {
  return request(`/olmoearth/demo-pairs/prebake?pair_id=${encodeURIComponent(pairId)}`, {
    method: "POST",
  });
}

export interface PolygonStatsResponse {
  perimeter_km: number;
  area_km2: number;
  centroid: { lat: number; lon: number };
  bbox: BBox;
  vertex_count: number;
  elevation_sample_count: number;
  elevation: {
    min_m: number;
    max_m: number;
    mean_m: number;
    median_m: number;
    range_m: number;
    source: string;
  } | null;
  error?: string;
}

export async function getPolygonStats(
  geometry: GeoJSON.Polygon | GeoJSON.MultiPolygon,
  opts: { includeElevation?: boolean; resolution?: number } = {}
): Promise<PolygonStatsResponse> {
  // /polygon-stats hits Open-Meteo for the full elevation sample grid
  // (default 20×20 = 400 points). Cold-cache or large polygons can run
  // 30-50 s, which previously surfaced as "API timeout after 30000 ms:
  // /polygon-stats" — give it a 60 s budget. Backend additionally skips
  // elevation entirely on absurdly large polygons (> 10⁶ km²) since the
  // grid would need thousands of samples and the result is meaningless
  // at country-scale anyway.
  return request("/polygon-stats", {
    method: "POST",
    timeoutMs: 60_000,
    body: JSON.stringify({
      geometry,
      include_elevation: opts.includeElevation ?? true,
      resolution: opts.resolution ?? 20,
    }),
  });
}

export interface StacItem {
  id: string;
  collection: string;
  datetime: string;
  cloud_cover: number | null;
  bbox: number[];
  assets: string[];
  thumbnail_url: string | null;
}

export interface StacSearchResponse {
  count: number;
  matched: number | null;
  items: StacItem[];
  error?: string;
}

export async function searchStacImagery(args: {
  bbox: BBox;
  datetime: string;
  collections?: string[];
  maxCloudCover?: number;
  limit?: number;
}): Promise<StacSearchResponse> {
  return request("/stac/search", {
    method: "POST",
    body: JSON.stringify({
      bbox: args.bbox,
      datetime: args.datetime,
      collections: args.collections,
      max_cloud_cover: args.maxCloudCover ?? 20,
      limit: args.limit ?? 10,
    }),
  });
}

export interface CompositeTileResponse {
  tile_url: string;
  tilejson_url: string;
  search_id: string;
  collection: string;
  assets: string[];
  datetime_range: string;
  bbox: BBox;
  notes?: string[];
  error?: string;
}

export async function getCompositeTileUrl(args: {
  bbox: BBox;
  datetime: string;
  collection?: string;
  assets?: string[];
  maxCloudCover?: number;
}): Promise<CompositeTileResponse> {
  return request("/stac/composite-tile-url", {
    method: "POST",
    body: JSON.stringify({
      bbox: args.bbox,
      datetime: args.datetime,
      collection: args.collection ?? "sentinel-2-l2a",
      assets: args.assets,
      max_cloud_cover: args.maxCloudCover ?? 20,
    }),
  });
}

// ---------------------------------------------------------------------------
// Projects — persistent session container. Mirrors OlmoEarth Studio's
// /api/v1/projects resource shape (POST create, GET read, PUT update,
// DELETE delete, POST /search list). State is opaque JSON so the frontend
// can evolve what it stores without touching the backend schema.
// ---------------------------------------------------------------------------

export interface ProjectRead {
  id: string;
  name: string;
  description: string | null;
  created_at: string;
  updated_at: string;
  state: Record<string, unknown>;
}

export interface ProjectWrite {
  name: string;
  description?: string | null;
  state: Record<string, unknown>;
}

export async function createProject(payload: ProjectWrite): Promise<ProjectRead> {
  return request("/v1/projects", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export async function readProject(id: string): Promise<ProjectRead> {
  return request(`/v1/projects/${encodeURIComponent(id)}`);
}

export async function updateProject(id: string, payload: ProjectWrite): Promise<ProjectRead> {
  return request(`/v1/projects/${encodeURIComponent(id)}`, {
    method: "PUT",
    body: JSON.stringify(payload),
  });
}

export async function deleteProject(id: string): Promise<{ deleted: string }> {
  return request(`/v1/projects/${encodeURIComponent(id)}`, { method: "DELETE" });
}

export async function searchProjects(
  opts: { nameContains?: string; limit?: number; offset?: number } = {},
): Promise<ProjectRead[]> {
  return request("/v1/projects/search", {
    method: "POST",
    body: JSON.stringify({
      name_contains: opts.nameContains,
      limit: opts.limit ?? 50,
      offset: opts.offset ?? 0,
    }),
  });
}
