import { useMemo, useState, type FC } from "react";
import { Panel, SectionTitle } from "./ui/Panel";
import { DrawPolygon, DrawPoint, DrawLine, DownloadGeoJSON, type IconState } from "./icons";

type IconCmp = FC<{ className?: string; state?: IconState }>;

// Icon component picker for the label-type selector — keeps the JSX
// branchless inside the .map() call below. Wrapped with explicit cast
// because each Figma-generated icon has a slightly narrower local
// IconProps type that doesn't satisfy IconCmp without a widen cast.
const TYPE_ICON: Record<"point" | "polygon" | "line", IconCmp> = {
  polygon: DrawPolygon as unknown as IconCmp,
  point: DrawPoint as unknown as IconCmp,
  line: DrawLine as unknown as IconCmp,
};

// Default tag set mirrors the LAND_COVER_COLORS already used elsewhere in the
// app — keeps colors consistent between manual labels, auto-label results, and
// Analysis-tab readouts. User can extend with "Custom…" tags; those get a
// deterministic color from the tag name so the same tag always renders the
// same shade across reloads.
export const DEFAULT_TAGS: { name: string; color: string }[] = [
  { name: "Forest",   color: "#228b22" },
  { name: "Cropland", color: "#f0e68c" },
  { name: "Urban",    color: "#808080" },
  { name: "Water",    color: "#1e90ff" },
  { name: "Grassland", color: "#90ee90" },
  { name: "Barren",   color: "#d2b48c" },
  { name: "Wetland",  color: "#5f9ea0" },
  { name: "Snow/Ice", color: "#f0f8ff" },
];

const FALLBACK_PALETTE = [
  "#e11d48", "#7c3aed", "#0891b2", "#16a34a",
  "#ea580c", "#db2777", "#0ea5e9", "#65a30d",
];

export function colorForTag(name: string, customTags: string[] = []): string {
  const def = DEFAULT_TAGS.find((t) => t.name === name);
  if (def) return def.color;
  const all = [...DEFAULT_TAGS.map((t) => t.name), ...customTags];
  const idx = all.indexOf(name);
  if (idx === -1) {
    // Hash → palette index. Stable across reloads so same tag = same color.
    let h = 0;
    for (let i = 0; i < name.length; i++) h = (h * 31 + name.charCodeAt(i)) >>> 0;
    return FALLBACK_PALETTE[h % FALLBACK_PALETTE.length];
  }
  return FALLBACK_PALETTE[(idx - DEFAULT_TAGS.length) % FALLBACK_PALETTE.length];
}

export type LabelType = "point" | "polygon" | "line";

export interface LabelFeature {
  type: "Feature";
  id: string;
  properties: {
    tag: string;
    label_type: LabelType;
    color: string;
    class_name: string; // alias of tag — lets the existing MapView hover popup show it
    metadata?: Record<string, unknown>;
    created_at: string;
    source: "manual";
  };
  geometry: GeoJSON.Point | GeoJSON.Polygon | GeoJSON.LineString;
}

export interface LabelMode {
  active: boolean;
  type: LabelType;
  tag: string;
}

interface LabelPanelProps {
  projectName: string;
  onProjectNameChange: (v: string) => void;
  labelMode: LabelMode;
  onLabelModeChange: (m: LabelMode) => void;
  customTags: string[];
  onAddCustomTag: (tag: string) => void;
  onRemoveCustomTag: (tag: string) => void;
  features: LabelFeature[];
  onDeleteFeature: (id: string) => void;
  onClearAll: () => void;
  /** Push the current label FeatureCollection to the map's Added Layer
   * list as a single toggleable vector layer. Called by the
   * "Add to map layers" button; App handles the actual layer insertion.
   * Optional so LabelPanel remains usable in contexts without the
   * layer-merge affordance. */
  onMergeToMapLayers?: () => void;
  /** Whether the label set is currently present in the Added Layer list.
   * Drives the button's disabled / "on map" state so the user can tell
   * they've already merged this label set (re-clicking would just
   * overwrite the existing layer with the same id). */
  mergedToMapLayers?: boolean;
}

function downloadGeoJSON(projectName: string, features: LabelFeature[]) {
  const fc: GeoJSON.FeatureCollection = {
    type: "FeatureCollection",
    features: features as unknown as GeoJSON.Feature[],
  };
  // Wrap with project metadata at the FeatureCollection level — non-standard
  // but harmless for any GeoJSON consumer (extra keys are ignored), and lets
  // the file round-trip back into Roger Studio with project name preserved.
  const wrapped = {
    ...fc,
    properties: {
      project_name: projectName || "Untitled label set",
      generated_at: new Date().toISOString(),
      source: "roger-studio-manual-labels",
      feature_count: features.length,
    },
  };
  const blob = new Blob([JSON.stringify(wrapped, null, 2)], {
    type: "application/geo+json",
  });
  const url = URL.createObjectURL(blob);
  const safeName = (projectName || "labels").trim().replace(/[^a-z0-9-_]+/gi, "_") || "labels";
  const a = document.createElement("a");
  a.href = url;
  // Local timezone YYYY-MM-DD (sv-SE locale). Avoids evening-local imports
  // being stamped with tomorrow's UTC date.
  a.download = `${safeName}_${new Date().toLocaleDateString("sv-SE")}.geojson`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

const LABEL_TYPE_OPTIONS: { value: LabelType; label: string; hint: string }[] = [
  { value: "polygon", label: "Polygon", hint: "Areas (land cover, parcels)" },
  { value: "point",   label: "Point",   hint: "Sites (sinkholes, wells)" },
  { value: "line",    label: "Line",    hint: "Linear (roads, rivers)" },
];

export function LabelPanel({
  projectName,
  onProjectNameChange,
  labelMode,
  onLabelModeChange,
  customTags,
  onAddCustomTag,
  onRemoveCustomTag,
  features,
  onDeleteFeature,
  onClearAll,
  onMergeToMapLayers,
  mergedToMapLayers,
}: LabelPanelProps) {
  const [newTag, setNewTag] = useState("");

  const allTags = useMemo(
    () => [...DEFAULT_TAGS.map((t) => t.name), ...customTags],
    [customTags],
  );

  const handleAddTag = () => {
    const t = newTag.trim();
    if (!t || allTags.includes(t)) return;
    onAddCustomTag(t);
    onLabelModeChange({ ...labelMode, tag: t });
    setNewTag("");
  };

  const toggleActive = () => {
    onLabelModeChange({ ...labelMode, active: !labelMode.active });
  };

  const featuresByTag = useMemo(() => {
    const m = new Map<string, number>();
    for (const f of features) {
      const t = f.properties.tag || "(untagged)";
      m.set(t, (m.get(t) || 0) + 1);
    }
    return [...m.entries()].sort((a, b) => b[1] - a[1]);
  }, [features]);

  return (
    <Panel>
      <div className="flex items-center justify-between mb-3">
        <SectionTitle>Build Labels</SectionTitle>
        <span
          className="text-[10px] font-semibold uppercase tracking-widest px-2 py-1 rounded bg-geo-accent-soft text-geo-accent"
          title="Manually annotate features on the map and export as GeoJSON. Inspired by OlmoEarth Studio."
        >
          MVP
        </span>
      </div>

      {/* Project name */}
      <label className="block text-[11px] font-semibold text-geo-text mb-1">
        Project name
      </label>
      <input
        type="text"
        value={projectName}
        onChange={(e) => onProjectNameChange(e.target.value)}
        placeholder="Untitled label set"
        spellCheck={false}
        className="w-full mb-3 px-2 py-1.5 bg-geo-bg border border-geo-border rounded text-[12px] text-geo-text placeholder:text-geo-dim focus:outline-none focus:border-geo-accent"
      />

      {/* Label type */}
      <label className="block text-[11px] font-semibold text-geo-text mb-1">
        Label type
      </label>
      <div className="grid grid-cols-3 gap-1.5 mb-3">
        {LABEL_TYPE_OPTIONS.map((opt) => {
          const Icon = TYPE_ICON[opt.value];
          const selected = labelMode.type === opt.value;
          return (
            <button
              key={opt.value}
              onClick={() => onLabelModeChange({ ...labelMode, type: opt.value })}
              className={`px-2 py-1.5 text-[11px] font-medium rounded border cursor-pointer transition-all inline-flex items-center justify-center gap-1.5 ${
                selected
                  ? "bg-geo-accent text-white border-geo-accent"
                  : "bg-geo-surface text-geo-text border-geo-border hover:border-geo-accent"
              }`}
              title={opt.hint}
            >
              <Icon
                className={`w-3.5 h-3.5 ${selected ? "[&_*]:!stroke-white" : ""}`}
                state={selected ? "default" : "default"}
              />
              {opt.label}
            </button>
          );
        })}
      </div>

      {/* Tag picker */}
      <label className="block text-[11px] font-semibold text-geo-text mb-1">
        Tag
      </label>
      <select
        value={labelMode.tag}
        onChange={(e) => onLabelModeChange({ ...labelMode, tag: e.target.value })}
        className="w-full mb-2 px-2 py-1.5 bg-geo-bg border border-geo-border rounded text-[12px] text-geo-text focus:outline-none focus:border-geo-accent cursor-pointer"
      >
        <optgroup label="Land cover (defaults)">
          {DEFAULT_TAGS.map((t) => (
            <option key={t.name} value={t.name}>
              {t.name}
            </option>
          ))}
        </optgroup>
        {customTags.length > 0 && (
          <optgroup label="Custom">
            {customTags.map((t) => (
              <option key={t} value={t}>
                {t}
              </option>
            ))}
          </optgroup>
        )}
      </select>

      {/* Add custom tag */}
      <div className="flex gap-1.5 mb-2">
        <input
          type="text"
          value={newTag}
          onChange={(e) => setNewTag(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter") {
              e.preventDefault();
              handleAddTag();
            }
          }}
          placeholder="New tag (e.g., mangrove)"
          spellCheck={false}
          className="flex-1 px-2 py-1.5 bg-geo-bg border border-geo-border rounded text-[11px] text-geo-text placeholder:text-geo-dim focus:outline-none focus:border-geo-accent"
        />
        <button
          onClick={handleAddTag}
          disabled={!newTag.trim() || allTags.includes(newTag.trim())}
          className="px-3 py-1.5 text-[11px] font-semibold rounded bg-geo-surface border border-geo-border text-geo-text cursor-pointer transition-all hover:border-geo-accent disabled:opacity-40 disabled:cursor-not-allowed"
        >
          Add
        </button>
      </div>

      {/* Custom-tag chips with ✕ for removal. Removing a tag does NOT delete
          features tagged with it — they keep their tag (rendered with the
          same hash-fallback color as before, since colorForTag is stable on
          the tag string) but the tag disappears from the picker. The user
          can clean up orphaned features manually from the feature list. */}
      {customTags.length > 0 && (
        <div className="flex flex-wrap gap-1 mb-3">
          {customTags.map((t) => {
            const usedBy = features.filter((f) => f.properties.tag === t).length;
            return (
              <span
                key={t}
                className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] font-medium bg-geo-surface border border-geo-border text-geo-text"
                title={
                  usedBy > 0
                    ? `${usedBy} feature${usedBy > 1 ? "s" : ""} use this tag — they'll keep the tag but it'll vanish from the picker.`
                    : "Click ✕ to remove this custom tag."
                }
              >
                <span
                  className="inline-block w-2 h-2 rounded-sm"
                  style={{ background: colorForTag(t, customTags) }}
                />
                {t}
                {usedBy > 0 && (
                  <span className="font-mono text-geo-muted">·{usedBy}</span>
                )}
                <button
                  onClick={() => {
                    if (usedBy > 0) {
                      const ok = window.confirm(
                        `Remove tag "${t}"?\n\n${usedBy} feature${
                          usedBy > 1 ? "s" : ""
                        } use this tag — they'll keep the tag in their properties but "${t}" will no longer appear in the picker. (Delete those features individually from the list below if you also want them gone.)`,
                      );
                      if (!ok) return;
                    }
                    onRemoveCustomTag(t);
                    // If the picker was pointed at the removed tag, fall back
                    // to the first default tag so Start labeling stays valid.
                    if (labelMode.tag === t) {
                      onLabelModeChange({ ...labelMode, tag: DEFAULT_TAGS[0].name });
                    }
                  }}
                  className="ml-0.5 px-1 text-geo-muted hover:text-red-700 cursor-pointer leading-none"
                  title={`Remove "${t}" from custom tags`}
                >
                  ✕
                </button>
              </span>
            );
          })}
        </div>
      )}

      {/* Start/stop button */}
      <button
        onClick={toggleActive}
        className={`w-full py-2.5 mb-3 rounded-lg text-[13px] font-semibold cursor-pointer transition-all shadow-sm ${
          labelMode.active
            ? "bg-red-600 text-white hover:bg-red-700"
            : "text-white hover:shadow-lg hover:-translate-y-0.5"
        }`}
        style={
          labelMode.active
            ? undefined
            : { background: "linear-gradient(135deg, #5b8bb5 0%, #3a6690 100%)" }
        }
        title={
          labelMode.active
            ? "Click to stop drawing labels"
            : `Start drawing ${labelMode.type}s tagged "${labelMode.tag}"`
        }
      >
        {labelMode.active
          ? `■ Stop labeling (${features.length} drawn)`
          : `● Start labeling — ${labelMode.type} · ${labelMode.tag}`}
      </button>

      {/* Feature list summary */}
      {features.length > 0 ? (
        <>
          <div className="flex items-center justify-between mb-2">
            <span className="text-[11px] font-semibold text-geo-text">
              {features.length} feature{features.length > 1 ? "s" : ""} drawn
            </span>
            <button
              onClick={() => {
                if (window.confirm(`Clear all ${features.length} labeled features?`)) {
                  onClearAll();
                }
              }}
              className="text-[10px] font-semibold px-2 py-1 rounded text-geo-muted cursor-pointer hover:text-red-700 hover:bg-red-50 transition-all"
            >
              Clear all
            </button>
          </div>

          {/* Per-tag count summary */}
          <div className="space-y-1 mb-3 max-h-[120px] overflow-y-auto">
            {featuresByTag.map(([tag, count]) => (
              <div
                key={tag}
                className="flex items-center justify-between text-[11px] px-2 py-1 bg-geo-surface border border-geo-border rounded"
              >
                <span className="flex items-center gap-2 text-geo-text">
                  <span
                    className="inline-block w-3 h-3 rounded-sm shrink-0"
                    style={{ background: colorForTag(tag, customTags) }}
                  />
                  {tag}
                </span>
                <span className="font-mono text-geo-muted">{count}</span>
              </div>
            ))}
          </div>

          {/* Per-feature list with delete */}
          <details className="mb-3">
            <summary className="text-[11px] text-geo-muted cursor-pointer hover:text-geo-text">
              All features ({features.length}) — click to expand
            </summary>
            <div className="mt-2 space-y-1 max-h-[180px] overflow-y-auto">
              {features.map((f, i) => (
                <div
                  key={f.id}
                  className="flex items-center justify-between text-[10px] px-2 py-1 bg-geo-bg border border-geo-border rounded font-mono"
                >
                  <span className="truncate">
                    [{i}] {f.properties.label_type} · {f.properties.tag}
                  </span>
                  <button
                    onClick={() => onDeleteFeature(f.id)}
                    className="ml-2 text-geo-muted hover:text-red-700 cursor-pointer"
                    title="Delete this feature"
                  >
                    ✕
                  </button>
                </div>
              ))}
            </div>
          </details>

          {/* Action row — Download + Merge-to-map live side by side. Both
              operate on the SAME FeatureCollection; download ships it out
              to disk, merge pushes it into the map's Added Layer list as
              a toggleable/removable vector layer. Users previously had no
              way to see their labeled features ON the map as a persisted
              layer once drawing mode ended — they either downloaded and
              re-imported, or left the features floating with no layer
              management affordance. */}
          <div className="flex gap-2">
            {onMergeToMapLayers && (
              <button
                onClick={onMergeToMapLayers}
                disabled={mergedToMapLayers}
                className={`flex-1 py-2 rounded-lg text-[12px] font-semibold cursor-pointer shadow-sm transition-all inline-flex items-center justify-center gap-2 border ${
                  mergedToMapLayers
                    ? "bg-geo-success/10 border-geo-success/40 text-geo-success cursor-default"
                    : "bg-geo-surface border-geo-border text-geo-text hover:border-geo-accent hover:text-geo-accent hover:shadow-md hover:-translate-y-px active:translate-y-0"
                }`}
                title={
                  mergedToMapLayers
                    ? "Label set is already on the map — toggle or remove it from the Added Layer panel (top-right of map)"
                    : "Push these features into the Added Layer list so you can toggle / remove them like any other map layer"
                }
              >
                {mergedToMapLayers ? "On map layers" : `+ Add to map layers (${features.length})`}
              </button>
            )}
            <button
              onClick={() => downloadGeoJSON(projectName, features)}
              className="flex-1 py-2 rounded-lg text-[12px] font-semibold text-white cursor-pointer shadow-sm transition-all hover:shadow-md hover:-translate-y-px active:translate-y-0 inline-flex items-center justify-center gap-2"
              style={{ background: "linear-gradient(135deg, #5b8bb5 0%, #3a6690 100%)" }}
              title="Download as GeoJSON FeatureCollection"
            >
              <DownloadGeoJSON className="w-4 h-4 [&_*]:!stroke-white" />
              Download ({features.length})
            </button>
          </div>
        </>
      ) : (
        <p className="text-[11px] text-geo-muted leading-relaxed">
          No features yet. Pick a label type + tag, click <span className="font-semibold text-geo-text">Start labeling</span>,
          then click on the map to draw. Use the existing imagery + OlmoEarth coverage layers (Map tab) as visual reference.
        </p>
      )}
    </Panel>
  );
}
