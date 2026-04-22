import { useState } from "react";

export interface LabelClass {
  name: string;
  prompt: string;
  color: string;
}

// Preset templates for common use cases
const PRESETS: Record<string, LabelClass[]> = {
  "Land Cover": [
    { name: "Forest", prompt: "dense forest or woodland with trees", color: "#228b22" },
    { name: "Cropland", prompt: "agricultural cropland or farmland", color: "#f0e68c" },
    { name: "Grassland", prompt: "grassland or meadow or pasture", color: "#90ee90" },
    { name: "Urban", prompt: "urban area with buildings and roads", color: "#808080" },
    { name: "Water", prompt: "water body such as river lake or ocean", color: "#1e90ff" },
    { name: "Barren", prompt: "barren land or desert or bare soil", color: "#d2b48c" },
  ],
  "Karst / Geohazard": [
    { name: "Sinkhole", prompt: "sinkhole or ground collapse depression", color: "#ef4444" },
    { name: "Cave Entrance", prompt: "cave opening or entrance in rock", color: "#a855f7" },
    { name: "Surface Depression", prompt: "shallow surface depression or dip in terrain", color: "#f97316" },
    { name: "Stable Ground", prompt: "stable flat ground with no deformation", color: "#22c55e" },
    { name: "Bedrock Outcrop", prompt: "exposed rock or bedrock outcrop", color: "#78716c" },
  ],
  "Solar Energy": [
    { name: "High Solar Potential", prompt: "open flat area with high sun exposure", color: "#f59e0b" },
    { name: "Low Solar Potential", prompt: "shaded area or north-facing slope", color: "#6b7280" },
    { name: "Existing Solar Farm", prompt: "solar panel arrays or solar farm", color: "#3b82f6" },
    { name: "Excluded (Water)", prompt: "water body unsuitable for solar", color: "#1e90ff" },
    { name: "Excluded (Forest)", prompt: "forest area unsuitable for solar", color: "#228b22" },
  ],
  "Urban Planning": [
    { name: "Residential", prompt: "residential housing area with houses", color: "#f59e0b" },
    { name: "Commercial", prompt: "commercial buildings offices or shops", color: "#3b82f6" },
    { name: "Industrial", prompt: "industrial area with factories or warehouses", color: "#6b7280" },
    { name: "Green Space", prompt: "park garden or green open space", color: "#22c55e" },
    { name: "Road/Infrastructure", prompt: "road highway or transportation infrastructure", color: "#a3a3a3" },
  ],
};

const PALETTE = ["#ef4444", "#f59e0b", "#22c55e", "#3b82f6", "#a855f7", "#ec4899", "#6b7280", "#78716c", "#1e90ff", "#228b22", "#f0e68c", "#d2b48c"];

interface LabelClassEditorProps {
  classes: LabelClass[];
  onChange: (classes: LabelClass[]) => void;
}

export function LabelClassEditor({ classes, onChange }: LabelClassEditorProps) {
  const [expanded, setExpanded] = useState(false);

  const addNew = () => {
    const idx = classes.length;
    onChange([...classes, {
      name: `Class ${idx + 1}`,
      prompt: `description of class ${idx + 1}`,
      color: PALETTE[idx % PALETTE.length],
    }]);
  };

  const remove = (i: number) => {
    onChange(classes.filter((_, j) => j !== i));
  };

  const update = (i: number, field: keyof LabelClass, val: string) => {
    const next = [...classes];
    next[i] = { ...next[i], [field]: val };
    onChange(next);
  };

  const loadPreset = (key: string) => {
    onChange([...PRESETS[key]]);
  };

  return (
    <div className="mb-2.5">
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full flex justify-between items-center py-1.5 bg-transparent border-none text-geo-muted cursor-pointer text-xs"
      >
        <span>Label Classes ({classes.length})</span>
        <span className="text-[10px]">{expanded ? "\u25B2" : "\u25BC"}</span>
      </button>

      {expanded && (
        <div className="bg-geo-surface border border-geo-border rounded-lg p-2.5">
          {/* Presets */}
          <div className="flex flex-wrap gap-1 mb-2">
            {Object.keys(PRESETS).map((key) => (
              <button
                key={key}
                onClick={() => loadPreset(key)}
                className="px-2 py-0.5 text-[10px] bg-geo-border text-geo-text border-none rounded cursor-pointer hover:bg-geo-elevated transition-colors"
              >
                {key}
              </button>
            ))}
          </div>

          {/* Class list */}
          {classes.map((cls, i) => (
            <div
              key={i}
              className="flex items-center gap-1 mb-1"
            >
              <input
                type="color"
                value={cls.color}
                onChange={(e) => update(i, "color", e.target.value)}
                className="w-5 h-5 border-none p-0 cursor-pointer bg-transparent"
              />
              <input
                value={cls.name}
                onChange={(e) => update(i, "name", e.target.value)}
                placeholder="Name"
                className="w-[70px] px-1.5 py-0.5 bg-geo-bg border border-geo-border rounded text-geo-text text-[11px]"
              />
              <input
                value={cls.prompt}
                onChange={(e) => update(i, "prompt", e.target.value)}
                placeholder="Text prompt for TIPSv2"
                className="flex-1 px-1.5 py-0.5 bg-geo-bg border border-geo-border rounded text-geo-muted text-[10px]"
              />
              <button
                onClick={() => remove(i)}
                className="bg-transparent border-none text-geo-dim cursor-pointer text-xs px-0.5 hover:text-geo-danger"
              >
                x
              </button>
            </div>
          ))}

          <button
            onClick={addNew}
            className="mt-1 px-2.5 py-1 text-[10px] bg-geo-border text-geo-muted border-none rounded cursor-pointer hover:bg-geo-elevated transition-colors"
          >
            + Add Class
          </button>
        </div>
      )}
    </div>
  );
}

export { PRESETS };
