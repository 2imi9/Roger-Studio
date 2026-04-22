import { useState } from "react";
import type { BBox } from "../types";

interface AreaSelectorProps {
  onSelect: (bbox: BBox) => void;
}

export function AreaSelector({ onSelect }: AreaSelectorProps) {
  const [west, setWest] = useState("");
  const [south, setSouth] = useState("");
  const [east, setEast] = useState("");
  const [north, setNorth] = useState("");

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    const bbox: BBox = {
      west: parseFloat(west),
      south: parseFloat(south),
      east: parseFloat(east),
      north: parseFloat(north),
    };
    if (
      !isNaN(bbox.west) &&
      !isNaN(bbox.south) &&
      !isNaN(bbox.east) &&
      !isNaN(bbox.north)
    ) {
      onSelect(bbox);
    }
  };

  return (
    <form onSubmit={handleSubmit} style={{ display: "flex", flexDirection: "column", gap: 8 }}>
      <h4 style={{ margin: "0 0 4px", fontSize: 13, color: "#94a3b8" }}>
        Enter Coordinates
      </h4>
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 6 }}>
        {[
          { label: "West", value: west, set: setWest, placeholder: "-122.5" },
          { label: "East", value: east, set: setEast, placeholder: "-122.0" },
          { label: "South", value: south, set: setSouth, placeholder: "37.5" },
          { label: "North", value: north, set: setNorth, placeholder: "38.0" },
        ].map(({ label, value, set, placeholder }) => (
          <div key={label}>
            <label style={{ fontSize: 11, color: "#64748b" }}>{label}</label>
            <input
              type="number"
              step="any"
              value={value}
              onChange={(e) => set(e.target.value)}
              placeholder={placeholder}
              style={{
                width: "100%",
                padding: "6px 8px",
                background: "#0f172a",
                border: "1px solid #334155",
                borderRadius: 4,
                color: "#e2e8f0",
                fontSize: 12,
                fontFamily: "monospace",
                boxSizing: "border-box",
              }}
            />
          </div>
        ))}
      </div>
      <button
        type="submit"
        style={{
          padding: "8px 0",
          background: "#334155",
          color: "#e2e8f0",
          border: "none",
          borderRadius: 6,
          cursor: "pointer",
          fontSize: 13,
        }}
      >
        Set Area
      </button>
    </form>
  );
}
