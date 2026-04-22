interface BadgeProps {
  label: string;
  color?: string;
  variant?: "solid" | "outline" | "dot";
  size?: "sm" | "md";
}

const PRESETS: Record<string, string> = {
  geotiff: "text-geo-success bg-geo-success/10",
  cog: "text-geo-success bg-geo-success/10",
  netcdf: "text-geo-accent bg-geo-accent/10",
  zarr: "text-geo-accent bg-geo-accent/10",
  geopackage: "text-geo-warn bg-geo-warn/10",
  geojson: "text-geo-warn bg-geo-warn/10",
  shapefile: "text-geo-warn bg-geo-warn/10",
  las: "text-purple-400 bg-purple-400/10",
  laz: "text-purple-400 bg-purple-400/10",
  parquet: "text-pink-400 bg-pink-400/10",
  geoparquet: "text-pink-400 bg-pink-400/10",
  csv: "text-geo-muted bg-geo-muted/10",
  tipsv2: "text-geo-success bg-geo-success/10",
  spectral: "text-geo-accent bg-geo-accent/10",
  review: "text-geo-warn bg-geo-warn/10",
};

export function Badge({ label, color, variant = "solid", size = "sm" }: BadgeProps) {
  const preset = PRESETS[label.toLowerCase()] || "text-geo-muted bg-geo-muted/10";
  const sizeClass = size === "sm" ? "text-[9px] px-1.5 py-0.5" : "text-[11px] px-2 py-0.5";

  if (variant === "dot") {
    return (
      <span className="inline-flex items-center gap-1.5">
        <span
          className="w-1.5 h-1.5 rounded-full shrink-0"
          style={{ backgroundColor: color || "currentColor" }}
        />
        <span className={`font-mono ${sizeClass} text-geo-muted`}>{label}</span>
      </span>
    );
  }

  return (
    <span
      className={`inline-block font-mono font-bold uppercase tracking-wider rounded ${sizeClass} ${preset}`}
    >
      {label}
    </span>
  );
}
