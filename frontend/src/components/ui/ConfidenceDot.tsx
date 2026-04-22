interface ConfidenceDotProps {
  value: number; // 0-1
  showLabel?: boolean;
  size?: "sm" | "md";
}

export function ConfidenceDot({ value, showLabel = false, size = "sm" }: ConfidenceDotProps) {
  const color =
    value >= 0.7 ? "bg-geo-success" :
    value >= 0.4 ? "bg-geo-warn" :
    "bg-geo-danger";

  const label =
    value >= 0.7 ? "High" :
    value >= 0.4 ? "Medium" :
    "Low";

  const dotSize = size === "sm" ? "w-2 h-2" : "w-2.5 h-2.5";

  return (
    <span className="inline-flex items-center gap-1.5">
      <span className={`${dotSize} rounded-full ${color} shrink-0`} />
      {showLabel && (
        <span className="text-[10px] font-mono text-geo-muted">{label}</span>
      )}
    </span>
  );
}

export function confidenceColor(value: number): string {
  if (value >= 0.7) return "#10b981";
  if (value >= 0.4) return "#f59e0b";
  return "#ef4444";
}
