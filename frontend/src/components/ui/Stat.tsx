interface StatProps {
  label: string;
  value: string | number;
  unit?: string;
  color?: string;
  size?: "sm" | "lg";
}

export function Stat({ label, value, unit, color, size = "sm" }: StatProps) {
  const valSize = size === "lg" ? "text-[26px]" : "text-[20px]";
  return (
    <div className="text-center py-2">
      <div
        className={`font-semibold tracking-tight ${valSize}`}
        style={color ? { color } : { color: "var(--color-geo-text)" }}
      >
        {value}
        {unit && <span className="text-[13px] text-geo-muted ml-1 font-normal">{unit}</span>}
      </div>
      <div className="text-[13px] text-geo-muted mt-1.5 font-medium">{label}</div>
    </div>
  );
}

export function StatRow({ label, value, color }: { label: string; value: string | number; color?: string }) {
  return (
    <div className="flex justify-between items-center py-1.5">
      <span className="text-[14px] text-geo-muted">{label}</span>
      <span
        className="text-[14px] font-mono font-semibold"
        style={color ? { color } : undefined}
      >
        {value}
      </span>
    </div>
  );
}
