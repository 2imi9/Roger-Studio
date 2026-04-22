import type { ReactNode } from "react";

interface PanelProps {
  children: ReactNode;
  className?: string;
  border?: boolean;
}

export function Panel({ children, className = "", border = true }: PanelProps) {
  return (
    <div
      className={`bg-gradient-panel rounded-xl p-5 ${
        border ? "border border-geo-border shadow-soft" : ""
      } ${className}`}
    >
      {children}
    </div>
  );
}

export function SectionTitle({ children }: { children: ReactNode }) {
  return (
    <h3 className="text-[15px] font-semibold text-geo-text mb-3 tracking-tight">
      {children}
    </h3>
  );
}

export function Divider() {
  return <div className="border-t border-geo-border my-6" />;
}
