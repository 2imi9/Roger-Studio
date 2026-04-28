import type { IconState } from './ZoomIn';

interface IconProps {
  state?: IconState;
  className?: string;
}

// TIPSv2 — image+text dual encoder. Glyph: a tile (raster) bracketed by
// quote marks (text) to suggest "label-by-prompt".
export function TIPSv2Tab({ state = 'default', className = '' }: IconProps) {
  const strokeColor = state === 'active' ? '#3a6690' : state === 'disabled' ? '#9b9588' : '#2a2620';
  const opacity = state === 'disabled' ? 0.4 : 1;
  const fill = state === 'active' ? '#e2ebf3' : 'none';
  const fillOpacity = state === 'active' ? 0.1 : 0;

  return (
    <svg
      width="24"
      height="24"
      viewBox="0 0 24 24"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      className={className}
      style={{ opacity }}
    >
      {/* Raster tile in the middle */}
      <rect x="7" y="7" width="10" height="10" rx="1.5" fill={fill} fillOpacity={fillOpacity} stroke={strokeColor} strokeWidth="1.75" strokeLinecap="round" strokeLinejoin="round" />
      {/* Patch grid hints */}
      <line x1="10.3" y1="7" x2="10.3" y2="17" stroke={strokeColor} strokeWidth="0.9" strokeLinecap="round" opacity="0.55" />
      <line x1="13.7" y1="7" x2="13.7" y2="17" stroke={strokeColor} strokeWidth="0.9" strokeLinecap="round" opacity="0.55" />
      <line x1="7" y1="10.3" x2="17" y2="10.3" stroke={strokeColor} strokeWidth="0.9" strokeLinecap="round" opacity="0.55" />
      <line x1="7" y1="13.7" x2="17" y2="13.7" stroke={strokeColor} strokeWidth="0.9" strokeLinecap="round" opacity="0.55" />
      {/* Left text-bracket (quote-style) */}
      <path d="M4 9 L4 15" stroke={strokeColor} strokeWidth="2" strokeLinecap="round" />
      <path d="M4 9 L5.5 9" stroke={strokeColor} strokeWidth="2" strokeLinecap="round" />
      <path d="M4 15 L5.5 15" stroke={strokeColor} strokeWidth="2" strokeLinecap="round" />
      {/* Right text-bracket */}
      <path d="M20 9 L20 15" stroke={strokeColor} strokeWidth="2" strokeLinecap="round" />
      <path d="M20 9 L18.5 9" stroke={strokeColor} strokeWidth="2" strokeLinecap="round" />
      <path d="M20 15 L18.5 15" stroke={strokeColor} strokeWidth="2" strokeLinecap="round" />
    </svg>
  );
}
