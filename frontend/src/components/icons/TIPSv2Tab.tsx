import type { IconState } from './ZoomIn';

interface IconProps {
  state?: IconState;
  className?: string;
}

// TIPSv2 — image+text dual encoder for zero-shot segmentation.
// Glyph: a camera viewfinder on the left half capturing the image,
// "En" text on the right half representing the language prompt.
// Reads as "capture imagery, segment by text".
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
      {/* Outer frame */}
      <rect x="3" y="6" width="18" height="12" rx="2" fill={fill} fillOpacity={fillOpacity} stroke={strokeColor} strokeWidth="1.75" strokeLinecap="round" strokeLinejoin="round" />
      {/* Vertical divider between camera half and text half */}
      <line x1="12" y1="7" x2="12" y2="17" stroke={strokeColor} strokeWidth="1" strokeDasharray="1.5 1.5" opacity="0.7" />
      {/* Left half: camera viewfinder lens */}
      <circle cx="7.5" cy="12" r="2.6" stroke={strokeColor} strokeWidth="1.5" fill="none" />
      <circle cx="7.5" cy="12" r="0.9" fill={strokeColor} />
      {/* Right half: "En" text glyph — capital E + lowercase n */}
      <text
        x="14"
        y="15.2"
        fontFamily="ui-sans-serif, system-ui, -apple-system, sans-serif"
        fontSize="6.8"
        fontWeight="700"
        fill={strokeColor}
        letterSpacing="-0.4"
      >
        En
      </text>
    </svg>
  );
}
