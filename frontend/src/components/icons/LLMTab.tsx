import type { IconState } from './ZoomIn';

interface IconProps {
  state?: IconState;
  className?: string;
}

export function LLMTab({ state = 'default', className = '' }: IconProps) {
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
      {/* Shifted up by 4 px from the original Figma export — the chat-bubble
          glyph was drawn in the lower half of the 24×24 viewBox (y 11→22),
          which made the icon sit visibly below the tab text baseline when
          scaled down to 16×16 next to "LLM". Now centered vertically. */}
      <path d="M 5,13 L 5,9 Q 5,7 7,7 L 17,7 Q 19,7 19,9 L 19,13 Q 19,15 17,15 L 9,15 L 5,18 Z" fill={fill} fillOpacity={fillOpacity} stroke={strokeColor} strokeWidth="1.75" strokeLinecap="round" strokeLinejoin="round" />
      <circle cx="9" cy="11" r="0.75" fill={strokeColor} />
      <circle cx="12" cy="11" r="0.75" fill={strokeColor} />
      <circle cx="15" cy="11" r="0.75" fill={strokeColor} />
    </svg>
  );
}
