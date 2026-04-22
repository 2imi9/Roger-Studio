import type { IconState } from './ZoomIn';

interface IconProps {
  state?: IconState;
  className?: string;
}

export function LayerToggleOff({ state = 'default', className = '' }: IconProps) {
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
      <path d="M 3,12 Q 3,8 6,8 L 18,8 Q 21,8 21,12 Q 21,16 18,16 L 6,16 Q 3,16 3,12 Z" fill={fill} fillOpacity={fillOpacity} stroke={strokeColor} strokeWidth="1.75" strokeLinecap="round" strokeLinejoin="round" />
      <circle cx="12" cy="12" r="3" fill="none" stroke={strokeColor} strokeWidth="1.75" />
      <circle cx="12" cy="12" r="1.5" fill={strokeColor} />
      <line x1="4" y1="4" x2="20" y2="20" stroke={strokeColor} strokeWidth="1.75" strokeLinecap="round" />
    </svg>
  );
}
