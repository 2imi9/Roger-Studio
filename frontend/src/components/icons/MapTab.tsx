import type { IconState } from './ZoomIn';

interface IconProps {
  state?: IconState;
  className?: string;
}

export function MapTab({ state = 'default', className = '' }: IconProps) {
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
      <path d="M 4,6 L 4,18 L 9,20 L 15,17 L 20,19 L 20,7 L 15,5 L 9,8 Z" fill={fill} fillOpacity={fillOpacity} stroke={strokeColor} strokeWidth="1.75" strokeLinecap="round" strokeLinejoin="round" />
      <line x1="9" y1="8" x2="9" y2="20" stroke={strokeColor} strokeWidth="1.75" />
      <line x1="15" y1="5" x2="15" y2="17" stroke={strokeColor} strokeWidth="1.75" />
    </svg>
  );
}
