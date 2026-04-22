import type { IconState } from './ZoomIn';

interface IconProps {
  state?: IconState;
  className?: string;
}

export function Globe3DTab({ state = 'default', className = '' }: IconProps) {
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
      <circle cx="12" cy="12" r="8" fill={fill} fillOpacity={fillOpacity} stroke={strokeColor} strokeWidth="1.75" />
      <ellipse cx="12" cy="12" rx="4" ry="8" stroke={strokeColor} strokeWidth="1.75" fill="none" />
      <ellipse cx="12" cy="12" rx="8" ry="4" stroke={strokeColor} strokeWidth="1.75" fill="none" />
      <line x1="4" y1="12" x2="20" y2="12" stroke={strokeColor} strokeWidth="1.75" />
      <line x1="12" y1="4" x2="12" y2="20" stroke={strokeColor} strokeWidth="1.75" />
    </svg>
  );
}
