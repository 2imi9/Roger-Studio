import type { IconState } from './ZoomIn';

interface IconProps {
  state?: IconState;
  className?: string;
  size?: number;
}

export function StatusCached({ state = 'default', className = '', size = 24 }: IconProps) {
  const strokeColor = state === 'active' ? '#3a6690' : state === 'disabled' ? '#9b9588' : '#2a2620';
  const opacity = state === 'disabled' ? 0.4 : 1;
  const fill = state === 'active' ? '#e2ebf3' : 'none';
  const fillOpacity = state === 'active' ? 0.1 : 0;

  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      className={className}
      style={{ opacity }}
    >
      <rect x="6" y="6" width="12" height="14" rx="1" fill={fill} fillOpacity={fillOpacity} stroke={strokeColor} strokeWidth="1.75" strokeLinecap="round" strokeLinejoin="round" />
      <circle cx="12" cy="13" r="3.5" fill="none" stroke={strokeColor} strokeWidth="1.75" />
      <circle cx="12" cy="13" r="1.5" fill={strokeColor} />
      <line x1="9" y1="9" x2="15" y2="9" stroke={strokeColor} strokeWidth="1.75" strokeLinecap="round" />
    </svg>
  );
}
