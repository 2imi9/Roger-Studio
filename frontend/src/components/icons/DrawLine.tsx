import type { IconState } from './ZoomIn';

interface IconProps {
  state?: IconState;
  className?: string;
}

export function DrawLine({ state = 'default', className = '' }: IconProps) {
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
      <circle cx="12" cy="12" r="8" fill={fill} fillOpacity={fillOpacity} />
      <polyline points="4,18 8,12 12,14 16,8 20,10" stroke={strokeColor} strokeWidth="1.75" strokeLinecap="round" strokeLinejoin="round" fill="none" />
      <circle cx="4" cy="18" r="1.5" fill={strokeColor} />
      <circle cx="8" cy="12" r="1.5" fill={strokeColor} />
      <circle cx="12" cy="14" r="1.5" fill={strokeColor} />
      <circle cx="16" cy="8" r="1.5" fill={strokeColor} />
      <circle cx="20" cy="10" r="1.5" fill={strokeColor} />
    </svg>
  );
}
