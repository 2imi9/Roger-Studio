import type { IconState } from './ZoomIn';

interface IconProps {
  state?: IconState;
  className?: string;
}

export function Pan({ state = 'default', className = '' }: IconProps) {
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
      <circle cx="12" cy="12" r="2" fill={fill} fillOpacity={fillOpacity} />
      <polyline points="12,4 12,10" stroke={strokeColor} strokeWidth="1.75" strokeLinecap="round" strokeLinejoin="round" />
      <polyline points="12,14 12,20" stroke={strokeColor} strokeWidth="1.75" strokeLinecap="round" strokeLinejoin="round" />
      <polyline points="4,12 10,12" stroke={strokeColor} strokeWidth="1.75" strokeLinecap="round" strokeLinejoin="round" />
      <polyline points="14,12 20,12" stroke={strokeColor} strokeWidth="1.75" strokeLinecap="round" strokeLinejoin="round" />
      <polygon points="12,2 10,5 14,5" fill={strokeColor} />
      <polygon points="12,22 10,19 14,19" fill={strokeColor} />
      <polygon points="2,12 5,10 5,14" fill={strokeColor} />
      <polygon points="22,12 19,10 19,14" fill={strokeColor} />
    </svg>
  );
}
