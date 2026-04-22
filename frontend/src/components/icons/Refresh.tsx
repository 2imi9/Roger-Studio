import type { IconState } from './ZoomIn';

interface IconProps {
  state?: IconState;
  className?: string;
}

export function Refresh({ state = 'default', className = '' }: IconProps) {
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
      <path d="M 4,12 Q 4,7 8,5" stroke={strokeColor} strokeWidth="1.75" strokeLinecap="round" strokeLinejoin="round" fill="none" />
      <path d="M 20,12 Q 20,17 16,19" stroke={strokeColor} strokeWidth="1.75" strokeLinecap="round" strokeLinejoin="round" fill="none" />
      <polyline points="8,8 8,5 5,5" stroke={strokeColor} strokeWidth="1.75" strokeLinecap="round" strokeLinejoin="round" />
      <polyline points="16,16 16,19 19,19" stroke={strokeColor} strokeWidth="1.75" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}
