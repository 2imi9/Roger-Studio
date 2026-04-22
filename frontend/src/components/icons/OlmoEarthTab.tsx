import type { IconState } from './ZoomIn';

interface IconProps {
  state?: IconState;
  className?: string;
}

export function OlmoEarthTab({ state = 'default', className = '' }: IconProps) {
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
      <rect x="4" y="4" width="16" height="16" rx="2" fill={fill} fillOpacity={fillOpacity} stroke={strokeColor} strokeWidth="1.75" strokeLinecap="round" strokeLinejoin="round" />
      <line x1="12" y1="7" x2="12" y2="17" stroke={strokeColor} strokeWidth="2.5" strokeLinecap="round" />
      <line x1="7" y1="12" x2="17" y2="12" stroke={strokeColor} strokeWidth="2.5" strokeLinecap="round" />
      <circle cx="12" cy="12" r="2" fill={strokeColor} />
    </svg>
  );
}
