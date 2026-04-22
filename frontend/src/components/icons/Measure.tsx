import type { IconState } from './ZoomIn';

interface IconProps {
  state?: IconState;
  className?: string;
}

export function Measure({ state = 'default', className = '' }: IconProps) {
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
      <rect x="3" y="9" width="20" height="6" rx="1" transform="rotate(-30 12 12)" fill={fill} fillOpacity={fillOpacity} stroke={strokeColor} strokeWidth="1.75" strokeLinecap="round" strokeLinejoin="round" />
      <line x1="7" y1="12" x2="7" y2="14" transform="rotate(-30 12 12)" stroke={strokeColor} strokeWidth="1.75" strokeLinecap="round" />
      <line x1="10" y1="12" x2="10" y2="13" transform="rotate(-30 12 12)" stroke={strokeColor} strokeWidth="1.75" strokeLinecap="round" />
      <line x1="13" y1="12" x2="13" y2="14" transform="rotate(-30 12 12)" stroke={strokeColor} strokeWidth="1.75" strokeLinecap="round" />
      <line x1="16" y1="12" x2="16" y2="13" transform="rotate(-30 12 12)" stroke={strokeColor} strokeWidth="1.75" strokeLinecap="round" />
      <line x1="19" y1="12" x2="19" y2="14" transform="rotate(-30 12 12)" stroke={strokeColor} strokeWidth="1.75" strokeLinecap="round" />
    </svg>
  );
}
