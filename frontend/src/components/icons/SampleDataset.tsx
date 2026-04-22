import type { IconState } from './ZoomIn';

interface IconProps {
  state?: IconState;
  className?: string;
}

export function SampleDataset({ state = 'default', className = '' }: IconProps) {
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
      <rect x="5" y="6" width="14" height="12" rx="1" fill={fill} fillOpacity={fillOpacity} stroke={strokeColor} strokeWidth="1.75" strokeLinecap="round" strokeLinejoin="round" />
      <line x1="5" y1="10" x2="19" y2="10" stroke={strokeColor} strokeWidth="1.75" />
      <line x1="5" y1="14" x2="19" y2="14" stroke={strokeColor} strokeWidth="1.75" />
      <line x1="11" y1="10" x2="11" y2="18" stroke={strokeColor} strokeWidth="1.75" />
    </svg>
  );
}
