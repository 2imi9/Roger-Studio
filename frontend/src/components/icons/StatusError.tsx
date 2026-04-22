import type { IconState } from './ZoomIn';

interface IconProps {
  state?: IconState;
  className?: string;
  size?: number;
}

export function StatusError({ state = 'default', className = '', size = 24 }: IconProps) {
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
      <path d="M 12,4 L 20,18 L 4,18 Z" fill={fill} fillOpacity={fillOpacity} stroke={strokeColor} strokeWidth="1.75" strokeLinecap="round" strokeLinejoin="round" />
      <line x1="12" y1="10" x2="12" y2="14" stroke={strokeColor} strokeWidth="1.75" strokeLinecap="round" />
      <circle cx="12" cy="16.5" r="0.75" fill={strokeColor} />
    </svg>
  );
}
