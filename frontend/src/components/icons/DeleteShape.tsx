import type { IconState } from './ZoomIn';

interface IconProps {
  state?: IconState;
  className?: string;
}

export function DeleteShape({ state = 'default', className = '' }: IconProps) {
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
      <path d="M 8,7 L 8,20 L 16,20 L 16,7" fill={fill} fillOpacity={fillOpacity} stroke={strokeColor} strokeWidth="1.75" strokeLinecap="round" strokeLinejoin="round" />
      <line x1="6" y1="7" x2="18" y2="7" stroke={strokeColor} strokeWidth="1.75" strokeLinecap="round" />
      <line x1="10" y1="4" x2="14" y2="4" stroke={strokeColor} strokeWidth="1.75" strokeLinecap="round" />
      <line x1="10" y1="11" x2="10" y2="16" stroke={strokeColor} strokeWidth="1.75" strokeLinecap="round" />
      <line x1="14" y1="11" x2="14" y2="16" stroke={strokeColor} strokeWidth="1.75" strokeLinecap="round" />
    </svg>
  );
}
