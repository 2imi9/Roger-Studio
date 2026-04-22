import type { IconState } from './ZoomIn';

interface IconProps {
  state?: IconState;
  className?: string;
  size?: number;
}

export function StatusNoKey({ state = 'default', className = '', size = 24 }: IconProps) {
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
      <circle cx="8" cy="9" r="3" fill={fill} fillOpacity={fillOpacity} stroke={strokeColor} strokeWidth="1.75" />
      <path d="M 11,9 L 19,9 L 19,13 L 17,13 L 17,11 L 15,11 L 15,13 L 13,13 L 13,9" stroke={strokeColor} strokeWidth="1.75" strokeLinecap="round" strokeLinejoin="round" fill="none" />
      <line x1="5" y1="5" x2="19" y2="19" stroke={strokeColor} strokeWidth="1.75" strokeLinecap="round" />
    </svg>
  );
}
