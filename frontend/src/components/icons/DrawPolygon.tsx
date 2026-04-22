import type { IconState } from './ZoomIn';

interface IconProps {
  state?: IconState;
  className?: string;
}

export function DrawPolygon({ state = 'default', className = '' }: IconProps) {
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
      <polygon points="12,4 19,8 18,16 9,19 5,13" fill={fill} fillOpacity={fillOpacity} stroke={strokeColor} strokeWidth="1.75" strokeLinecap="round" strokeLinejoin="round" />
      <circle cx="12" cy="4" r="1.5" fill={strokeColor} />
      <circle cx="19" cy="8" r="1.5" fill={strokeColor} />
      <circle cx="18" cy="16" r="1.5" fill={strokeColor} />
      <circle cx="9" cy="19" r="1.5" fill={strokeColor} />
      <circle cx="5" cy="13" r="1.5" fill={strokeColor} />
    </svg>
  );
}
