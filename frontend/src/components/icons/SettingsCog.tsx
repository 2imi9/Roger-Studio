import type { IconState } from './ZoomIn';

interface IconProps {
  state?: IconState;
  className?: string;
}

export function SettingsCog({ state = 'default', className = '' }: IconProps) {
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
      <circle cx="12" cy="12" r="6" fill={fill} fillOpacity={fillOpacity} stroke={strokeColor} strokeWidth="1.75" />
      <path d="M 8,12 L 10,12" stroke={strokeColor} strokeWidth="1.75" strokeLinecap="round" />
      <path d="M 14,12 L 16,12" stroke={strokeColor} strokeWidth="1.75" strokeLinecap="round" />
      <path d="M 9,10 L 10.5,12 L 9,14" stroke={strokeColor} strokeWidth="1.75" strokeLinecap="round" strokeLinejoin="round" fill="none" />
      <path d="M 15,10 L 13.5,12 L 15,14" stroke={strokeColor} strokeWidth="1.75" strokeLinecap="round" strokeLinejoin="round" fill="none" />
      <circle cx="12" cy="12" r="1" fill={strokeColor} />
    </svg>
  );
}
