import type { IconState } from './ZoomIn';

interface IconProps {
  state?: IconState;
  className?: string;
  size?: number;
}

export function StatusOnline({ state = 'default', className = '', size = 24 }: IconProps) {
  const strokeColor = state === 'active' ? '#3a6690' : state === 'disabled' ? '#9b9588' : '#2a2620';
  const opacity = state === 'disabled' ? 0.4 : 1;

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
      <circle cx="12" cy="12" r="3" fill={strokeColor} />
      <circle cx="12" cy="12" r="6" stroke={strokeColor} strokeWidth="1.75" opacity="0.3" />
    </svg>
  );
}
