import type { RouteType } from '@/lib/types';
import styles from './Badge.module.css';

type BadgeVariant = 'primary' | 'success' | 'warning' | 'error' | 'neutral' | 'route';

interface BadgeProps {
  children: React.ReactNode;
  variant?: BadgeVariant;
  route?: RouteType;
  size?: 'sm' | 'md';
  dot?: boolean;
}

const routeConfig: Record<RouteType, { variant: BadgeVariant; label: string }> = {
  LOCAL:  { variant: 'primary', label: 'Catalog' },
  WEB:    { variant: 'success', label: 'Web' },
  NONE:   { variant: 'neutral', label: 'Direct' },
};

export function Badge({ children, variant = 'neutral', route, size = 'sm', dot }: BadgeProps) {
  const resolved = route ? routeConfig[route] : null;
  const finalVariant = resolved?.variant ?? variant;

  return (
    <span className={[styles.badge, styles[finalVariant], styles[size]].join(' ')}>
      {dot && <span className={styles.dot} aria-hidden="true" />}
      {resolved?.label ?? children}
    </span>
  );
}
