'use client';

import { createContext, useContext, useEffect, useRef, useState, useCallback } from 'react';
import logger from '@/lib/logger';

type Theme = 'light' | 'dark' | 'system';

interface ThemeContextValue {
  theme: Theme;
  resolvedTheme: 'light' | 'dark';
  setTheme: (theme: Theme) => void;
}

const ThemeContext = createContext<ThemeContextValue>({
  theme: 'system',
  resolvedTheme: 'light',
  setTheme: () => {},
});

export function ThemeProvider({ children }: { children: React.ReactNode }) {
  const [theme, setThemeState] = useState<Theme>('system');
  const [resolvedTheme, setResolvedTheme] = useState<'light' | 'dark'>('light');

  /**
   * Keep a ref in sync with the latest theme value.
   * The OS color-scheme change handler reads from this ref instead of capturing
   * `theme` via closure, which would go stale after the first theme switch.
   * (F-06: previously the effect re-subscribed on every theme change but still
   *  captured a stale closure value inside the handler.)
   */
  const themeRef = useRef<Theme>(theme);
  themeRef.current = theme;

  const applyTheme = useCallback((t: Theme) => {
    const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
    const resolved = t === 'system' ? (prefersDark ? 'dark' : 'light') : t;
    document.documentElement.setAttribute('data-theme', resolved);
    setResolvedTheme(resolved);
    logger.debug(`Theme applied: ${resolved} (preference: ${t})`);
  }, []); // stable — no captured state

  // Register the OS theme-change listener once.
  // applyTheme is stable (empty deps useCallback), so this effect runs once on mount.
  useEffect(() => {
    const stored = (localStorage.getItem('md-theme') as Theme) ?? 'system';
    setThemeState(stored);
    applyTheme(stored);

    const mq = window.matchMedia('(prefers-color-scheme: dark)');
    // Handler reads themeRef.current — always reflects the latest theme state
    const handler = () => {
      if (themeRef.current === 'system') applyTheme('system');
    };
    mq.addEventListener('change', handler);
    return () => mq.removeEventListener('change', handler);
  }, [applyTheme]); // applyTheme is stable → single registration

  const setTheme = useCallback((t: Theme) => {
    localStorage.setItem('md-theme', t);
    setThemeState(t);
    applyTheme(t);
  }, [applyTheme]);

  return (
    <ThemeContext.Provider value={{ theme, resolvedTheme, setTheme }}>
      {children}
    </ThemeContext.Provider>
  );
}

export const useTheme = () => useContext(ThemeContext);
