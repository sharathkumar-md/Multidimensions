'use client';

import { createContext, useContext, useEffect, useState, useCallback } from 'react';
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

  const applyTheme = useCallback((t: Theme) => {
    const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
    const resolved = t === 'system' ? (prefersDark ? 'dark' : 'light') : t;
    document.documentElement.setAttribute('data-theme', resolved);
    setResolvedTheme(resolved);
    logger.debug(`Theme applied: ${resolved} (preference: ${t})`);
  }, []);

  useEffect(() => {
    const stored = (localStorage.getItem('md-theme') as Theme) ?? 'system';
    setThemeState(stored);
    applyTheme(stored);

    const mq = window.matchMedia('(prefers-color-scheme: dark)');
    const handler = () => {
      if (theme === 'system') applyTheme('system');
    };
    mq.addEventListener('change', handler);
    return () => mq.removeEventListener('change', handler);
  }, [theme, applyTheme]);

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
