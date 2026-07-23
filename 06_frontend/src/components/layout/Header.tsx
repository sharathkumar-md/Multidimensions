'use client';

import { Sun, Moon, Monitor, Cpu } from 'lucide-react';
import { useTheme } from '@/contexts/ThemeContext';
import { useChatStore, useUiStore } from '@/lib/store';
import styles from './Header.module.css';

interface HeaderProps {
  sessionTitle?: string;
}

export function Header({ sessionTitle }: HeaderProps) {
  const { theme, setTheme } = useTheme();
  const { isStreaming } = useChatStore();

  const themeOptions = [
    { value: 'light' as const,  icon: <Sun size={14} />,     label: 'Light' },
    { value: 'dark'  as const,  icon: <Moon size={14} />,    label: 'Dark'  },
    { value: 'system' as const, icon: <Monitor size={14} />, label: 'System' },
  ];

  return (
    <header className={styles.header}>
      <div className={styles.left}>
        <h1 className={styles.title} title={sessionTitle}>
          {sessionTitle ?? 'MultiDimensions RAG'}
        </h1>
        {isStreaming && (
          <span className={styles.streamingBadge} aria-live="polite">
            <span className={styles.streamingDot} />
            Generating…
          </span>
        )}
      </div>

      <div className={styles.right}>
        {/* Model badge */}
        <div className={styles.modelBadge} title="Active AI model">
          <Cpu size={13} />
          <span>Qwen3-8B · 4-bit</span>
        </div>

        {/* Theme switcher */}
        <div className={styles.themeGroup} role="group" aria-label="Theme">
          {themeOptions.map((opt) => (
            <button
              key={opt.value}
              onClick={() => setTheme(opt.value)}
              className={[styles.themeBtn, theme === opt.value ? styles.active : ''].join(' ')}
              aria-label={`${opt.label} theme`}
              aria-pressed={theme === opt.value}
              title={`${opt.label} theme`}
            >
              {opt.icon}
            </button>
          ))}
        </div>
      </div>
    </header>
  );
}
