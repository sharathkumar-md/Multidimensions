'use client';

import { Sun, Moon, Monitor, Cpu, Download } from 'lucide-react';
import { useTheme } from '@/contexts/ThemeContext';
import { useChatStore, useUiStore } from '@/lib/store';
import styles from './Header.module.css';
import { useCallback } from 'react';

interface HeaderProps {
  sessionTitle?: string;
}

export function Header({ sessionTitle }: HeaderProps) {
  const { theme, setTheme } = useTheme();
  const { isStreaming, messages, activeSessionId } = useChatStore();

  const themeOptions = [
    { value: 'light' as const,  icon: <Sun size={14} />,     label: 'Light' },
    { value: 'dark'  as const,  icon: <Moon size={14} />,    label: 'Dark'  },
    { value: 'system' as const, icon: <Monitor size={14} />, label: 'System' },
  ];

  const handleExport = useCallback(() => {
    if (!activeSessionId) return;
    const sessionMessages = messages[activeSessionId] || [];
    if (sessionMessages.length === 0) return;

    let text = `Chat Export: ${sessionTitle || 'MultiDimensions RAG'}\n\n`;
    sessionMessages.forEach((msg) => {
      text += `--- ${msg.role.toUpperCase()} ---\n${msg.content}\n\n`;
    });

    const blob = new Blob([text], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `Chat_Export_${new Date().toISOString().split('T')[0]}.txt`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }, [activeSessionId, messages, sessionTitle]);

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
        {/* Export badge */}
        <button className={styles.exportBtn} onClick={handleExport} title="Export Chat as TXT" aria-label="Export Chat">
          <Download size={13} />
          <span>Export</span>
        </button>

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
