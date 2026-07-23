'use client';

import { useRef, useState, useCallback, useEffect } from 'react';
import { Send, Globe, StopCircle } from 'lucide-react';
import styles from './ChatInput.module.css';

interface ChatInputProps {
  onSend: (question: string, webSearch: boolean) => void;
  onStop?: () => void;
  isStreaming?: boolean;
  disabled?: boolean;
  placeholder?: string;
}

export function ChatInput({
  onSend,
  onStop,
  isStreaming = false,
  disabled = false,
  placeholder = 'Ask about the product catalog…',
}: ChatInputProps) {
  const [value, setValue] = useState('');
  const [webSearch, setWebSearch] = useState(false);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // Auto-resize textarea
  const autoResize = useCallback(() => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = 'auto';
    el.style.height = `${Math.min(el.scrollHeight, 160)}px`;
  }, []);

  useEffect(() => { autoResize(); }, [value, autoResize]);

  const handleSend = useCallback(() => {
    const trimmed = value.trim();
    if (!trimmed || isStreaming || disabled) return;
    onSend(trimmed, webSearch);
    setValue('');
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.focus();
    }
  }, [value, isStreaming, disabled, onSend, webSearch]);

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <div className={styles.wrap}>
      <div className={styles.inner}>
        {/* Web search toggle */}
        <button
          type="button"
          onClick={() => setWebSearch((w) => !w)}
          className={[styles.webToggle, webSearch ? styles.webActive : ''].join(' ')}
          aria-label={webSearch ? 'Web search enabled — click to disable' : 'Enable web search'}
          aria-pressed={webSearch}
          title={webSearch ? 'Web search ON' : 'Web search OFF'}
          disabled={disabled || isStreaming}
        >
          <Globe size={16} />
          {webSearch && <span className={styles.webLabel}>Web</span>}
        </button>

        {/* Text input */}
        <textarea
          ref={textareaRef}
          value={value}
          onChange={(e) => setValue(e.target.value)}
          onKeyDown={handleKeyDown}
          onInput={autoResize}
          placeholder={placeholder}
          rows={1}
          className={styles.textarea}
          disabled={disabled || isStreaming}
          aria-label="Chat message"
          aria-multiline="true"
          aria-disabled={disabled || isStreaming}
          spellCheck
        />

        {/* Send / Stop */}
        {isStreaming ? (
          <button
            type="button"
            onClick={onStop}
            className={[styles.sendBtn, styles.stopBtn].join(' ')}
            aria-label="Stop generating"
            title="Stop"
          >
            <StopCircle size={18} />
          </button>
        ) : (
          <button
            type="button"
            onClick={handleSend}
            disabled={!value.trim() || disabled}
            className={styles.sendBtn}
            aria-label="Send message"
            title="Send (Enter)"
          >
            <Send size={16} />
          </button>
        )}
      </div>
      <p className={styles.hint}>
        <kbd>Enter</kbd> to send · <kbd>Shift+Enter</kbd> for new line
      </p>
    </div>
  );
}
