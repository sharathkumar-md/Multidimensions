'use client';

import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { oneLight, oneDark } from 'react-syntax-highlighter/dist/esm/styles/prism';
import { Copy, Check } from 'lucide-react';
import { useState, useCallback } from 'react';
import type { Message } from '@/lib/types';
import { Badge } from '@/components/ui/Badge';
import { SourcesPanel } from './SourcesPanel';
import { ProductGallery } from './ProductGallery';
import { useTheme } from '@/contexts/ThemeContext';
import styles from './AssistantMessage.module.css';

interface AssistantMessageProps {
  message: Message;
}

function CopyButton({ text }: { text: string }) {
  const [copied, setCopied] = useState(false);
  const handleCopy = useCallback(async () => {
    await navigator.clipboard.writeText(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  }, [text]);

  return (
    <button onClick={handleCopy} className={styles.copyBtn} aria-label="Copy code" title="Copy">
      {copied ? <Check size={13} /> : <Copy size={13} />}
    </button>
  );
}

export function AssistantMessage({ message }: AssistantMessageProps) {
  const { resolvedTheme } = useTheme();

  const syntaxStyle = resolvedTheme === 'dark' ? oneDark : oneLight;

  return (
    <article className={styles.wrap} aria-label="Assistant message">
      <div className={styles.avatar} aria-hidden="true">AI</div>

      <div className={styles.content}>
        {/* Route badge */}
        {message.route && !message.isStreaming && (
          <div className={styles.meta}>
            <Badge route={message.route} />
          </div>
        )}

        {/* Markdown body */}
        <div className={[styles.bubble, message.isStreaming ? styles.streaming : ''].join(' ')}>
          <div className="prose">
            <ReactMarkdown
              remarkPlugins={[remarkGfm]}
              components={{
                code({ node, className, children, ...props }) {
                  const match = /language-(\w+)/.exec(className ?? '');
                  const code = String(children).replace(/\n$/, '');
                  const isInline = !match;

                  if (isInline) {
                    return <code className={className} {...props}>{children}</code>;
                  }

                  return (
                    <div className={styles.codeBlock}>
                      <div className={styles.codeHeader}>
                        <span className={styles.codeLang}>{match[1]}</span>
                        <CopyButton text={code} />
                      </div>
                      <SyntaxHighlighter
                        style={syntaxStyle}
                        language={match[1]}
                        PreTag="div"
                        customStyle={{
                          margin: 0,
                          borderRadius: '0 0 6px 6px',
                          fontSize: '0.8rem',
                          background: 'transparent',
                        }}
                      >
                        {code}
                      </SyntaxHighlighter>
                    </div>
                  );
                },
              }}
            >
              {message.content}
            </ReactMarkdown>
            {message.isStreaming && (
              <span className={styles.cursor} aria-hidden="true" />
            )}
          </div>
        </div>

        {/* Product images */}
        {message.productImages && message.productImages.length > 0 && !message.isStreaming && (
          <ProductGallery images={message.productImages} />
        )}

        {/* Sources */}
        {message.sources && message.sources.length > 0 && !message.isStreaming && (
          <SourcesPanel sources={message.sources} />
        )}
      </div>
    </article>
  );
}
