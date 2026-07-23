'use client';

import { useState } from 'react';
import { ChevronDown, FileText, Globe } from 'lucide-react';
import type { Source } from '@/lib/types';
import styles from './SourcesPanel.module.css';

interface SourcesPanelProps {
  sources: Source[];
}

export function SourcesPanel({ sources }: SourcesPanelProps) {
  const [open, setOpen] = useState(false);

  return (
    <div className={styles.wrap}>
      <button
        onClick={() => setOpen((o) => !o)}
        className={styles.toggle}
        aria-expanded={open}
        aria-controls="sources-list"
      >
        <FileText size={13} />
        <span>{sources.length} source{sources.length !== 1 ? 's' : ''}</span>
        <ChevronDown
          size={13}
          className={[styles.chevron, open ? styles.open : ''].join(' ')}
        />
      </button>

      {open && (
        <ul id="sources-list" className={styles.list} role="list">
          {sources.map((src, i) => {
            const isWeb = src.sourceDoc.startsWith('http');
            return (
              <li key={i} className={styles.item}>
                <div className={styles.itemHeader}>
                  {isWeb ? <Globe size={12} className={styles.iconWeb} /> : <FileText size={12} className={styles.iconDoc} />}
                  {isWeb ? (
                    <a
                      href={src.sourceDoc}
                      target="_blank"
                      rel="noopener noreferrer"
                      className={styles.docName}
                    >
                      {src.sourceDoc}
                    </a>
                  ) : (
                    <span className={styles.docName}>{src.sourceDoc}</span>
                  )}
                  {!isWeb && (
                    <span className={styles.page}>p.&nbsp;{src.pageNum}</span>
                  )}
                </div>
                <p className={styles.snippet}>{src.snippet}</p>
              </li>
            );
          })}
        </ul>
      )}
    </div>
  );
}
