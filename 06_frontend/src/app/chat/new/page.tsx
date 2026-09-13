'use client';

import { useEffect, useRef, useCallback, useState } from 'react';
import { useRouter } from 'next/navigation';
import { useChatStore } from '@/lib/store';
import { createSession } from '@/lib/api';
import logger from '@/lib/logger';

/**
 * /chat/new — creates a new session and immediately redirects into it.
 * This is the landing page when no session is selected.
 */
export default function NewChatPage() {
  const router = useRouter();
  const { addSession, setActiveSession } = useChatStore();
  const creating = useRef(false);
  const [status, setStatus] = useState('Starting conversation…');

  const create = useCallback(async () => {
    if (creating.current) return;
    creating.current = true;

    const MAX_RETRIES = 10;
    for (let attempt = 0; attempt < MAX_RETRIES; attempt++) {
      try {
        const session = await createSession();
        addSession(session);
        setActiveSession(session.id);
        logger.info('Auto-created new session', { id: session.id });
        router.replace(`/chat/${session.id}`);
        return;
      } catch (err: unknown) {
        const msg = (err as Error).message;
        logger.warn(`Failed to create session (attempt ${attempt + 1}/${MAX_RETRIES})`, { error: msg });
        if (attempt < MAX_RETRIES - 1) {
          const delay = Math.min(1000 * Math.pow(2, attempt), 30000);
          setStatus(`Backend is starting up… retrying (${attempt + 2}/${MAX_RETRIES})`);
          await new Promise((r) => setTimeout(r, delay));
        } else {
          setStatus('Could not reach backend. Please refresh the page.');
          logger.error('Exhausted retries creating session', { error: msg });
        }
      }
    }
    creating.current = false;
  }, [router, addSession, setActiveSession]);

  useEffect(() => { create(); }, [create]);

  return (
    <div style={{
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      height: '100%',
      color: 'var(--color-text-tertiary)',
      fontFamily: 'var(--font-sans)',
      fontSize: '0.875rem',
      gap: '0.5rem',
    }}>
      <span style={{
        width: '16px', height: '16px',
        border: '2px solid transparent',
        borderTopColor: 'var(--color-primary)',
        borderRadius: '50%',
        animation: 'spin 0.7s linear infinite',
        display: 'inline-block',
      }} />
      {status}
    </div>
  );
}
