'use client';

import { useEffect, useRef, useCallback } from 'react';
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

  const create = useCallback(async () => {
    if (creating.current) return;
    creating.current = true;
    try {
      const session = await createSession();
      addSession(session);
      setActiveSession(session.id);
      logger.info('Auto-created new session', { id: session.id });
      router.replace(`/chat/${session.id}`);
    } catch (err: unknown) {
      logger.error('Failed to auto-create session', { error: (err as Error).message });
      // Fallback: stay on new chat with a friendly message
    }
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
      Starting conversation…
    </div>
  );
}
