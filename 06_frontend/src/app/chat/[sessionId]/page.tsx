'use client';

import { useEffect, useRef, useCallback, useState } from 'react';
import { useParams, useRouter } from 'next/navigation';
import { Header } from '@/components/layout/Header';
import { UserMessage } from '@/components/chat/UserMessage';
import { AssistantMessage } from '@/components/chat/AssistantMessage';
import { ThinkingIndicator } from '@/components/chat/ThinkingIndicator';
import { ChatInput } from '@/components/chat/ChatInput';
import { useChatStore } from '@/lib/store';
import { getMessages, streamChat } from '@/lib/api';
import type { Message, StreamToken } from '@/lib/types';
import logger from '@/lib/logger';
import styles from './page.module.css';

function EmptyState() {
  return (
    <div className={styles.empty}>
      <div className={styles.emptyIcon}>💬</div>
      <h2 className={styles.emptyTitle}>Start a conversation</h2>
      <p className={styles.emptyDesc}>
        Ask anything about the product catalog. I can find specifications,
        compare products, and search the web for the latest information.
      </p>
      <div className={styles.suggestions}>
        {[
          'What are the specifications of the DP-7 actuator?',
          'Compare all available pressure sensors by range',
          'What is the latest price for Model X?',
          'List all products suitable for high-temperature environments',
        ].map((s) => (
          <button key={s} className={styles.suggestion} onClick={() => {}}>
            {s}
          </button>
        ))}
      </div>
    </div>
  );
}

export default function SessionPage() {
  const params = useParams<{ sessionId: string }>();
  const router = useRouter();
  const sessionId = params.sessionId;

  const {
    messages, sessions, setMessages, addMessage, appendToken,
    finalizeMessage, setStreaming, isStreaming, setActiveSession,
  } = useChatStore();

  const abortRef = useRef<AbortController | null>(null);
  const bottomRef = useRef<HTMLDivElement>(null);
  const [showThinking, setShowThinking] = useState(false);

  const sessionMessages = messages[sessionId] ?? [];
  const session = sessions.find((s) => s.id === sessionId);

  // Load messages on mount
  useEffect(() => {
    setActiveSession(sessionId);
    if (messages[sessionId]) return; // already loaded
    getMessages(sessionId)
      .then((msgs) => setMessages(sessionId, msgs))
      .catch((e) => {
        logger.warn('Failed to load messages', { error: e.message, sessionId });
        router.replace('/chat');
      });
  }, [sessionId, setActiveSession, messages, setMessages, router]);

  // Auto-scroll to bottom
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [sessionMessages.length, showThinking]);

  const handleSend = useCallback(async (question: string) => {
    if (isStreaming) return;

    // Optimistically add user message
    const userMsg: Message = {
      id: `user-${Date.now()}`,
      role: 'user',
      content: question,
      createdAt: new Date().toISOString(),
    };
    addMessage(sessionId, userMsg);
    setShowThinking(true);

    // Placeholder AI message (streaming)
    const aiMsgId = `ai-${Date.now()}`;
    const aiMsg: Message = {
      id: aiMsgId,
      role: 'assistant',
      content: '',
      createdAt: new Date().toISOString(),
      isStreaming: true,
    };

    const abort = new AbortController();
    abortRef.current = abort;
    setStreaming(true, sessionId);

    try {
      let firstToken = true;

      for await (const raw of streamChat(sessionId, question, abort.signal)) {
        let parsed: StreamToken;
        try { parsed = JSON.parse(raw); } catch { continue; }

        if (parsed.error) {
          logger.error('Stream error from server', { error: parsed.error });
          break;
        }

        if (parsed.token) {
          if (firstToken) {
            setShowThinking(false);
            addMessage(sessionId, aiMsg);
            firstToken = false;
          }
          appendToken(sessionId, aiMsgId, parsed.token);
        }

        if (parsed.done) {
          finalizeMessage(sessionId, aiMsgId, {
            sources: parsed.sources,
            productImages: parsed.productImages ?? parsed.product_images,
            route: parsed.route,
          });
          logger.info('Stream complete', { sessionId, route: parsed.route });
          break;
        }
      }
    } catch (e: unknown) {
      if ((e as Error).name !== 'AbortError') {
        logger.error('Stream failed', { error: (e as Error).message });
        finalizeMessage(sessionId, aiMsgId, {
          content: '⚠️ Something went wrong. Please try again.',
        });
      }
    } finally {
      setShowThinking(false);
      setStreaming(false);
      abortRef.current = null;
    }
  }, [sessionId, isStreaming, addMessage, appendToken, finalizeMessage, setStreaming]);

  const handleStop = useCallback(() => {
    abortRef.current?.abort();
    logger.info('Stream aborted by user', { sessionId });
  }, [sessionId]);

  return (
    <div className={styles.page}>
      <Header sessionTitle={session?.title} />

      <div className={styles.messages} role="log" aria-label="Conversation" aria-live="polite">
        <div className={styles.messagesInner}>
          {sessionMessages.length === 0 && !showThinking ? (
            <EmptyState />
          ) : (
            sessionMessages.map((msg) =>
              msg.role === 'user' ? (
                <UserMessage key={msg.id} message={msg} />
              ) : (
                <AssistantMessage key={msg.id} message={msg} />
              ),
            )
          )}
          {showThinking && <ThinkingIndicator />}
          <div ref={bottomRef} aria-hidden="true" />
        </div>
      </div>

      <ChatInput
        onSend={handleSend}
        onStop={handleStop}
        isStreaming={isStreaming}
      />
    </div>
  );
}
