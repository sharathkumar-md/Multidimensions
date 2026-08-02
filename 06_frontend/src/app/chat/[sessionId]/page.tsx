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

function EmptyState({ onSuggestionClick }: { onSuggestionClick: (value: string) => void }) {
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
          <button key={s} className={styles.suggestion} onClick={() => onSuggestionClick(s)}>
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

  const handleSend = useCallback(async (question: string, webSearch: boolean = false) => {
    if (isStreaming) return;

    // Optimistically add user message — Fix 011: crypto.randomUUID() avoids same-ms collision
    const userMsg: Message = {
      id: crypto.randomUUID(),
      role: 'user',
      content: question,
      createdAt: new Date().toISOString(),
    };
    addMessage(sessionId, userMsg);
    setShowThinking(true);

    // Placeholder AI message (streaming) — Fix 011: separate UUID, no Date.now() collision
    const aiMsgId = crypto.randomUUID();
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

    // Track whether aiMsg was added to the store — used by the abort/error handlers
    // to know whether finalizeMessage must be called to clear isStreaming: true.
    let messageAdded = false;

    try {
      for await (const raw of streamChat(sessionId, question, abort.signal, webSearch)) {
        let parsed: StreamToken;
        try { parsed = JSON.parse(raw); } catch { continue; }

        if (parsed.error) {
          logger.error('Stream error from server', { error: parsed.error });
          break;
        }

        if (parsed.token) {
          if (!messageAdded) {
            setShowThinking(false);
            addMessage(sessionId, aiMsg);
            messageAdded = true;
          }
          appendToken(sessionId, aiMsgId, parsed.token);
        }

        if (parsed.done) {
          finalizeMessage(sessionId, aiMsgId, {
            sources: parsed.sources,
            productImages: parsed.productImages,
            route: parsed.route,
          });
          logger.info('Stream complete', { sessionId, route: parsed.route });
          break;
        }
      }
    } catch (e: unknown) {
      if ((e as Error).name === 'AbortError') {
        // User stopped the stream — finalize the message if it was already in the store,
        // otherwise it stays stuck with isStreaming:true and a permanent blinking cursor.
        if (messageAdded) {
          finalizeMessage(sessionId, aiMsgId, { isStreaming: false });
          logger.info('Stream aborted mid-stream', { sessionId });
        }
      } else {
        logger.error('Stream failed', { error: (e as Error).message });
        if (messageAdded) {
          finalizeMessage(sessionId, aiMsgId, {
            content: '⚠️ Something went wrong. Please try again.',
          });
        }
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
            <EmptyState onSuggestionClick={(s) => handleSend(s, false)} />
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
