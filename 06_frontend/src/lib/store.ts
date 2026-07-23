import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import type { Session, Message, User, StreamToken } from './types';
import logger from './logger';

// ── Chat Store ────────────────────────────────────────────────────────────

interface ChatState {
  sessions: Session[];
  activeSessionId: string | null;
  messages: Record<string, Message[]>;
  isStreaming: boolean;
  streamingSessionId: string | null;

  // Actions
  setSessions: (sessions: Session[]) => void;
  addSession: (session: Session) => void;
  removeSession: (id: string) => void;
  setActiveSession: (id: string) => void;
  setMessages: (sessionId: string, messages: Message[]) => void;
  addMessage: (sessionId: string, message: Message) => void;
  appendToken: (sessionId: string, messageId: string, token: string) => void;
  finalizeMessage: (sessionId: string, messageId: string, data: Partial<Message>) => void;
  setStreaming: (streaming: boolean, sessionId?: string) => void;
  updateSessionTitle: (sessionId: string, title: string) => void;
}

export const useChatStore = create<ChatState>()(
  persist(
    (set, get) => ({
      sessions: [],
      activeSessionId: null,
      messages: {},
      isStreaming: false,
      streamingSessionId: null,

      setSessions: (sessions) => {
        logger.debug('Store: set sessions', { count: sessions.length });
        set({ sessions });
      },

      addSession: (session) =>
        set((s) => ({ sessions: [session, ...s.sessions] })),

      removeSession: (id) =>
        set((s) => {
          const sessions = s.sessions.filter((sess) => sess.id !== id);
          const messages = { ...s.messages };
          delete messages[id];
          const activeSessionId =
            s.activeSessionId === id ? (sessions[0]?.id ?? null) : s.activeSessionId;
          return { sessions, messages, activeSessionId };
        }),

      setActiveSession: (id) => {
        logger.debug('Store: set active session', { id });
        set({ activeSessionId: id });
      },

      setMessages: (sessionId, messages) =>
        set((s) => ({ messages: { ...s.messages, [sessionId]: messages } })),

      addMessage: (sessionId, message) =>
        set((s) => ({
          messages: {
            ...s.messages,
            [sessionId]: [...(s.messages[sessionId] ?? []), message],
          },
        })),

      appendToken: (sessionId, messageId, token) =>
        set((s) => {
          const msgs = s.messages[sessionId] ?? [];
          return {
            messages: {
              ...s.messages,
              [sessionId]: msgs.map((m) =>
                m.id === messageId ? { ...m, content: m.content + token } : m,
              ),
            },
          };
        }),

      finalizeMessage: (sessionId, messageId, data) =>
        set((s) => {
          const msgs = s.messages[sessionId] ?? [];
          return {
            messages: {
              ...s.messages,
              [sessionId]: msgs.map((m) =>
                m.id === messageId ? { ...m, ...data, isStreaming: false } : m,
              ),
            },
          };
        }),

      setStreaming: (streaming, sessionId) =>
        set({ isStreaming: streaming, streamingSessionId: sessionId ?? null }),

      updateSessionTitle: (sessionId, title) =>
        set((s) => ({
          sessions: s.sessions.map((sess) =>
            sess.id === sessionId ? { ...sess, title } : sess,
          ),
        })),
    }),
    {
      name: 'md-chat-store',
      partialize: (s) => ({
        activeSessionId: s.activeSessionId,
        // Don't persist messages — load from API on mount
      }),
    },
  ),
);

// ── Auth Store ────────────────────────────────────────────────────────────

interface AuthState {
  user: User | null;
  isLoading: boolean;
  setUser: (user: User | null) => void;
  setLoading: (loading: boolean) => void;
}

export const useAuthStore = create<AuthState>()((set) => ({
  user: null,
  isLoading: true,
  setUser: (user) => {
    logger.info('Auth: user set', { email: user?.email ?? 'null' });
    set({ user });
  },
  setLoading: (isLoading) => set({ isLoading }),
}));

// ── UI Store ──────────────────────────────────────────────────────────────

interface UiState {
  sidebarOpen: boolean;
  searchQuery: string;
  setSidebarOpen: (open: boolean) => void;
  toggleSidebar: () => void;
  setSearchQuery: (q: string) => void;
}

export const useUiStore = create<UiState>()((set) => ({
  sidebarOpen: true,
  searchQuery: '',
  setSidebarOpen: (open) => set({ sidebarOpen: open }),
  toggleSidebar: () => set((s) => ({ sidebarOpen: !s.sidebarOpen })),
  setSearchQuery: (searchQuery) => set({ searchQuery }),
}));
