'use client';

import { useEffect, useRef, useState } from 'react';
import Link from 'next/link';
import { usePathname, useRouter } from 'next/navigation';
import {
  MessageSquare, Plus, Search, Trash2, Settings,
  LogOut, ChevronLeft, ChevronRight, Database,
} from 'lucide-react';
import { useChatStore, useAuthStore, useUiStore } from '@/lib/store';
import { createSession, deleteSession, getSessions } from '@/lib/api';
import { signOutAction } from '@/lib/actions';
import logger from '@/lib/logger';
import styles from './Sidebar.module.css';

function formatDate(dateStr: string): string {
  const d = new Date(dateStr);
  const now = new Date();
  const diff = now.getTime() - d.getTime();
  if (diff < 86_400_000) return 'Today';
  if (diff < 172_800_000) return 'Yesterday';
  if (diff < 604_800_000) return d.toLocaleDateString('en', { weekday: 'long' });
  return d.toLocaleDateString('en', { month: 'short', day: 'numeric' });
}

export function Sidebar() {
  const router = useRouter();
  const pathname = usePathname();
  const { sessions, setSessions, addSession, removeSession, setActiveSession, activeSessionId } = useChatStore();
  const { user } = useAuthStore();
  const { sidebarOpen, setSidebarOpen, toggleSidebar, searchQuery, setSearchQuery } = useUiStore();
  const [isCreating, setIsCreating] = useState(false);
  const searchRef = useRef<HTMLInputElement>(null);

  // Load sessions on mount
  useEffect(() => {
    getSessions()
      .then(setSessions)
      .catch((e) => logger.error('Failed to load sessions', { error: e.message }));
  }, [setSessions]);

  const handleNewChat = async () => {
    if (isCreating) return;
    setIsCreating(true);
    try {
      const session = await createSession();
      addSession(session);
      setActiveSession(session.id);
      router.push(`/chat/${session.id}`);
      logger.info('Created new session', { id: session.id });
    } catch (e: unknown) {
      logger.error('Failed to create session', { error: (e as Error).message });
    } finally {
      setIsCreating(false);
    }
  };

  const handleDelete = async (e: React.MouseEvent, sessionId: string) => {
    e.preventDefault();
    e.stopPropagation();
    // Optimistic removal — snapshot current sessions for rollback on failure
    const snapshot = [...sessions];
    removeSession(sessionId);
    if (activeSessionId === sessionId) router.push('/chat');
    try {
      await deleteSession(sessionId);
      logger.info('Deleted session', { id: sessionId });
    } catch (err: unknown) {
      // Rollback: restore true server state rather than leaving a phantom deletion
      getSessions().then(setSessions).catch(() => {});
      logger.warn('Failed to delete session — rolled back', { error: (err as Error).message });
    }
  };

  // Group sessions by date
  const filtered = sessions.filter((s) =>
    s.title.toLowerCase().includes(searchQuery.toLowerCase()),
  );
  const groups: Record<string, typeof sessions> = {};
  for (const sess of filtered) {
    const label = formatDate(sess.updatedAt ?? sess.createdAt);
    if (!groups[label]) groups[label] = [];
    groups[label].push(sess);
  }

  if (!sidebarOpen) {
    return (
      <aside className={[styles.sidebar, styles.collapsed].join(' ')}>
        <button
          onClick={toggleSidebar}
          className={styles.collapseToggle}
          aria-label="Expand sidebar"
          title="Expand sidebar"
        >
          <ChevronRight size={16} />
        </button>
        <button
          onClick={handleNewChat}
          className={styles.collapseNewChat}
          aria-label="New chat"
          title="New chat"
          disabled={isCreating}
        >
          <Plus size={18} />
        </button>
      </aside>
    );
  }

  return (
    <aside className={styles.sidebar}>
      {/* Header */}
      <div className={styles.header}>
        <Link href="/chat" className={styles.logo}>
          <Database size={18} className={styles.logoIcon} />
          <span className={styles.logoText}>MultiDimensions</span>
        </Link>
        <button
          onClick={toggleSidebar}
          className={styles.collapseToggle}
          aria-label="Collapse sidebar"
          title="Collapse sidebar"
        >
          <ChevronLeft size={16} />
        </button>
      </div>

      {/* New Chat */}
      <div className={styles.actions}>
        <button
          onClick={handleNewChat}
          disabled={isCreating}
          className={styles.newChatBtn}
          aria-label="Start new conversation"
        >
          <Plus size={16} />
          <span>New chat</span>
        </button>
      </div>

      {/* Search */}
      <div className={styles.searchWrap}>
        <Search size={14} className={styles.searchIcon} />
        <input
          ref={searchRef}
          type="search"
          placeholder="Search conversations…"
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          className={styles.searchInput}
          aria-label="Search conversations"
        />
      </div>

      {/* Session list */}
      <nav className={styles.nav} aria-label="Conversations">
        {Object.keys(groups).length === 0 ? (
          <p className={styles.empty}>
            {searchQuery ? 'No conversations match.' : 'No conversations yet.'}
          </p>
        ) : (
          Object.entries(groups).map(([label, group]) => (
            <div key={label} className={styles.group}>
              <p className={styles.groupLabel}>{label}</p>
              {group.map((sess) => {
                const isActive = pathname === `/chat/${sess.id}`;
                return (
                  <div
                    key={sess.id}
                    className={[styles.sessionItem, isActive ? styles.active : ''].join(' ')}
                  >
                    <Link
                      href={`/chat/${sess.id}`}
                      className={styles.sessionLink}
                      onClick={() => setActiveSession(sess.id)}
                      title={sess.title}
                    >
                      <MessageSquare size={14} className={styles.sessionIcon} />
                      <span className={styles.sessionTitle}>{sess.title}</span>
                    </Link>
                    <button
                      onClick={(e) => handleDelete(e, sess.id)}
                      className={styles.deleteBtn}
                      aria-label={`Delete conversation: ${sess.title}`}
                      title="Delete"
                    >
                      <Trash2 size={13} />
                    </button>
                  </div>
                );
              })}
            </div>
          ))
        )}
      </nav>

      {/* Footer */}
      <div className={styles.footer}>
        {user?.isAdmin && (
          <Link href="/admin" className={styles.footerLink}>
            <Settings size={15} />
            <span>Admin Hub</span>
          </Link>
        )}
        <div className={styles.userRow}>
          <div className={styles.avatar} aria-hidden="true">
            {user?.name?.charAt(0).toUpperCase() ?? '?'}
          </div>
          <div className={styles.userInfo}>
            <p className={styles.userName}>{user?.name ?? 'User'}</p>
            <p className={styles.userEmail}>{user?.email ?? ''}</p>
          </div>
          {/* Use a form + server action so sign-out is a CSRF-protected POST,
              not a spoofable GET that any third-party page can trigger */}
          <form action={signOutAction}>
            <button
              type="submit"
              className={styles.logoutBtn}
              aria-label="Log out"
              title="Log out"
              onClick={() => logger.info('Logout triggered')}
            >
              <LogOut size={15} />
            </button>
          </form>
        </div>
      </div>
    </aside>
  );
}
