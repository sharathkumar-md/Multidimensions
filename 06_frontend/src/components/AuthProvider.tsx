'use client';

/**
 * AuthProvider — hydrates the client-side useAuthStore from the server-resolved session.
 *
 * The root layout calls auth() (server-side) and passes the user object here.
 * This component sets it in Zustand immediately on mount, so all client components
 * that read useAuthStore().user see a populated value on first render — not null.
 *
 * This fixes the broken admin guard (F-03) which relied on user being non-null.
 */

import { useEffect } from 'react';
import { useAuthStore } from '@/lib/store';
import type { User } from '@/lib/types';

interface AuthProviderProps {
  user: User | null;
  children: React.ReactNode;
}

export function AuthProvider({ user, children }: AuthProviderProps) {
  const setUser = useAuthStore((s) => s.setUser);
  const setLoading = useAuthStore((s) => s.setLoading);

  useEffect(() => {
    setUser(user);
    setLoading(false);
  }, [user, setUser, setLoading]);

  return <>{children}</>;
}
