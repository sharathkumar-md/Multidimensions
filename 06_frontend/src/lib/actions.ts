'use server';

/**
 * Server Actions for authentication flows.
 * Using Server Actions (POST + CSRF token) instead of GET links
 * prevents CSRF-based forced-logout attacks.
 */

import { signOut } from '@/auth';

/**
 * Signs the user out of NextAuth and redirects to /login.
 * Also clears the Keycloak SSO session via the id_token_hint.
 */
export async function signOutAction(): Promise<void> {
  await signOut({ redirectTo: '/login' });
}
