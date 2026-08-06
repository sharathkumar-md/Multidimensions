'use server';

/**
 * Server Actions for authentication flows.
 * Using Server Actions (POST + CSRF token) instead of GET links
 * prevents CSRF-based forced-logout attacks.
 */

import { signOut } from '@/auth';

/**
 * Signs the user out of NextAuth and redirects to /login.
 *
 * NOTE: This clears the NextAuth session cookie but does NOT terminate the
 * upstream Keycloak SSO session. The Keycloak session remains valid, so the
 * user may re-authenticate without re-entering credentials until the Keycloak
 * session itself expires. Full SSO logout requires a Keycloak end_session_endpoint
 * call with an id_token_hint — not currently implemented.
 */
export async function signOutAction(): Promise<void> {
  await signOut({ redirectTo: '/login' });
}
