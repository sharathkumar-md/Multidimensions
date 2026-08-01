import NextAuth from "next-auth"
import Keycloak from "next-auth/providers/keycloak"

/**
 * Augment NextAuth built-in types so that session.accessToken, session.isAdmin,
 * and token.roles are properly typed — removing the need for @ts-ignore.
 */
declare module "next-auth" {
  interface Session {
    accessToken?: string;
    isAdmin?: boolean;
    roles?: string[];
    tokenExpiresAt?: number;
  }
}

declare module "next-auth/jwt" {
  interface JWT {
    accessToken?: string;
    idToken?: string;
    isAdmin?: boolean;
    roles?: string[];
    expiresAt?: number;
  }
}

export const { handlers, signIn, signOut, auth } = NextAuth({
  providers: [
    Keycloak({
      clientId: process.env.KEYCLOAK_CLIENT_ID ?? "rag-sales-bot",
      clientSecret: process.env.KEYCLOAK_CLIENT_SECRET ?? "",
      issuer: process.env.KEYCLOAK_ISSUER ?? "http://localhost:8081/realms/multidimensions",
    }),
  ],
  callbacks: {
    async jwt({ token, account, profile }) {
      if (account) {
        token.accessToken = account.access_token;
        token.idToken = account.id_token;
        token.expiresAt = account.expires_at;

        // Extract Keycloak realm roles from the OIDC profile.
        // Keycloak includes roles under realm_access.roles in the ID token profile.
        const kc = profile as { realm_access?: { roles?: string[] } } | undefined;
        const roles = kc?.realm_access?.roles ?? [];
        token.roles = roles;
        token.isAdmin = roles.includes('admin');
      }
      return token;
    },
    async session({ session, token }) {
      session.accessToken = token.accessToken;
      session.isAdmin = token.isAdmin ?? false;
      session.roles = token.roles ?? [];
      // expiresAt is a Unix timestamp in seconds; multiply to ms for consistency
      session.tokenExpiresAt = (token.expiresAt ?? 0) * 1000;
      return session;
    },
  },
})
