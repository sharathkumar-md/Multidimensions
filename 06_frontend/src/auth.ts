import NextAuth from "next-auth"
import Google from "next-auth/providers/google"

declare module "next-auth" {
  interface Session {
    accessToken?: string;
    isAdmin?: boolean;
    roles?: string[];
  }
}

declare module "next-auth/jwt" {
  interface JWT {
    accessToken?: string;
    isAdmin?: boolean;
    roles?: string[];
  }
}

export const { handlers, signIn, signOut, auth } = NextAuth({
  providers: [
    Google({
      clientId: process.env.GOOGLE_CLIENT_ID,
      clientSecret: process.env.GOOGLE_CLIENT_SECRET,
    }),
  ],
  callbacks: {
    async jwt({ token, account, profile }) {
      if (account) {
        // Send Google's secure ID token to the FastAPI backend
        token.accessToken = account.id_token; 
        
        // Lock down Admin Hub access
        const ADMIN_EMAILS = ["research@multidimensions.co.in"];
        const userEmail = token.email || profile?.email || "";
        token.isAdmin = ADMIN_EMAILS.includes(userEmail);
        token.roles = token.isAdmin ? ["admin", "sales"] : ["sales"];
      }
      return token;
    },
    async session({ session, token }) {
      session.accessToken = token.accessToken;
      session.isAdmin = token.isAdmin;
      session.roles = token.roles;
      return session;
    },
  },
})
