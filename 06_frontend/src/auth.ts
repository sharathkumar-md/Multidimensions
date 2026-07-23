import NextAuth from "next-auth"
import Keycloak from "next-auth/providers/keycloak"

export const { handlers, signIn, signOut, auth } = NextAuth({
  providers: [
    Keycloak({
      clientId: process.env.KEYCLOAK_CLIENT_ID ?? "rag-sales-bot",
      clientSecret: process.env.KEYCLOAK_CLIENT_SECRET ?? "",
      issuer: process.env.KEYCLOAK_ISSUER ?? "http://localhost:8081/realms/multidimensions",
    }),
  ],
  callbacks: {
    async jwt({ token, account }) {
      if (account) {
        token.accessToken = account.access_token
        token.idToken = account.id_token
      }
      return token
    },
    async session({ session, token }) {
      // @ts-ignore
      session.accessToken = token.accessToken
      return session
    },
  },
})
