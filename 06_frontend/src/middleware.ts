import { auth } from "@/auth"

export default auth((req) => {
  if (process.env.AUTH_ENABLED === 'false') return;
  const isLoggedIn = !!req.auth;
  const isProtected = req.nextUrl.pathname.startsWith(/chat) || req.nextUrl.pathname.startsWith(/admin);
  if (!isLoggedIn && isProtected) {
    return Response.redirect(new URL(/login, req.nextUrl));
  }
})

export const config = {
  matcher: ['/((?!api|_next/static|_next/image|favicon.ico).*)'],
}
