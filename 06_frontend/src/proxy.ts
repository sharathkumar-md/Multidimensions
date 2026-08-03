import { auth } from "@/auth"
import { NextResponse } from "next/server"

export const proxy = auth((req) => {
  // Allow local dev bypass — AUTH_ENABLED must be explicitly 'false' (string).
  // NOTE: must NOT use NEXT_PUBLIC_ prefix — that would expose the flag in the client bundle
  // and allow an attacker to enumerate the bypass condition (N-02).
  if (process.env.AUTH_ENABLED === 'false') {
    return NextResponse.next();
  }

  const isLoggedIn = !!req.auth;
  const isAuthPage = req.nextUrl.pathname.startsWith('/login');
  const isNextAuthRoute = req.nextUrl.pathname.startsWith('/api/auth');
  const isApiRoute = req.nextUrl.pathname.startsWith('/api/') && !isNextAuthRoute;

  // Classify admin routes: the /admin page and all /api/admin/* endpoints
  const isAdminPageRoute = req.nextUrl.pathname.startsWith('/admin');
  const isAdminApiRoute = isApiRoute && req.nextUrl.pathname.startsWith('/api/admin');

  // Allow NextAuth routes to process OAuth callbacks without auth check
  if (isNextAuthRoute) {
    return NextResponse.next();
  }

  // Unauthenticated: redirect or reject
  if (!isLoggedIn && !isAuthPage) {
    if (isApiRoute) {
      return new NextResponse(JSON.stringify({ error: "Unauthorized" }), { status: 401 });
    }
    return NextResponse.redirect(new URL('/login', req.nextUrl));
  }

  // Already logged in — redirect away from login page
  if (isLoggedIn && isAuthPage) {
    return NextResponse.redirect(new URL('/chat', req.nextUrl));
  }

  // Server-side admin-only enforcement — client-side guard alone is insufficient
  if (isLoggedIn && (isAdminPageRoute || isAdminApiRoute) && !req.auth?.isAdmin) {
    if (isAdminApiRoute) {
      return new NextResponse(JSON.stringify({ error: "Forbidden" }), { status: 403 });
    }
    return NextResponse.redirect(new URL('/chat', req.nextUrl));
  }

  // Inject Authorization header for all authenticated API requests so the
  // FastAPI backend can validate the Keycloak access token
  if (isLoggedIn && isApiRoute) {
    const token = req.auth?.accessToken;
    if (token) {
      const requestHeaders = new Headers(req.headers);
      requestHeaders.set('Authorization', `Bearer ${token}`);
      return NextResponse.next({
        request: { headers: requestHeaders },
      });
    }
    // Session exists but access token is absent (e.g. after a silent Keycloak token
    // refresh failure). Reject rather than forwarding an unsigned request to the backend.
    // This prevents the backend from receiving requests without a valid JWT (N-06).
    return new NextResponse(
      JSON.stringify({ error: 'Session token unavailable — please sign in again.' }),
      { status: 401 }
    );
  }

  return NextResponse.next();
});

export const config = {
  matcher: ["/((?!_next/static|_next/image|favicon.ico).*)"],
};
