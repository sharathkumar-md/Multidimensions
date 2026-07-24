import { auth } from "@/auth"
import { NextResponse } from "next/server"

export const proxy = auth((req) => {
  // Allow local dev bypass
  if (process.env.NEXT_PUBLIC_AUTH_ENABLED === 'false') {
    return NextResponse.next();
  }

  const isLoggedIn = !!req.auth;
  const isAuthPage = req.nextUrl.pathname.startsWith('/login');
  
  const isNextAuthRoute = req.nextUrl.pathname.startsWith('/api/auth');
  const isApiRoute = req.nextUrl.pathname.startsWith('/api/') && !isNextAuthRoute;
  
  // Allow NextAuth routes to process OAuth callbacks
  if (isNextAuthRoute) {
    return NextResponse.next();
  }

  if (!isLoggedIn && !isAuthPage) {
    if (isApiRoute) {
       return new NextResponse(JSON.stringify({ error: "Unauthorized" }), { status: 401 });
    }
    return NextResponse.redirect(new URL('/login', req.nextUrl));
  }
  
  if (isLoggedIn && isAuthPage) {
    return NextResponse.redirect(new URL('/chat', req.nextUrl));
  }

  if (isLoggedIn && isApiRoute) {
    // @ts-ignore
    const token = req.auth.accessToken;
    console.log("[PROXY] Injecting auth token?", !!token, "Auth object keys:", Object.keys(req.auth || {}));
    if (token) {
      const requestHeaders = new Headers(req.headers);
      requestHeaders.set('Authorization', `Bearer ${token}`);
      return NextResponse.next({
        request: { headers: requestHeaders },
      });
    }
  }
  
  return NextResponse.next();
});

export const config = {
  matcher: ["/((?!_next/static|_next/image|favicon.ico).*)"],
};
