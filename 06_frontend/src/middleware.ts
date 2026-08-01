/**
 * Next.js middleware entry point.
 * Delegates to proxy.ts which wraps NextAuth's `auth()` handler.
 * Next.js resolves `src/middleware.ts` (or root `middleware.ts`) as the middleware file.
 */
export { proxy as default, config } from './proxy';
