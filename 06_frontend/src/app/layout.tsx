import type { Metadata, Viewport } from 'next';
import { Inter } from 'next/font/google';
import './globals.css';
import { ThemeProvider } from '@/contexts/ThemeContext';
import { AuthProvider } from '@/components/AuthProvider';
import type { User } from '@/lib/types';

const inter = Inter({
  subsets: ['latin'],
  variable: '--font-inter',
  display: 'swap',
});

export const metadata: Metadata = {
  title: { default: 'MultiDimensions RAG', template: '%s | MultiDimensions' },
  description: 'Industrial product catalog AI assistant — ask anything about the catalog.',
  keywords: ['RAG', 'product catalog', 'AI assistant', 'industrial'],
  robots: { index: false, follow: false }, // internal tool — no crawling
  icons: { icon: '/favicon.ico' },
};

export const viewport: Viewport = {
  width: 'device-width',
  initialScale: 1,
  themeColor: [
    { media: '(prefers-color-scheme: light)', color: '#F8FAFC' },
    { media: '(prefers-color-scheme: dark)',  color: '#0B1120' },
  ],
};

export default async function RootLayout({ children }: { children: React.ReactNode }) {
  // Resolve the server-side session so AuthProvider can pre-populate the Zustand
  // auth store before any client component renders. This eliminates the race where
  // the admin guard in admin/page.tsx saw user=null and never redirected.
  let user: User | null = null;

  if (process.env.NEXT_PUBLIC_AUTH_ENABLED !== 'false') {
    try {
      // Lazy import avoids loading the auth module on the login page when
      // auth is disabled — auth() reads the encrypted JWT cookie (no network call).
      const { auth } = await import('@/auth');
      const session = await auth();
      if (session?.user) {
        user = {
          email: session.user.email ?? '',
          name: session.user.name ?? '',
          roles: session.roles ?? [],
          isAdmin: session.isAdmin ?? false,
          tokenExpiresAt: session.tokenExpiresAt ?? 0,
        };
      }
    } catch {
      // auth() may throw on public pages or when Keycloak is unreachable;
      // fall through with user=null (the middleware handles forced redirects).
    }
  }

  return (
    <html lang="en" suppressHydrationWarning>
      <head>
        {/* Prevent flash of wrong theme — must run before React hydration */}
        <script
          dangerouslySetInnerHTML={{
            __html: `
              (function() {
                try {
                  var t = localStorage.getItem('md-theme') || 'system';
                  var d = t === 'system'
                    ? (window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light')
                    : t;
                  document.documentElement.setAttribute('data-theme', d);
                } catch(e) {}
              })();
            `,
          }}
        />
      </head>
      <body className={inter.variable}>
        <ThemeProvider>
          <AuthProvider user={user}>
            {children}
          </AuthProvider>
        </ThemeProvider>
      </body>
    </html>
  );
}
