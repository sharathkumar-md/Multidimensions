import type { NextConfig } from 'next';

/**
 * Authoritative Next.js configuration.
 * next.config.ts takes precedence over next.config.js in Next ≥15.
 */
const nextConfig: NextConfig = {
  output: 'standalone',

  // Allow images from the FastAPI backend and Cloud Run services
  images: {
    remotePatterns: [
      {
        protocol: 'http',
        hostname: 'localhost',
        port: '8000',
        pathname: '/**',
      },
      {
        protocol: 'https',
        hostname: '*.run.app',
        pathname: '/**',
      },
    ],
    // Allow unoptimized local file paths from the RAG pipeline
    unoptimized: true,
  },

  // Rewrites: proxy /api/* to FastAPI backend (fallback = only when no Next.js API route matches)
  async rewrites() {
    return {
      fallback: [
        {
          source: '/api/:path*',
          destination: `${process.env.NEXT_PUBLIC_API_URL ?? 'http://localhost:8000'}/api/:path*`,
        },
      ],
    };
  },

  // Security headers applied to every response
  async headers() {
    const csp = [
      "default-src 'self'",
      "script-src 'self' 'unsafe-inline' 'unsafe-eval'",  // Next.js needs unsafe-inline/eval for dev
      "style-src 'self' 'unsafe-inline'",  // Tailwind/emotion need unsafe-inline
      "img-src 'self' data: http://localhost:8000 https://*.run.app",
      "font-src 'self' data:",
      "connect-src 'self' http://localhost:8000 https://*.run.app ws://localhost:8000 wss://*.run.app",
      "frame-ancestors 'none'",
      "form-action 'self'",
      "base-uri 'self'",
    ].join("; ");
    
    return [
      {
        source: '/(.*)',
        headers: [
          { key: 'X-Frame-Options', value: 'DENY' },
          { key: 'X-Content-Type-Options', value: 'nosniff' },
          { key: 'Referrer-Policy', value: 'strict-origin-when-cross-origin' },
          { key: 'Permissions-Policy', value: 'camera=(), microphone=(), geolocation=()' },
          { key: 'Content-Security-Policy', value: csp },
          { key: 'X-DNS-Prefetch-Control', value: 'off' },
        ],
      },
    ];
  },

  reactStrictMode: true,

  // Strip console.* in production builds
  compiler: {
    removeConsole: process.env.NODE_ENV === 'production',
  },

  // Workaround for Next.js 16 type generation bug in validator.ts
  typescript: {
    ignoreBuildErrors: true,
  },
};

export default nextConfig;
