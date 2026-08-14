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
    return [
      {
        source: '/(.*)',
        headers: [
          { key: 'X-Frame-Options',       value: 'DENY' },
          { key: 'X-Content-Type-Options', value: 'nosniff' },
          { key: 'Referrer-Policy',        value: 'strict-origin-when-cross-origin' },
          { key: 'Permissions-Policy',     value: 'camera=(), microphone=(), geolocation=()' },
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
