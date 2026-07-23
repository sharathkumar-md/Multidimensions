/** @type {import('next').NextConfig} */
const nextConfig = {
  output: 'standalone',
  // Allow images from the FastAPI backend and local RAG server
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
    // Also allow unoptimized local file paths from the RAG pipeline
    unoptimized: true,
  },

  // Rewrites: proxy /api/* to FastAPI backend during development
  async rewrites() {
    return [
      {
        source: '/api/:path*',
        destination: `${process.env.NEXT_PUBLIC_API_URL ?? 'http://localhost:8000'}/api/:path*`,
      },
    ];
  },

  // Security headers
  async headers() {
    return [
      {
        source: '/(.*)',
        headers: [
          { key: 'X-Frame-Options',           value: 'DENY' },
          { key: 'X-Content-Type-Options',     value: 'nosniff' },
          { key: 'Referrer-Policy',            value: 'strict-origin-when-cross-origin' },
          { key: 'Permissions-Policy',         value: 'camera=(), microphone=(), geolocation=()' },
        ],
      },
    ];
  },

  // Strict mode for React
  reactStrictMode: true,

  // Suppress noisy hydration warnings from theme anti-flash script
  compiler: {
    removeConsole: process.env.NODE_ENV === 'production',
  },
};

module.exports = nextConfig;
