/**
 * Client-side logger using structured output.
 * In production, replace console with a remote sink (e.g. Datadog, Sentry).
 */

type LogLevel = 'debug' | 'info' | 'warn' | 'error';

const isDev = process.env.NODE_ENV === 'development';

function log(level: LogLevel, message: string, meta?: Record<string, unknown>) {
  if (level === 'debug' && !isDev) return; // suppress debug in production

  const entry = {
    timestamp: new Date().toISOString(),
    level,
    message,
    ...(meta ?? {}),
  };

  switch (level) {
    case 'debug': console.debug('[DEBUG]', entry); break;
    case 'info':  console.info('[INFO]',  entry); break;
    case 'warn':  console.warn('[WARN]',  entry); break;
    case 'error': console.error('[ERROR]', entry); break;
  }
}

const logger = {
  debug: (msg: string, meta?: Record<string, unknown>) => log('debug', msg, meta),
  info:  (msg: string, meta?: Record<string, unknown>) => log('info',  msg, meta),
  warn:  (msg: string, meta?: Record<string, unknown>) => log('warn',  msg, meta),
  error: (msg: string, meta?: Record<string, unknown>) => log('error', msg, meta),
};

export default logger;
