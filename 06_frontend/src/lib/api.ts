import type { Session, Message, IndexStats, IngestionStatus } from './types';
import logger from './logger';

const isBrowser = typeof window !== 'undefined';
// Use relative path in browser to route through Next.js proxy (avoids CORS preflight)
const API_BASE = isBrowser ? '' : (process.env.NEXT_PUBLIC_API_URL ?? 'http://localhost:8000');

class ApiError extends Error {
  constructor(public status: number, public code: string, message: string) {
    super(message);
    this.name = 'ApiError';
  }
}

type ApiSource = {
  source_doc?: string;
  page_num?: number;
  snippet?: string;
  sourceDoc?: string;
  pageNum?: number;
};

type ApiProductImage = {
  image_path?: string;
  title?: string;
  source_doc?: string;
  imagePath?: string;
  sourceDoc?: string;
};

type ApiMessage = {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  created_at?: string;
  createdAt?: string;
  sources?: ApiSource[];
  product_images?: ApiProductImage[];
  productImages?: ApiProductImage[];
  route?: 'LOCAL' | 'WEB' | 'NONE';
};

type ApiSession = {
  id: string;
  title: string;
  user_id?: string;
  userId?: string;
  created_at?: string;
  createdAt?: string;
  updated_at?: string;
  updatedAt?: string;
  message_count?: number;
  messageCount?: number;
};

type ApiIndexStats = {
  n_chunks: number;
  n_docs: number;
  last_updated?: string | null;
  gpu_available: boolean;
};

type ApiIngestionStatus = {
  running: boolean;
  progress?: number;
  current_file?: string | null;
  error?: string | null;
};

function normalizeSource(source: ApiSource) {
  return {
    sourceDoc: source.sourceDoc ?? source.source_doc ?? '',
    pageNum: source.pageNum ?? source.page_num ?? 0,
    snippet: source.snippet ?? '',
  };
}

function normalizeProductImage(image: ApiProductImage) {
  return {
    imagePath: image.imagePath ?? image.image_path ?? '',
    title: image.title ?? '',
    sourceDoc: image.sourceDoc ?? image.source_doc ?? '',
  };
}

function normalizeMessage(message: ApiMessage): Message {
  return {
    id: message.id,
    role: message.role,
    content: message.content,
    createdAt: message.createdAt ?? message.created_at ?? new Date().toISOString(),
    sources: (message.sources ?? []).map(normalizeSource),
    productImages: (message.productImages ?? message.product_images ?? []).map(normalizeProductImage),
    route: message.route,
  };
}

function normalizeSession(session: ApiSession): Session {
  return {
    id: session.id,
    title: session.title,
    createdAt: session.createdAt ?? session.created_at ?? new Date().toISOString(),
    updatedAt: session.updatedAt ?? session.updated_at ?? new Date().toISOString(),
    messageCount: session.messageCount ?? session.message_count ?? 0,
    userId: session.userId ?? session.user_id ?? '',
  };
}

function normalizeIndexStats(stats: ApiIndexStats): IndexStats {
  return {
    nChunks: stats.n_chunks,
    nDocs: stats.n_docs,
    lastUpdated: stats.last_updated ?? null,
    gpuAvailable: stats.gpu_available,
  };
}

function normalizeIngestionStatus(status: ApiIngestionStatus): IngestionStatus {
  return {
    running: status.running,
    progress: status.progress ?? 0,
    currentFile: status.current_file ?? null,
    error: status.error ?? null,
  };
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const url = `${API_BASE}${path}`;
  logger.debug(`API request: ${init?.method ?? 'GET'} ${url}`);

  const res = await fetch(url, {
    ...init,
    headers: {
      'Content-Type': 'application/json',
      ...init?.headers,
    },
    credentials: 'include',
  });

  if (!res.ok) {
    let errBody: { code?: string; message?: string } = {};
    try { errBody = await res.json(); } catch { /* plain error */ }
    const message = errBody.message ?? `HTTP ${res.status}`;
    if (res.status === 404) {
      logger.warn(`API error: ${res.status} ${url}`, { code: errBody.code, message });
    } else {
      logger.error(`API error: ${res.status} ${url}`, { code: errBody.code, message });
    }
    throw new ApiError(res.status, errBody.code ?? 'UNKNOWN', message);
  }

  return res.json() as Promise<T>;
}

// ── Sessions ──────────────────────────────────────────────────────────────

export async function getSessions(): Promise<Session[]> {
  const sessions = await request<ApiSession[]>('/api/sessions');
  return sessions.map(normalizeSession);
}

export async function createSession(title?: string): Promise<Session> {
  const session = await request<ApiSession>('/api/sessions', {
    method: 'POST',
    body: JSON.stringify({ title: title ?? 'New conversation' }),
  });
  return normalizeSession(session);
}

export async function deleteSession(sessionId: string): Promise<void> {
  await request<void>(`/api/sessions/${sessionId}`, { method: 'DELETE' });
}

export async function getMessages(sessionId: string): Promise<Message[]> {
  const messages = await request<ApiMessage[]>(`/api/sessions/${sessionId}/messages`);
  return messages.map(normalizeMessage);
}

// ── Streaming chat ─────────────────────────────────────────────────────────

export async function* streamChat(
  sessionId: string,
  question: string,
  signal?: AbortSignal,
  webSearch: boolean = false,  // Fix 001: web search toggle wired end-to-end
): AsyncGenerator<string> {
  const url = `${API_BASE}/api/chat`;
  logger.info('Starting chat stream', { sessionId, question: question.slice(0, 80), webSearch });

  const res = await fetch(url, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ session_id: sessionId, question, web_search: webSearch }),
    credentials: 'include',
    signal,
  });

  if (!res.ok || !res.body) {
    logger.error(`Stream failed: HTTP ${res.status}`);
    throw new ApiError(res.status, 'STREAM_ERROR', `Stream failed: HTTP ${res.status}`);
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop() ?? '';

      for (const line of lines) {
        if (line.startsWith('data: ')) {
          const raw = line.slice(6).trim();
          if (raw) yield raw;
        }
      }
    }
  } finally {
    // Always release the reader lock even when the AbortSignal fires mid-stream.
    // Without this, the ReadableStream lock is held indefinitely, exhausting the
    // browser connection pool across repeated stop/start cycles (N-03).
    reader.cancel().catch(() => {});
  }
}

// ── Admin ─────────────────────────────────────────────────────────────────

export async function getIndexStats(): Promise<IndexStats> {
  const stats = await request<ApiIndexStats>('/api/index/stats');
  return normalizeIndexStats(stats);
}

export async function getIngestionStatus(): Promise<IngestionStatus> {
  const status = await request<ApiIngestionStatus>('/api/admin/ingest/status');
  return normalizeIngestionStatus(status);
}

export async function uploadPdf(file: File): Promise<{ filename: string }> {
  const form = new FormData();
  form.append('file', file);
  const url = `${API_BASE}/api/admin/upload`;
  logger.info('Uploading PDF', { filename: file.name, size: file.size });

  const res = await fetch(url, {
    method: 'POST',
    body: form,
    credentials: 'include',
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new ApiError(res.status, err.code ?? 'UPLOAD_ERROR', err.message ?? 'Upload failed');
  }
  return res.json();
}

export async function refreshIndex(): Promise<void> {
  await request<void>('/api/admin/refresh', { method: 'POST' });
}

export { ApiError };
