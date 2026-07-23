import type { Session, Message, IndexStats, IngestionStatus } from './types';
import logger from './logger';

const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? 'http://localhost:8000';

class ApiError extends Error {
  constructor(public status: number, public code: string, message: string) {
    super(message);
    this.name = 'ApiError';
  }
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
  return request<Session[]>('/api/sessions');
}

export async function createSession(title?: string): Promise<Session> {
  return request<Session>('/api/sessions', {
    method: 'POST',
    body: JSON.stringify({ title: title ?? 'New conversation' }),
  });
}

export async function deleteSession(sessionId: string): Promise<void> {
  await request<void>(`/api/sessions/${sessionId}`, { method: 'DELETE' });
}

export async function getMessages(sessionId: string): Promise<Message[]> {
  return request<Message[]>(`/api/sessions/${sessionId}/messages`);
}

// ── Streaming chat ─────────────────────────────────────────────────────────

export async function* streamChat(
  sessionId: string,
  question: string,
  signal?: AbortSignal,
): AsyncGenerator<string> {
  const url = `${API_BASE}/api/chat`;
  logger.info('Starting chat stream', { sessionId, question: question.slice(0, 80) });

  const res = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ sessionId, question }),
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
}

// ── Admin ─────────────────────────────────────────────────────────────────

export async function getIndexStats(): Promise<IndexStats> {
  return request<IndexStats>('/api/index/stats');
}

export async function getIngestionStatus(): Promise<IngestionStatus> {
  return request<IngestionStatus>('/api/admin/ingest/status');
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
