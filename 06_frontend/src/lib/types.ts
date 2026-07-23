// Shared TypeScript types across the frontend

export interface User {
  email: string;
  name: string;
  roles: string[];
  isAdmin: boolean;
  tokenExpiresAt: number;
}

export interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  createdAt: string;
  sources?: Source[];
  productImages?: ProductImage[];
  route?: 'LOCAL' | 'WEB' | 'NONE';
  isStreaming?: boolean;
}

export interface Source {
  sourceDoc: string;
  pageNum: number;
  snippet: string;
}

export interface ProductImage {
  imagePath: string;
  title: string;
  sourceDoc: string;
}

export interface ProductImageResult {
  images: ProductImage[];
  fromIndex: boolean;
}

export interface Session {
  id: string;
  title: string;
  createdAt: string;
  updatedAt: string;
  messageCount: number;
  userId: string;
}

export interface IndexStats {
  nChunks: number;
  nDocs: number;
  lastUpdated: string | null;
  gpuAvailable: boolean;
}

export interface IngestionStatus {
  running: boolean;
  progress: number;
  currentFile: string | null;
  error: string | null;
}

export interface ChatRequest {
  sessionId: string;
  question: string;
}

export interface StreamToken {
  token?: string;
  done?: boolean;
  sources?: Source[];
  productImages?: ProductImage[];
  route?: 'LOCAL' | 'WEB' | 'NONE';
  error?: string;
}

export interface ApiError {
  code: string;
  message: string;
  details?: Record<string, unknown>;
}

export type RouteType = 'LOCAL' | 'WEB' | 'NONE';
