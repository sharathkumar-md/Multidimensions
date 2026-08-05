'use client';

import { useEffect, useState, useCallback, useRef } from 'react';
import { useRouter } from 'next/navigation';
import {
  Upload, RefreshCw, FileText, Database,
  CheckCircle, AlertCircle, Loader,
} from 'lucide-react';
import { Header } from '@/components/layout/Header';
import { Button } from '@/components/ui/Button';
import { Badge } from '@/components/ui/Badge';
import { useAuthStore } from '@/lib/store';
import { getIndexStats, uploadPdf, refreshIndex, getIngestionStatus } from '@/lib/api';
import type { IndexStats, IngestionStatus } from '@/lib/types';
import logger from '@/lib/logger';
import styles from './page.module.css';

function StatCard({ label, value, sub }: { label: string; value: string | number; sub?: string }) {
  return (
    <div className={styles.statCard}>
      <p className={styles.statLabel}>{label}</p>
      <p className={styles.statValue}>{value}</p>
      {sub && <p className={styles.statSub}>{sub}</p>}
    </div>
  );
}

export default function AdminPage() {
  const router = useRouter();
  const { user } = useAuthStore();
  const [stats, setStats] = useState<IndexStats | null>(null);
  const [ingest, setIngest] = useState<IngestionStatus | null>(null);
  const [uploading, setUploading] = useState(false);
  const [refreshing, setRefreshing] = useState(false);
  const [uploadResult, setUploadResult] = useState<{ ok: boolean; msg: string } | null>(null);
  const [dragging, setDragging] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const pollRef = useRef<NodeJS.Timeout | null>(null);

  // Guard: redirect non-admins
  useEffect(() => {
    if (user && !user.isAdmin) {
      router.replace('/chat');
    }
  }, [user, router]);

  const loadStats = useCallback(async () => {
    try {
      const s = await getIndexStats();
      setStats(s);
    } catch (e: unknown) {
      logger.error('Failed to load index stats', { error: (e as Error).message });
    }
  }, []);

  const pollIngestion = useCallback(async () => {
    try {
      const s = await getIngestionStatus();
      setIngest(s);
      if (s.running) {
        pollRef.current = setTimeout(pollIngestion, 2000);
      } else {
        loadStats(); // refresh stats once done
      }
    } catch (err: unknown) {
      // Surface the failure — silently swallowing it hides backend outages
      logger.error('Ingestion status poll failed', { error: (err as Error).message });
      setIngest((prev) =>
        prev ? { ...prev, error: 'Status unavailable — backend may be unreachable.' } : null
      );
    }
  }, [loadStats]);

  useEffect(() => {
    // Guard: don't fetch until auth store is hydrated and user is confirmed admin.
    // Without this check, the API calls fire before AuthProvider.useEffect has run,
    // meaning a non-admin SPA navigation briefly dispatches admin API requests (N-04).
    if (!user?.isAdmin) return;
    loadStats();
    pollIngestion();
    return () => { if (pollRef.current) clearTimeout(pollRef.current); };
  }, [loadStats, pollIngestion, user]);

  const handleFiles = useCallback(async (files: FileList | null) => {
    if (!files || files.length === 0) return;
    const pdf = Array.from(files).find((f) => f.name.toLowerCase().endsWith('.pdf'));
    if (!pdf) {
      setUploadResult({ ok: false, msg: 'Only PDF files are accepted.' });
      return;
    }
    // Enforce a 50 MB client-side limit to prevent browser OOM and backend DoS
    const MAX_PDF_BYTES = 50 * 1024 * 1024;
    if (pdf.size > MAX_PDF_BYTES) {
      setUploadResult({ ok: false, msg: 'File exceeds the 50 MB limit. Please upload a smaller PDF.' });
      return;
    }
    // Validate MIME type when the browser can detect it (not all browsers populate this)
    if (pdf.type && pdf.type !== 'application/pdf') {
      setUploadResult({ ok: false, msg: 'Only PDF files are accepted.' });
      return;
    }
    setUploading(true);
    setUploadResult(null);
    try {
      const res = await uploadPdf(pdf);
      setUploadResult({ ok: true, msg: `✓ ${res.filename} uploaded. Ingestion started…` });
      logger.info('PDF uploaded', { filename: res.filename });
      // Start polling ingestion status
      pollIngestion();
    } catch (e: unknown) {
      setUploadResult({ ok: false, msg: (e as Error).message ?? 'Upload failed.' });
      logger.error('PDF upload failed', { error: (e as Error).message });
    } finally {
      setUploading(false);
    }
  }, [pollIngestion]);

  const handleRefresh = useCallback(async () => {
    setRefreshing(true);
    try {
      await refreshIndex();
      await loadStats();
      setUploadResult({ ok: true, msg: '✓ Index refreshed.' });
    } catch (e: unknown) {
      setUploadResult({ ok: false, msg: (e as Error).message ?? 'Refresh failed.' });
    } finally {
      setRefreshing(false);
    }
  }, [loadStats]);

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDragging(false);
    handleFiles(e.dataTransfer.files);
  };

  // Don't render (or trigger data-fetching effects) until auth store hydration is complete.
  // middleware.ts handles server-side redirects for direct URL navigation;
  // this handles client-side SPA navigation where middleware does not run.
  if (!user) return null;

  return (
    <div className={styles.page}>
      <Header sessionTitle="Admin Hub" />

      <div className={styles.content}>
        <div className={styles.inner}>

          {/* Index Stats */}
          <section className={styles.section}>
            <div className={styles.sectionHeader}>
              <h2 className={styles.sectionTitle}>
                <Database size={16} />
                Index Status
              </h2>
              <Button
                variant="ghost"
                size="sm"
                icon={<RefreshCw size={14} className={refreshing ? styles.spin : ''} />}
                onClick={loadStats}
                disabled={refreshing}
              >
                Refresh
              </Button>
            </div>
            <div className={styles.statsGrid}>
              <StatCard
                label="Total Chunks"
                value={stats?.nChunks ?? '—'}
                sub="Vector embeddings indexed"
              />
              <StatCard
                label="Documents"
                value={stats?.nDocs ?? '—'}
                sub="PDF catalogs in knowledge base"
              />
              <StatCard
                label="GPU"
                value={stats ? (stats.gpuAvailable ? 'Available' : 'CPU mode') : '—'}
                sub={stats?.gpuAvailable ? 'CUDA ready' : 'Running on CPU (slow)'}
              />
              <StatCard
                label="Last Updated"
                value={
                  stats?.lastUpdated
                    ? new Date(stats.lastUpdated).toLocaleString('en', { dateStyle: 'medium', timeStyle: 'short' })
                    : '—'
                }
                sub="Index rebuild timestamp"
              />
            </div>
          </section>

          {/* Ingestion status */}
          {ingest?.running && (
            <div className={styles.ingestBanner}>
              <Loader size={15} className={styles.spin} />
              <span>
                Ingesting{ingest.currentFile ? `: ${ingest.currentFile}` : '…'}
                {ingest.progress > 0 && ` (${Math.round(ingest.progress * 100)}%)`}
              </span>
            </div>
          )}

          {/* Upload */}
          <section className={styles.section}>
            <div className={styles.sectionHeader}>
              <h2 className={styles.sectionTitle}>
                <Upload size={16} />
                Upload PDF Catalog
              </h2>
            </div>

            <div
              className={[styles.dropzone, dragging ? styles.dragging : ''].join(' ')}
              onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
              onDragLeave={() => setDragging(false)}
              onDrop={handleDrop}
              onClick={() => fileInputRef.current?.click()}
              role="button"
              tabIndex={0}
              aria-label="Drop a PDF here or click to browse"
              onKeyDown={(e) => e.key === 'Enter' && fileInputRef.current?.click()}
            >
              <input
                ref={fileInputRef}
                type="file"
                accept=".pdf"
                className={styles.fileInput}
                onChange={(e) => handleFiles(e.target.files)}
                aria-hidden="true"
              />
              <FileText size={36} className={styles.dropIcon} />
              <p className={styles.dropTitle}>
                {dragging ? 'Drop PDF here' : 'Drag & drop a PDF catalog'}
              </p>
              <p className={styles.dropSub}>or click to browse — PDF files only</p>
              {uploading && (
                <div className={styles.uploadingIndicator}>
                  <Loader size={16} className={styles.spin} />
                  <span>Uploading…</span>
                </div>
              )}
            </div>

            {uploadResult && (
              <div className={[styles.uploadResult, uploadResult.ok ? styles.ok : styles.err].join(' ')}>
                {uploadResult.ok
                  ? <CheckCircle size={14} />
                  : <AlertCircle size={14} />}
                {uploadResult.msg}
              </div>
            )}
          </section>

          {/* Refresh index */}
          <section className={styles.section}>
            <div className={styles.sectionHeader}>
              <h2 className={styles.sectionTitle}>
                <RefreshCw size={16} />
                Force Index Refresh
              </h2>
            </div>
            <p className={styles.sectionDesc}>
              Reload the vector index from disk without restarting the server.
              Use this after a manual ingestion or if the index seems stale.
            </p>
            <Button
              variant="secondary"
              size="sm"
              icon={<RefreshCw size={14} className={refreshing ? styles.spin : ''} />}
              onClick={handleRefresh}
              loading={refreshing}
            >
              Refresh Index Now
            </Button>
          </section>
        </div>
      </div>
    </div>
  );
}
