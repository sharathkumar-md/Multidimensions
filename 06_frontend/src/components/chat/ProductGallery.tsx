'use client';

import { useState } from 'react';
import Image from 'next/image';
import { X, ChevronLeft, ChevronRight, ZoomIn } from 'lucide-react';
import type { ProductImage } from '@/lib/types';
import styles from './ProductGallery.module.css';

/**
 * Reject URI schemes that can execute code when used as an image src.
 * data: URIs can carry malicious payloads; javascript: URIs execute scripts.
 */
function safeSrc(path: string): string {
  const trimmed = path.trim().toLowerCase();
  if (trimmed.startsWith('javascript:') || trimmed.startsWith('data:')) {
    return '';
  }
  return path;
}

interface ProductGalleryProps {
  images: ProductImage[];
}

export function ProductGallery({ images }: ProductGalleryProps) {
  const [lightboxIdx, setLightboxIdx] = useState<number | null>(null);
  const preview = images.slice(0, 4);
  const extra = images.length - preview.length;

  const closeLightbox = () => setLightboxIdx(null);
  const prev = () => setLightboxIdx((i) => (i !== null && i > 0 ? i - 1 : i));
  const next = () => setLightboxIdx((i) => (i !== null && i < images.length - 1 ? i + 1 : i));

  return (
    <>
      <div className={styles.grid}>
        {preview.map((img, i) => (
          <button
            key={i}
            className={styles.thumb}
            onClick={() => setLightboxIdx(i)}
            aria-label={`View product image: ${img.title}`}
            title={img.title}
          >
            <div className={styles.imgWrap}>
            <img
                src={safeSrc(img.imagePath)}
                alt={img.title}
                className={styles.img}
                loading="lazy"
              />
              <div className={styles.overlay}>
                <ZoomIn size={18} />
              </div>
            </div>
            <p className={styles.caption} title={`${img.title} · ${img.sourceDoc}`}>
              {img.title}
            </p>
          </button>
        ))}
        {extra > 0 && (
          <button
            className={[styles.thumb, styles.extraThumb].join(' ')}
            onClick={() => setLightboxIdx(4)}
            aria-label={`View ${extra} more images`}
          >
            <div className={styles.imgWrap}>
              <img src={safeSrc(images[4].imagePath)} alt="" className={[styles.img, styles.dimmed].join(' ')} loading="lazy" />
              <div className={styles.extraOverlay}>+{extra}</div>
            </div>
          </button>
        )}
      </div>

      {/* Lightbox */}
      {lightboxIdx !== null && (
        <div className={styles.lightbox} role="dialog" aria-modal="true" aria-label="Image viewer" onClick={closeLightbox}>
          <div className={styles.lightboxInner} onClick={(e) => e.stopPropagation()}>
            <button className={styles.lbClose} onClick={closeLightbox} aria-label="Close">
              <X size={20} />
            </button>
            {lightboxIdx > 0 && (
              <button className={[styles.lbNav, styles.lbPrev].join(' ')} onClick={prev} aria-label="Previous">
                <ChevronLeft size={22} />
              </button>
            )}
            <div className={styles.lbImgWrap}>
              <img
                src={safeSrc(images[lightboxIdx].imagePath)}
                alt={images[lightboxIdx].title}
                className={styles.lbImg}
              />
            </div>
            {lightboxIdx < images.length - 1 && (
              <button className={[styles.lbNav, styles.lbNext].join(' ')} onClick={next} aria-label="Next">
                <ChevronRight size={22} />
              </button>
            )}
            <div className={styles.lbCaption}>
              <p className={styles.lbTitle}>{images[lightboxIdx].title}</p>
              <p className={styles.lbSource}>{images[lightboxIdx].sourceDoc}</p>
              <p className={styles.lbCounter}>{lightboxIdx + 1} / {images.length}</p>
            </div>
          </div>
        </div>
      )}
    </>
  );
}
