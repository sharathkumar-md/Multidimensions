import type { Metadata } from 'next';
import { Sidebar } from '@/components/layout/Sidebar';
import styles from '../chat/layout.module.css';

export const metadata: Metadata = { title: 'Admin Hub' };

export default function AdminLayout({ children }: { children: React.ReactNode }) {
  return (
    <div className={styles.shell}>
      <Sidebar />
      <div className={styles.main}>{children}</div>
    </div>
  );
}
