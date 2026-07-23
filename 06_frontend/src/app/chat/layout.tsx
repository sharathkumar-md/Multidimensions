import type { Metadata } from 'next';
import { Sidebar } from '@/components/layout/Sidebar';
import styles from './layout.module.css';

export const metadata: Metadata = {
  title: 'Chat',
};

export default function ChatLayout({ children }: { children: React.ReactNode }) {
  return (
    <div className={styles.shell}>
      <Sidebar />
      <div className={styles.main}>{children}</div>
    </div>
  );
}
