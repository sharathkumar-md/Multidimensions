import type { Message } from '@/lib/types';
import styles from './UserMessage.module.css';

interface UserMessageProps {
  message: Message;
}

export function UserMessage({ message }: UserMessageProps) {
  return (
    <article className={styles.wrap} aria-label="Your message">
      <div className={styles.bubble}>
        <p className={styles.text}>{message.content}</p>
      </div>
    </article>
  );
}
