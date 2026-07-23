import styles from './ThinkingIndicator.module.css';

export function ThinkingIndicator() {
  return (
    <div className={styles.wrap} role="status" aria-label="Generating response">
      <div className={styles.avatar} aria-hidden="true">AI</div>
      <div className={styles.bubble}>
        <span className={styles.dot} style={{ animationDelay: '0ms' }} />
        <span className={styles.dot} style={{ animationDelay: '150ms' }} />
        <span className={styles.dot} style={{ animationDelay: '300ms' }} />
      </div>
    </div>
  );
}
