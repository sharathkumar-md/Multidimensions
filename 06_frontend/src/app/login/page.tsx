import type { Metadata } from 'next';
import styles from './page.module.css';

export const metadata: Metadata = { title: 'Login' };
import { redirect } from 'next/navigation';

export default function LoginPage() {
  if (process.env.NEXT_PUBLIC_AUTH_ENABLED === 'false') {
    redirect('/chat');
  }
  return (
    <main className={styles.page}>
      {/* Animated gradient background */}
      <div className={styles.bg} aria-hidden="true">
        <div className={styles.blob1} />
        <div className={styles.blob2} />
        <div className={styles.blob3} />
      </div>

      <div className={styles.card}>
        {/* Logo mark */}
        <div className={styles.logoWrap}>
          <div className={styles.logoMark} aria-hidden="true">MD</div>
          <div>
            <h1 className={styles.title}>MultiDimensions</h1>
            <p className={styles.subtitle}>AI Product Catalog Assistant</p>
          </div>
        </div>

        <div className={styles.divider} />

        <div className={styles.body}>
          <p className={styles.desc}>
            Sign in with your corporate account to access the AI-powered
            product catalog Q&amp;A system.
          </p>

          <form
            action={async () => {
              'use server';
              const { signIn } = await import('@/auth');
              await signIn('keycloak', { redirectTo: '/chat' });
            }}
          >
            <button
              type="submit"
              className={styles.loginBtn}
              id="keycloak-login-btn"
            >
              <svg
                viewBox="0 0 64 64"
                width="20"
                height="20"
                aria-hidden="true"
                className={styles.keycloakIcon}
              >
                <circle cx="32" cy="32" r="30" fill="currentColor" opacity="0.15" />
                <path
                  d="M32 12 L48 22 L48 42 L32 52 L16 42 L16 22Z"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="3"
                  strokeLinejoin="round"
                />
                <circle cx="32" cy="32" r="6" fill="currentColor" />
              </svg>
              Continue with Keycloak SSO
            </button>
          </form>

          <p className={styles.helpText}>
            Contact your IT administrator if you don&apos;t have access.
          </p>
        </div>

        <div className={styles.footer}>
          <p>Secured by Keycloak OIDC · Enterprise SSO</p>
        </div>
      </div>
    </main>
  );
}
