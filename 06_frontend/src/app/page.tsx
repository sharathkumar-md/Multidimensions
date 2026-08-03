import { redirect } from 'next/navigation';

// Root → redirect to /login
export default function Home() {
  if (process.env.AUTH_ENABLED === 'false') {
    redirect('/chat');
  }
  redirect('/login');
}
