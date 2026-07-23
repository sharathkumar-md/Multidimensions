import { redirect } from 'next/navigation';

// Root → redirect to /login
export default function Home() {
  redirect('/login');
}
