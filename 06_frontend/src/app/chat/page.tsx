import { redirect } from 'next/navigation';

// /chat → redirect to the first session or show new chat prompt
export default function ChatIndexPage() {
  // Server component: can't access client store here.
  // The sidebar handles routing to sessions; redirect to a welcome URL.
  redirect('/chat/new');
}
