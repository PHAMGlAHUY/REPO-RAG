import MessageItem from './MessageItem';

export default function MessageList({ messages, isStreaming }) {
  return (
    <ol className="messages" aria-live="polite">
      {messages.map((m, i) => (
        <MessageItem key={m.id} message={m} pending={isStreaming && i === messages.length - 1} />
      ))}
    </ol>
  );
}
