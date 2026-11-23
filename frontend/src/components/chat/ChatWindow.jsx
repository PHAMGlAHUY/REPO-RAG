import { useChat } from '../../store/ChatProvider';
import { useAutoScroll } from '../../hooks/useAutoScroll';
import MessageList from './MessageList';
import EmptyState from './EmptyState';
import ChatInput from './ChatInput';

export default function ChatWindow() {
  const { active, isStreaming, send, stop } = useChat();
  const messages = active?.messages ?? [];
  const last = messages.at(-1);
  const scrollRef = useAutoScroll(
    `${active?.id ?? ''}:${messages.length}:${last?.content?.length ?? 0}`
  );

  return (
    <section className="chat">
      <div className="chat__scroll" ref={scrollRef}>
        {messages.length === 0 ? (
          <EmptyState onPick={send} />
        ) : (
          <MessageList messages={messages} isStreaming={isStreaming} />
        )}
      </div>
      <ChatInput onSend={send} onStop={stop} isStreaming={isStreaming} />
    </section>
  );
}