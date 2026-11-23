import { useChat } from "../hooks/useChat";
import { useIndexStatus } from "../hooks/useIndexStatus";
import ChatInput from "./ChatInput";
import MessageList from "./MessageList";
import StatusBadge from "./StatusBadge";

export default function ChatWindow() {
  const { messages, loading, send, retry, stop, reset } = useChat();
  const idx = useIndexStatus();

  return (
    <div className="chat-window">
      <header className="chat-header">
        <h1>my_rag</h1>
        <div className="header-right">
          <StatusBadge paused={loading} idx={idx} />
          <button onClick={reset} disabled={messages.length === 0}>
            Cuộc trò chuyện mới
          </button>
        </div>
      </header>
      <MessageList messages={messages} loading={loading} onRetry={retry} />
      <ChatInput onSend={send} loading={loading} onStop={stop} />
    </div>
  );
}