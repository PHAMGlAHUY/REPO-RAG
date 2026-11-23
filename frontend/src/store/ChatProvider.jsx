import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState } from 'react';
import { streamReply } from '../api/chatApi';
import { createId } from '../utils/id';
import { loadState, saveState } from '../utils/storage';

const ChatContext = createContext(null);

const newConversation = () => ({
  id: createId(),
  title: 'Cuộc trò chuyện mới',
  messages: [],
  updatedAt: Date.now(),
});

function initState() {
  const saved = loadState();
  if (saved?.conversations?.length) {
    // Dọn tin nhắn bot rỗng còn sót do tải lại trang giữa lúc đang stream
    const conversations = saved.conversations.map((c) => ({
      ...c,
      messages: (c.messages ?? []).filter((m) => !(m.role === 'assistant' && !m.content && !m.error)),
    }));
    const activeId = conversations.some((c) => c.id === saved.activeId)
      ? saved.activeId
      : conversations[0].id;
    return { conversations, activeId };
  }
  const first = newConversation();
  return { conversations: [first], activeId: first.id };
}

export function ChatProvider({ children }) {
  const [state, setState] = useState(initState);
  const [isStreaming, setStreaming] = useState(false);
  const abortRef = useRef(null);

  // Không lưu mỗi token khi đang stream: lưu khi xong
  useEffect(() => {
    if (!isStreaming) saveState(state);
  }, [state, isStreaming]);

  const active = state.conversations.find((c) => c.id === state.activeId) ?? state.conversations[0];

  const patch = useCallback((id, fn) => {
    setState((s) => ({ ...s, conversations: s.conversations.map((c) => (c.id === id ? fn(c) : c)) }));
  }, []);

  const send = useCallback(
    async (text) => {
      const content = text.trim();
      if (!content || isStreaming) return;

      const convId = active.id;
      const userMsg = { id: createId(), role: 'user', content };
      const botMsg = { id: createId(), role: 'assistant', content: '' };
      // Chỉ gửi các tin hợp lệ làm lịch sử (bỏ tin lỗi/rỗng)
      const history = [...active.messages.filter((m) => m.content && !m.error), userMsg];

      patch(convId, (c) => ({
        ...c,
        title: c.messages.length ? c.title : content.slice(0, 40),
        messages: [...c.messages, userMsg, botMsg],
        updatedAt: Date.now(),
      }));

      const controller = new AbortController();
      abortRef.current = controller;
      setStreaming(true);

      const setBot = (fn) =>
        patch(convId, (c) => ({ ...c, messages: c.messages.map((m) => (m.id === botMsg.id ? fn(m) : m)) }));

      try {
        await streamReply({
          messages: history,
          signal: controller.signal,
          onToken: (tok) => setBot((m) => ({ ...m, content: m.content + tok })),
        });
      } catch (err) {
        if (err.name === 'AbortError') {
          // Người dùng bấm Dừng: bỏ tin bot nếu chưa có nội dung
          patch(convId, (c) => ({
            ...c,
            messages: c.messages.filter((m) => m.id !== botMsg.id || m.content),
          }));
        } else {
          setBot((m) => ({
            ...m,
            error: true,
            content: m.content || err?.message || 'Không nhận được phản hồi. Kiểm tra kết nối rồi gửi lại.',
          }));
        }
      } finally {
        if (abortRef.current === controller) {
          abortRef.current = null;
          setStreaming(false);
        }
      }
    },
    [active, isStreaming, patch]
  );

  const stop = useCallback(() => abortRef.current?.abort(), []);

  const newChat = useCallback(() => {
    if (isStreaming) return;
    if (active.messages.length === 0) return;
    const c = newConversation();
    setState((s) => ({ conversations: [c, ...s.conversations], activeId: c.id }));
  }, [active, isStreaming]);

  const select = useCallback((id) => {
    if (isStreaming) return;
    setState((s) => ({ ...s, activeId: id }));
  }, [isStreaming]);

  const remove = useCallback(
    (id) => {
      if (isStreaming) return;
      setState((s) => {
        const rest = s.conversations.filter((c) => c.id !== id);
        if (rest.length === 0) {
          const c = newConversation();
          return { conversations: [c], activeId: c.id };
        }
        return { conversations: rest, activeId: s.activeId === id ? rest[0].id : s.activeId };
      });
    },
    [isStreaming]
  );

  const value = useMemo(
    () => ({ conversations: state.conversations, active, isStreaming, send, stop, newChat, select, remove }),
    [state.conversations, active, isStreaming, send, stop, newChat, select, remove]
  );

  return <ChatContext.Provider value={value}>{children}</ChatContext.Provider>;
}

export function useChat() {
  const ctx = useContext(ChatContext);
  if (!ctx) throw new Error('useChat phải được dùng bên trong <ChatProvider>');
  return ctx;
}