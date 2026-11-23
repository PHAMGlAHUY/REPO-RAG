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
  if (saved?.conversations?.length) return saved;
  const first = newConversation();
  return { conversations: [first], activeId: first.id };
}

export function ChatProvider({ children }) {
  const [state, setState] = useState(initState);
  const [isStreaming, setStreaming] = useState(false);
  const abortRef = useRef(null);

  useEffect(() => saveState(state), [state]);

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
      const history = [...active.messages, userMsg];

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
        if (err.name !== 'AbortError') {
          setBot((m) => ({
            ...m,
            error: true,
            content: 'Không nhận được phản hồi. Kiểm tra kết nối rồi gửi lại.',
          }));
        }
      } finally {
        setStreaming(false);
        abortRef.current = null;
      }
    },
    [active, isStreaming, patch]
  );

  const stop = useCallback(() => abortRef.current?.abort(), []);

  const newChat = useCallback(() => {
    if (active.messages.length === 0) return;
    const c = newConversation();
    setState((s) => ({ conversations: [c, ...s.conversations], activeId: c.id }));
  }, [active]);

  const select = useCallback((id) => setState((s) => ({ ...s, activeId: id })), []);

  const remove = useCallback((id) => {
    setState((s) => {
      const rest = s.conversations.filter((c) => c.id !== id);
      if (rest.length === 0) {
        const c = newConversation();
        return { conversations: [c], activeId: c.id };
      }
      return { conversations: rest, activeId: s.activeId === id ? rest[0].id : s.activeId };
    });
  }, []);

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
