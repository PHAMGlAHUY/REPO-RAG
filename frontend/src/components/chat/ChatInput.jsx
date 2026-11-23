import { useEffect, useRef, useState } from 'react';

export default function ChatInput({ onSend, onStop, isStreaming }) {
  const [value, setValue] = useState('');
  const ref = useRef(null);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    el.style.height = 'auto';
    el.style.height = `${Math.min(el.scrollHeight, 200)}px`;
  }, [value]);

  const submit = () => {
    if (!value.trim() || isStreaming) return;
    onSend(value);
    setValue('');
  };

  const onKeyDown = (e) => {
    // Enter gửi, Shift+Enter xuống dòng (bỏ qua khi đang gõ tiếng Việt bằng IME)
    if (e.key === 'Enter' && !e.shiftKey && !e.nativeEvent.isComposing) {
      e.preventDefault();
      submit();
    }
  };

  return (
    <div className="composer">
      <div className="composer__box">
        <textarea
          ref={ref}
          rows={1}
          value={value}
          onChange={(e) => setValue(e.target.value)}
          onKeyDown={onKeyDown}
          placeholder="Nhập câu hỏi…"
          aria-label="Tin nhắn"
        />
        {isStreaming ? (
          <button className="send send--stop" onClick={onStop} aria-label="Dừng trả lời">
            <svg width="14" height="14" viewBox="0 0 14 14"><rect x="2" y="2" width="10" height="10" rx="2" fill="currentColor" /></svg>
          </button>
        ) : (
          <button className="send" onClick={submit} disabled={!value.trim()} aria-label="Gửi">
            <svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.8">
              <path d="M8 13V3M3.5 7.5L8 3l4.5 4.5" strokeLinecap="round" strokeLinejoin="round" />
            </svg>
          </button>
        )}
      </div>
      <p className="composer__note">Enter để gửi, Shift + Enter để xuống dòng.</p>
    </div>
  );
}
