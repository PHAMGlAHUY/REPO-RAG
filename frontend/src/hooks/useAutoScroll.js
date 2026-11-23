import { useEffect, useRef } from 'react';

/** Tự cuộn xuống cuối khi có tin nhắn mới, trừ khi người dùng đang cuộn lên đọc. */
export function useAutoScroll(dep) {
  const containerRef = useRef(null);
  const stickRef = useRef(true);

  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const onScroll = () => {
      stickRef.current = el.scrollHeight - el.scrollTop - el.clientHeight < 80;
    };
    el.addEventListener('scroll', onScroll, { passive: true });
    return () => el.removeEventListener('scroll', onScroll);
  }, []);

  useEffect(() => {
    const el = containerRef.current;
    if (el && stickRef.current) el.scrollTop = el.scrollHeight;
  }, [dep]);

  return containerRef;
}
