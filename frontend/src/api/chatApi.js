const API_URL = import.meta.env.VITE_API_URL;

const sleep = (ms, signal) =>
  new Promise((resolve, reject) => {
    const t = setTimeout(resolve, ms);
    signal?.addEventListener('abort', () => {
      clearTimeout(t);
      reject(new DOMException('Aborted', 'AbortError'));
    });
  });

/** Phản hồi giả lập, stream từng đoạn nhỏ như model thật. */
async function mockStream({ messages, signal, onToken }) {
  const last = messages.at(-1)?.content ?? '';
  const reply =
    `Bạn vừa hỏi: “${last}”.\n\nĐây là phản hồi giả lập để bạn kiểm tra giao diện. ` +
    'Hãy nối backend thật trong `src/api/chatApi.js`.\n\n' +
    '```js\nconst answer = await streamReply({ messages, onToken });\n```\n\n' +
    'Giao diện hỗ trợ stream, dừng giữa chừng, và lưu lịch sử trò chuyện.';

  await sleep(500, signal);
  for (let i = 0; i < reply.length; i += 3) {
    onToken(reply.slice(i, i + 3));
    await sleep(18, signal);
  }
}

/** Gọi backend thật: POST {messages} và đọc phản hồi dạng stream văn bản. */
async function remoteStream({ messages, signal, onToken }) {
  const res = await fetch(`${API_URL}/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ messages: messages.map(({ role, content }) => ({ role, content })) }),
    signal,
  });
  if (!res.ok || !res.body) throw new Error(`HTTP ${res.status}`);

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    onToken(decoder.decode(value, { stream: true }));
  }
}

export const streamReply = API_URL ? remoteStream : mockStream;
