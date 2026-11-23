const KEY = 'chat-ui:v1';

export function loadState() {
  try {
    const raw = localStorage.getItem(KEY);
    if (!raw) return null;
    const data = JSON.parse(raw);
    return data && typeof data === 'object' && Array.isArray(data.conversations) ? data : null;
  } catch {
    return null;
  }
}

export function saveState(state) {
  try {
    localStorage.setItem(KEY, JSON.stringify(state));
  } catch {
    /* bỏ qua: hết dung lượng hoặc chế độ riêng tư */
  }
}