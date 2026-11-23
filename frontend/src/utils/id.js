let counter = 0;

export const createId = () =>
  globalThis.crypto?.randomUUID?.() ??
  `${Date.now().toString(36)}-${(counter++).toString(36)}-${Math.random().toString(16).slice(2)}`;