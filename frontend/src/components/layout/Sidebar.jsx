import { useChat } from '../../store/ChatProvider';

export default function Sidebar({ open, onClose }) {
  const { conversations, active, newChat, select, remove, isStreaming } = useChat();

  const handleSelect = (id) => {
    select(id);
    onClose();
  };

  const handleRemove = (e, id) => {
    e.stopPropagation();
    remove(id);
  };

  return (
    <>
      <div className={`scrim ${open ? 'scrim--show' : ''}`} onClick={onClose} />
      <aside className={`sidebar ${open ? 'sidebar--open' : ''}`} aria-label="Lịch sử trò chuyện">
        <div className="sidebar__top">
          <span className="sidebar__brand">Trợ lý AI</span>
          <button
            type="button"
            className="btn-new"
            onClick={() => {
              newChat();
              onClose();
            }}
            disabled={isStreaming}
          >
            Cuộc trò chuyện mới
          </button>
        </div>

        <nav className="sidebar__list">
          {conversations.map((c) => (
            <div key={c.id} className={`conv ${c.id === active?.id ? 'conv--active' : ''}`}>
              <button
                type="button"
                className="conv__title"
                onClick={() => handleSelect(c.id)}
                disabled={isStreaming}
                title={c.title}
              >
                {c.title}
              </button>
              <button
                type="button"
                className="conv__delete"
                onClick={(e) => handleRemove(e, c.id)}
                disabled={isStreaming}
                aria-label={`Xóa “${c.title}”`}
              >
                <svg width="14" height="14" viewBox="0 0 14 14" fill="none" stroke="currentColor" strokeWidth="1.5">
                  <path d="M3 3l8 8M11 3l-8 8" strokeLinecap="round" />
                </svg>
              </button>
            </div>
          ))}
        </nav>
      </aside>
    </>
  );
}