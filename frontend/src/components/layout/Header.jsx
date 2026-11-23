import { useChat } from '../../store/ChatProvider';

export default function Header({ onMenu }) {
  const { active } = useChat();
  return (
    <header className="header">
      <button
        type="button"
        className="icon-btn header__menu"
        onClick={onMenu}
        aria-label="Mở danh sách trò chuyện"
      >
        <svg width="20" height="20" viewBox="0 0 20 20" fill="none" stroke="currentColor" strokeWidth="1.6">
          <path d="M3 5h14M3 10h14M3 15h14" strokeLinecap="round" />
        </svg>
      </button>
      <h1 className="header__title">{active?.title ?? 'Cuộc trò chuyện mới'}</h1>
    </header>
  );
}