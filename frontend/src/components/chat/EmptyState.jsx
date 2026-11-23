const SUGGESTIONS = [
  'Giải thích sự khác nhau giữa GDP danh nghĩa và GDP thực',
  'Viết hàm Python đảo ngược một chuỗi, kèm ví dụ',
  'Tóm tắt định luật II Newton bằng một ví dụ đơn giản',
];

export default function EmptyState({ onPick }) {
  return (
    <div className="empty">
      <h2 className="empty__title">Bạn muốn hỏi gì hôm nay?</h2>
      <p className="empty__hint">Nhập câu hỏi bên dưới, hoặc chọn một gợi ý.</p>
      <div className="empty__list">
        {SUGGESTIONS.map((s) => (
          <button key={s} className="chip" onClick={() => onPick(s)}>
            {s}
          </button>
        ))}
      </div>
    </div>
  );
}
