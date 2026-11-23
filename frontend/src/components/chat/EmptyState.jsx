const SUGGESTIONS = [
  'Dự án này có những module chính nào?',
  'Hàm chunk_file hoạt động như thế nào?',
  'Quy trình index repository vào Qdrant gồm những bước nào?',
];

export default function EmptyState({ onPick }) {
  return (
    <div className="empty">
      <h2 className="empty__title">Bạn muốn hỏi gì về repository?</h2>
      <p className="empty__hint">Nhập câu hỏi bên dưới, hoặc chọn một gợi ý.</p>
      <div className="empty__list">
        {SUGGESTIONS.map((s) => (
          <button key={s} type="button" className="chip" onClick={() => onPick(s)}>
            {s}
          </button>
        ))}
      </div>
    </div>
  );
}