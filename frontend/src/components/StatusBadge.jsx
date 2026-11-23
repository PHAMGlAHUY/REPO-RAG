import { useHealth } from "../hooks/useHealth";
import { indexLabel, useIndexStatus } from "../hooks/useIndexStatus";
import { triggerReindex } from "../api/chatApi";
import { useState } from "react";

const DOT = { ok: "ok", warn: "warn", error: "bad" };

export default function StatusBadge({ paused, idx }) {
  const health = useHealth(paused);
  const [err, setErr] = useState("");

  if (!health) return <div className="status">Đang kiểm tra…</div>;

  const items = health.offline
    ? [{ name: "Backend", ok: false, detail: "Không kết nối được backend" }]
    : [
        { name: "Qdrant", ok: health.qdrant.ok, detail: health.qdrant.detail },
        { name: "Ollama", ok: health.ollama.ok, detail: health.ollama.detail },
      ];

  const { text, tone } = indexLabel(idx);
  const disabled = !idx || idx.offline || !idx.models_ready || idx.indexing;

  const reindex = async () => {
    setErr("");
    try {
      await triggerReindex();
    } catch (e) {
      setErr(e.message);
    }
  };

  return (
    <div className="status">
      {items.map((s) => (
        <span key={s.name} className="status-item" title={s.ok ? "Hoạt động" : s.detail}>
          <span className={`dot ${s.ok ? "ok" : "bad"}`} />
          {s.name}
        </span>
      ))}
      <span className="status-item" title={err || text}>
        <span className={`dot ${DOT[tone]}`} />
        {text}
      </span>
      <button className="reindex" disabled={disabled} onClick={reindex}>
        Index lại
      </button>
    </div>
  );
}