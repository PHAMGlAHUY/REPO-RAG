import { useState } from "react";
import { triggerReindex } from "../api/chatApi";

const DOT = { ok: "ok", warn: "warn", error: "bad" };

export default function IndexStatus() {
  const idx = useIndexStatus();
  const [err, setErr] = useState("");
  const { text, tone } = indexLabel(idx);

  const reindex = async () => {
    setErr("");
    try {
      await triggerReindex();
    } catch (e) {
      setErr(e.message);
    }
  };

  const disabled = !idx || idx.offline || !idx.models_ready || idx.indexing;

  return (
    <>
      <span className="status-item" title={err || text}>
        <span className={`dot ${DOT[tone]}`} />
        {text}
      </span>
      <button className="reindex" disabled={disabled} onClick={reindex}>
        Index lại
      </button>
    </>
  );
}