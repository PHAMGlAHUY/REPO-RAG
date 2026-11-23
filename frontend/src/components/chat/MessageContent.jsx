import { useState } from 'react';

function CodeBlock({ lang, code }) {
  const [copied, setCopied] = useState(false);

  const copy = async () => {
    try {
      await navigator.clipboard.writeText(code);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      /* trình duyệt chặn clipboard */
    }
  };

  return (
    <div className="code">
      <div className="code__bar">
        <span>{lang || 'code'}</span>
        <button onClick={copy}>{copied ? 'Đã sao chép' : 'Sao chép'}</button>
      </div>
      <pre><code>{code}</code></pre>
    </div>
  );
}

/** Hiển thị văn bản + khối code ```lang ... ```; đoạn văn tách theo dòng trống. */
export default function MessageContent({ text }) {
  // split với 2 nhóm bắt: [văn bản, lang, code, văn bản, lang, code, ...]
  const parts = text.split(/```(\w*)\n?([\s\S]*?)(?:```|$)/g);
  const nodes = [];

  for (let i = 0; i < parts.length; i += 3) {
    parts[i]
      .split(/\n{2,}/)
      .filter((p) => p.trim())
      .forEach((p, j) => nodes.push(<p key={`t${i}-${j}`}>{p}</p>));
    if (i + 2 < parts.length) nodes.push(<CodeBlock key={`c${i}`} lang={parts[i + 1]} code={parts[i + 2].replace(/\n$/, '')} />);
  }

  return <div className="content">{nodes}</div>;
}
