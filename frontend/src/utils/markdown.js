export function linkCitations(text, validIndexes) {
  return String(text ?? "")
    .split(/(```[\s\S]*?(?:```|$)|`[^`\n]*`)/g) // vị trí lẻ là các đoạn code (kể cả khối code chưa đóng khi đang stream)
    .map((part, i) =>
      i % 2 === 1
        ? part
        : part.replace(/\[(\d+)\](?!\()/g, (m, n) =>
            validIndexes.has(Number(n)) ? `[\\[${n}\\]](#cite-${n})` : m
          )
    )
    .join("");
}