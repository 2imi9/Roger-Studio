import type { ReactNode } from "react";

/**
 * Minimal markdown renderer for chat messages. Chat panes previously rendered
 * LLM output as `whitespace-pre-wrap` plaintext, so `**bold**`, `` `code` ``,
 * lists, and links all showed as literal syntax. A full markdown dep would be
 * overkill for what chat output actually uses — bold / italic / inline code /
 * simple lists / links — so this renderer handles those inline and keeps
 * whitespace-pre-wrap behavior for everything else. HTML escaping is handled
 * via React (we only return ReactNode, never dangerouslySetInnerHTML), so
 * there's no XSS surface from model output.
 *
 * Supported:
 *  - **bold** / __bold__
 *  - *italic* / _italic_  (heuristic: lone * without a bold partner)
 *  - `inline code`
 *  - [link text](https://url) — only http(s) schemes accepted
 *  - Unordered lists (-, *) and ordered lists (1. 2. …) when a line starts
 *    with the marker. Detected per-paragraph.
 *  - Headings (# / ## / ###) at line start. Only 1–3 levels; chat doesn't
 *    need deeper.
 *
 * Deliberately not supported: tables, blockquotes, code fences (multi-line),
 * raw HTML, image syntax. If we need any of those, swap in react-markdown.
 */

// Inline parser — splits text into React children, handling bold/italic/code/link.
// Order matters: code first (so ** inside `code` stays literal), then link, then bold, then italic.
function renderInline(text: string, keyPrefix: string): ReactNode[] {
  const out: ReactNode[] = [];
  let i = 0;
  let buf = "";
  let keyIdx = 0;
  const flush = () => {
    if (buf) {
      out.push(buf);
      buf = "";
    }
  };
  const pushNode = (node: ReactNode) => {
    flush();
    out.push(node);
  };

  while (i < text.length) {
    const c = text[i];
    // Inline code `...`
    if (c === "`") {
      const end = text.indexOf("`", i + 1);
      if (end > i) {
        pushNode(
          <code
            key={`${keyPrefix}-c-${keyIdx++}`}
            className="px-1 py-0.5 rounded bg-geo-elevated text-geo-accent font-mono text-[0.92em]"
          >
            {text.slice(i + 1, end)}
          </code>,
        );
        i = end + 1;
        continue;
      }
    }
    // Link [text](https://url) — only http(s) to avoid javascript: schemes.
    if (c === "[") {
      const close = text.indexOf("]", i + 1);
      if (close > i && text[close + 1] === "(") {
        const urlEnd = text.indexOf(")", close + 2);
        if (urlEnd > close) {
          const label = text.slice(i + 1, close);
          const href = text.slice(close + 2, urlEnd);
          if (/^https?:\/\//i.test(href)) {
            pushNode(
              <a
                key={`${keyPrefix}-l-${keyIdx++}`}
                href={href}
                target="_blank"
                rel="noopener noreferrer"
                className="text-geo-accent underline hover:no-underline"
              >
                {label}
              </a>,
            );
            i = urlEnd + 1;
            continue;
          }
        }
      }
    }
    // Bold **...** or __...__
    if ((c === "*" || c === "_") && text[i + 1] === c) {
      const marker = c + c;
      const end = text.indexOf(marker, i + 2);
      if (end > i + 1) {
        pushNode(
          <strong key={`${keyPrefix}-b-${keyIdx++}`} className="font-semibold">
            {renderInline(text.slice(i + 2, end), `${keyPrefix}-b${keyIdx}`)}
          </strong>,
        );
        i = end + 2;
        continue;
      }
    }
    // Italic *...* or _..._ — require a non-alphanumeric neighbor to avoid
    // eating the _ inside snake_case identifiers or the * in math-looking
    // expressions the LLM sometimes emits.
    if (c === "*" || c === "_") {
      const prev = text[i - 1];
      const isWordBoundary = !prev || /[^A-Za-z0-9]/.test(prev);
      if (isWordBoundary) {
        // Find matching close
        const end = text.indexOf(c, i + 1);
        if (end > i) {
          const next = text[end + 1];
          const closeIsBoundary = !next || /[^A-Za-z0-9]/.test(next);
          const inner = text.slice(i + 1, end);
          if (closeIsBoundary && inner.length > 0 && !inner.includes("\n")) {
            pushNode(
              <em key={`${keyPrefix}-i-${keyIdx++}`} className="italic">
                {renderInline(inner, `${keyPrefix}-i${keyIdx}`)}
              </em>,
            );
            i = end + 1;
            continue;
          }
        }
      }
    }
    buf += c;
    i++;
  }
  flush();
  return out;
}

// Block-level parsing: split into paragraphs and detect headings / lists.
function renderBlocks(src: string): ReactNode[] {
  const lines = src.split("\n");
  const blocks: ReactNode[] = [];
  let i = 0;
  let keyIdx = 0;

  while (i < lines.length) {
    const line = lines[i];
    const trimmed = line.trimStart();

    // Heading (# / ## / ###)
    const hMatch = /^(#{1,3})\s+(.*)$/.exec(trimmed);
    if (hMatch) {
      const level = hMatch[1].length;
      const text = hMatch[2];
      const sizeClass = level === 1 ? "text-base" : level === 2 ? "text-sm" : "text-[13px]";
      blocks.push(
        <div
          key={`h-${keyIdx++}`}
          className={`${sizeClass} font-semibold text-geo-text mt-2 mb-1`}
        >
          {renderInline(text, `h${keyIdx}`)}
        </div>,
      );
      i++;
      continue;
    }

    // Unordered list (-, *). Collect consecutive list lines.
    if (/^(\s*)[-*]\s+/.test(line)) {
      const items: ReactNode[] = [];
      let itemIdx = 0;
      while (i < lines.length && /^(\s*)[-*]\s+/.test(lines[i])) {
        const m = /^(\s*)[-*]\s+(.*)$/.exec(lines[i])!;
        items.push(
          <li key={`ul-i-${keyIdx}-${itemIdx++}`} className="ml-4 list-disc">
            {renderInline(m[2], `uli${keyIdx}-${itemIdx}`)}
          </li>,
        );
        i++;
      }
      blocks.push(
        <ul key={`ul-${keyIdx++}`} className="my-1 space-y-0.5">
          {items}
        </ul>,
      );
      continue;
    }

    // Ordered list (1. 2. ...). Collect consecutive numbered lines.
    if (/^(\s*)\d+\.\s+/.test(line)) {
      const items: ReactNode[] = [];
      let itemIdx = 0;
      while (i < lines.length && /^(\s*)\d+\.\s+/.test(lines[i])) {
        const m = /^(\s*)\d+\.\s+(.*)$/.exec(lines[i])!;
        items.push(
          <li key={`ol-i-${keyIdx}-${itemIdx++}`} className="ml-4 list-decimal">
            {renderInline(m[2], `oli${keyIdx}-${itemIdx}`)}
          </li>,
        );
        i++;
      }
      blocks.push(
        <ol key={`ol-${keyIdx++}`} className="my-1 space-y-0.5">
          {items}
        </ol>,
      );
      continue;
    }

    // Blank line → separator
    if (trimmed === "") {
      i++;
      continue;
    }

    // Plain paragraph — absorb until blank line / block boundary.
    const paraStart = i;
    while (
      i < lines.length &&
      lines[i].trim() !== "" &&
      !/^(#{1,3})\s+/.test(lines[i].trimStart()) &&
      !/^(\s*)[-*]\s+/.test(lines[i]) &&
      !/^(\s*)\d+\.\s+/.test(lines[i])
    ) {
      i++;
    }
    const text = lines.slice(paraStart, i).join("\n");
    blocks.push(
      <p key={`p-${keyIdx++}`} className="whitespace-pre-wrap leading-relaxed">
        {renderInline(text, `p${keyIdx}`)}
      </p>,
    );
  }

  return blocks;
}

export function MarkdownText({ children }: { children: string }) {
  return <div className="space-y-1">{renderBlocks(children)}</div>;
}
