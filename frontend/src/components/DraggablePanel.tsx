/**
 * DraggablePanel — generic floating-window wrapper.
 *
 * Pattern mirrors Google Earth Studio's "Ask Google Earth" panel: the
 * content is detached from the sidebar and renders as a draggable +
 * resizable overlay above the map, so it can grow wider than the 480 px
 * sidebar allows. Used initially for the LLM chat pane; any other tab
 * that feels cramped can adopt it.
 *
 * Props are intentionally minimal:
 *   - ``title``      — shown in the drag-handle title bar
 *   - ``onClose``    — called when the user clicks the × (caller usually
 *     flips a ``floating`` state back to false so the content remounts in
 *     the sidebar).
 *   - ``storageKey`` — sessionStorage prefix so position + size survive a
 *     tab refresh. Pass a unique key per panel.
 *   - ``defaultX / defaultY / defaultWidth / defaultHeight`` — starting
 *     geometry used when there's nothing in sessionStorage yet.
 *   - ``minWidth / minHeight`` — refuse-to-shrink-below guards.
 *   - ``children`` — the actual panel content.
 *
 * The component tracks pointer capture for drag + bottom-right resize. It
 * clamps to the viewport (so the panel can't get dragged offscreen) and
 * keeps one drag/resize in-flight at a time.
 */
import { useCallback, useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";

interface DraggablePanelProps {
  title: string;
  onClose: () => void;
  storageKey: string;
  defaultX?: number;
  defaultY?: number;
  defaultWidth?: number;
  defaultHeight?: number;
  minWidth?: number;
  minHeight?: number;
  children: React.ReactNode;
}

interface Geometry {
  x: number;
  y: number;
  w: number;
  h: number;
}

const EDGE_PADDING = 8;

function readGeometry(key: string, fallback: Geometry): Geometry {
  try {
    const raw = sessionStorage.getItem(key);
    if (!raw) return fallback;
    const v = JSON.parse(raw);
    if (
      typeof v?.x === "number" && typeof v?.y === "number"
      && typeof v?.w === "number" && typeof v?.h === "number"
    ) return v;
  } catch { /* noop */ }
  return fallback;
}

function clampGeometry(g: Geometry, minW: number, minH: number): Geometry {
  const vw = typeof window !== "undefined" ? window.innerWidth : 1920;
  const vh = typeof window !== "undefined" ? window.innerHeight : 1080;
  const w = Math.max(minW, Math.min(g.w, vw - EDGE_PADDING * 2));
  const h = Math.max(minH, Math.min(g.h, vh - EDGE_PADDING * 2));
  const x = Math.max(EDGE_PADDING, Math.min(g.x, vw - w - EDGE_PADDING));
  const y = Math.max(EDGE_PADDING, Math.min(g.y, vh - h - EDGE_PADDING));
  return { x, y, w, h };
}

export function DraggablePanel({
  title,
  onClose,
  storageKey,
  defaultX = 120,
  defaultY = 120,
  defaultWidth = 560,
  defaultHeight = 720,
  minWidth = 320,
  minHeight = 400,
  children,
}: DraggablePanelProps) {
  const [geom, setGeom] = useState<Geometry>(() =>
    clampGeometry(
      readGeometry(storageKey, {
        x: defaultX, y: defaultY, w: defaultWidth, h: defaultHeight,
      }),
      minWidth, minHeight,
    ),
  );
  // Persist whenever geometry settles after a drag / resize.
  useEffect(() => {
    try { sessionStorage.setItem(storageKey, JSON.stringify(geom)); } catch { /* noop */ }
  }, [geom, storageKey]);

  // Re-clamp on window resize so the panel never ends up half-offscreen.
  useEffect(() => {
    const onResize = () => setGeom((g) => clampGeometry(g, minWidth, minHeight));
    window.addEventListener("resize", onResize);
    return () => window.removeEventListener("resize", onResize);
  }, [minWidth, minHeight]);

  // Drag — title bar pointer-down captures + listens for move/up on window.
  const dragRef = useRef<{ dx: number; dy: number } | null>(null);
  const onHeaderPointerDown = useCallback((e: React.PointerEvent) => {
    // Ignore if the pointer-down came from the close button.
    if ((e.target as HTMLElement).closest("[data-no-drag]")) return;
    e.preventDefault();
    dragRef.current = { dx: e.clientX - geom.x, dy: e.clientY - geom.y };
    (e.currentTarget as HTMLElement).setPointerCapture(e.pointerId);
  }, [geom.x, geom.y]);

  const onHeaderPointerMove = useCallback((e: React.PointerEvent) => {
    if (!dragRef.current) return;
    setGeom((g) => clampGeometry(
      { ...g, x: e.clientX - dragRef.current!.dx, y: e.clientY - dragRef.current!.dy },
      minWidth, minHeight,
    ));
  }, [minWidth, minHeight]);

  const onHeaderPointerUp = useCallback((e: React.PointerEvent) => {
    dragRef.current = null;
    try { (e.currentTarget as HTMLElement).releasePointerCapture(e.pointerId); }
    catch { /* noop */ }
  }, []);

  // Resize — bottom-right handle.
  const resizeRef = useRef<{ startX: number; startY: number; startW: number; startH: number } | null>(null);
  const onResizeDown = useCallback((e: React.PointerEvent) => {
    e.preventDefault();
    e.stopPropagation();
    resizeRef.current = {
      startX: e.clientX, startY: e.clientY,
      startW: geom.w, startH: geom.h,
    };
    (e.currentTarget as HTMLElement).setPointerCapture(e.pointerId);
  }, [geom.w, geom.h]);

  const onResizeMove = useCallback((e: React.PointerEvent) => {
    if (!resizeRef.current) return;
    const { startX, startY, startW, startH } = resizeRef.current;
    setGeom((g) => clampGeometry(
      { ...g, w: startW + (e.clientX - startX), h: startH + (e.clientY - startY) },
      minWidth, minHeight,
    ));
  }, [minWidth, minHeight]);

  const onResizeUp = useCallback((e: React.PointerEvent) => {
    resizeRef.current = null;
    try { (e.currentTarget as HTMLElement).releasePointerCapture(e.pointerId); }
    catch { /* noop */ }
  }, []);

  // Portal to document.body so the panel escapes any ancestor that might
  // toggle display:none (e.g. the sidebar collapses its content tree to a
  // 16 px rail via `.hidden`, which would otherwise hide the panel too even
  // though it's position:fixed). Fixed positioning does NOT escape an
  // ancestor with display:none — only a portal does.
  return createPortal(
    <div
      className="fixed z-40 rounded-xl bg-gradient-panel border border-geo-border shadow-2xl flex flex-col"
      data-testid="draggable-panel"
      style={{
        left: geom.x,
        top: geom.y,
        width: geom.w,
        height: geom.h,
      }}
    >
      <div
        className="flex-shrink-0 flex items-center justify-between gap-2 px-3 py-2 border-b border-geo-border rounded-t-xl bg-geo-surface cursor-grab active:cursor-grabbing select-none"
        onPointerDown={onHeaderPointerDown}
        onPointerMove={onHeaderPointerMove}
        onPointerUp={onHeaderPointerUp}
        onPointerCancel={onHeaderPointerUp}
        data-testid="draggable-panel-header"
      >
        <span className="text-[12px] font-semibold font-mono uppercase tracking-wider text-geo-text truncate">
          {title}
        </span>
        <button
          type="button"
          data-no-drag
          onClick={onClose}
          title="Close panel (dock back to sidebar)"
          className="flex-shrink-0 text-[14px] leading-none px-2 py-0.5 rounded hover:bg-geo-border/50 text-geo-muted hover:text-geo-text cursor-pointer"
          data-testid="draggable-panel-close"
        >
          ×
        </button>
      </div>
      <div className="flex-1 min-h-0 overflow-hidden p-3">
        {children}
      </div>
      {/* Corner resize handle — visual is a subtle chevron. Pointer capture on
          this element lets the user drag to resize without the underlying
          panel body catching the events. */}
      <div
        onPointerDown={onResizeDown}
        onPointerMove={onResizeMove}
        onPointerUp={onResizeUp}
        onPointerCancel={onResizeUp}
        className="absolute bottom-0 right-0 w-4 h-4 cursor-se-resize text-geo-muted/40 hover:text-geo-accent select-none"
        title="Drag to resize"
        data-testid="draggable-panel-resize"
        aria-label="Resize panel"
      >
        <svg viewBox="0 0 12 12" className="w-full h-full">
          <path d="M11 1 L11 11 L1 11" fill="none" stroke="currentColor" strokeWidth="1.5" />
          <path d="M11 5 L5 11" fill="none" stroke="currentColor" strokeWidth="1.5" />
        </svg>
      </div>
    </div>,
    document.body,
  );
}
