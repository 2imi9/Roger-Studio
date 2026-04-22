import { useCallback, useEffect, useLayoutEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";

/**
 * Shepherd.js-inspired guided tour.
 *
 * Why not install shepherd.js itself: ~70 KB gzipped + a style opinion
 * (pure CSS) that would clash with our Tailwind/design-token setup.
 * The tour surface we actually need is small — spotlight a target
 * element, show a tooltip nearby, step through with Back/Next — so
 * we roll ~150 lines that fit our design system.
 *
 * Usage:
 *   const [open, setOpen] = useState(true);
 *   {open && <GuidedTour steps={STEPS} onClose={() => setOpen(false)} />}
 *
 * Each step:
 *   - ``target``: CSS selector for the UI element to highlight. If it
 *     isn't in the DOM when the step fires, the step is skipped.
 *   - ``title`` + ``body``: what to tell the user.
 *   - ``placement``: "auto" (default — picks a sensible side based on
 *     where the target sits in the viewport), or a fixed side.
 */

export interface TourStep {
  /** CSS selector (one element) or array of selectors (union spotlight
   * covering all matching elements). Multi-target is useful when a
   * "feature" spans multiple UI surfaces, e.g., map canvas controls
   * live in separate top-left and top-right containers and should all
   * highlight together even though they don't share a parent. */
  target: string | string[];
  title: string;
  body: React.ReactNode;
  placement?: "auto" | "top" | "bottom" | "left" | "right";
}

interface GuidedTourProps {
  steps: TourStep[];
  onClose: () => void;
}

interface Layout {
  rect: DOMRect;
  placement: "top" | "bottom" | "left" | "right";
}

const TOOLTIP_WIDTH = 420;
const TOOLTIP_MARGIN = 16;
const SPOTLIGHT_PAD = 6;

function chooseAutoPlacement(rect: DOMRect): "top" | "bottom" | "left" | "right" {
  const vw = window.innerWidth;
  const vh = window.innerHeight;
  // Prefer below if there's room; otherwise above; otherwise right; otherwise left.
  if (vh - rect.bottom > 200) return "bottom";
  if (rect.top > 200) return "top";
  if (vw - rect.right > TOOLTIP_WIDTH + TOOLTIP_MARGIN) return "right";
  return "left";
}

function computeTooltipPos(
  layout: Layout,
): { top: number; left: number } {
  const { rect, placement } = layout;
  const vw = window.innerWidth;
  const vh = window.innerHeight;
  // Approximate tooltip height. We don't measure to avoid a double layout
  // pass. ~260 px is correct within a step or two for our fixed font
  // sizes after the font bump, and we clamp to the viewport below.
  const estHeight = 260;

  let top = 0;
  let left = 0;
  switch (placement) {
    case "bottom":
      top = rect.bottom + TOOLTIP_MARGIN;
      left = Math.max(
        TOOLTIP_MARGIN,
        Math.min(vw - TOOLTIP_WIDTH - TOOLTIP_MARGIN, rect.left + rect.width / 2 - TOOLTIP_WIDTH / 2),
      );
      break;
    case "top":
      top = rect.top - estHeight - TOOLTIP_MARGIN;
      left = Math.max(
        TOOLTIP_MARGIN,
        Math.min(vw - TOOLTIP_WIDTH - TOOLTIP_MARGIN, rect.left + rect.width / 2 - TOOLTIP_WIDTH / 2),
      );
      break;
    case "right":
      left = rect.right + TOOLTIP_MARGIN;
      top = Math.max(TOOLTIP_MARGIN, Math.min(vh - estHeight - TOOLTIP_MARGIN, rect.top));
      break;
    case "left":
      left = rect.left - TOOLTIP_WIDTH - TOOLTIP_MARGIN;
      top = Math.max(TOOLTIP_MARGIN, Math.min(vh - estHeight - TOOLTIP_MARGIN, rect.top));
      break;
  }
  // Final clamp so the tooltip can't be pushed fully off-screen on odd
  // viewports (e.g. preview iframe at 282 px wide).
  left = Math.max(TOOLTIP_MARGIN, Math.min(vw - TOOLTIP_WIDTH - TOOLTIP_MARGIN, left));
  top = Math.max(TOOLTIP_MARGIN, Math.min(vh - estHeight - TOOLTIP_MARGIN, top));
  return { top, left };
}

export function GuidedTour({ steps, onClose }: GuidedTourProps) {
  const [idx, setIdx] = useState(0);
  const [layout, setLayout] = useState<Layout | null>(null);
  // Ref-mirror of the current index. Used by the keyboard handler +
  // the Next/Finish button so the decision "advance vs. finish" always
  // reads the up-to-date index regardless of how stale the effect's
  // closure might be across re-renders. Without this we saw the Finish
  // button (and Enter/→ key) "recycle" to step 1 when the closure's
  // idx was captured from an earlier render.
  const idxRef = useRef(0);
  useEffect(() => { idxRef.current = idx; }, [idx]);
  // Guard against double-firing of onClose (button click + keydown
  // that follows within the same React batch). Once we've asked the
  // parent to close, don't advance or re-close even if another event
  // sneaks through before the component unmounts.
  const closingRef = useRef(false);
  const finish = useCallback(() => {
    if (closingRef.current) return;
    closingRef.current = true;
    onClose();
  }, [onClose]);
  const advance = useCallback(() => {
    if (closingRef.current) return;
    if (idxRef.current >= steps.length - 1) {
      finish();
    } else {
      setIdx((i) => Math.min(steps.length - 1, i + 1));
    }
  }, [steps.length, finish]);
  const retreat = useCallback(() => {
    if (closingRef.current) return;
    setIdx((i) => Math.max(0, i - 1));
  }, []);

  const recompute = useCallback(() => {
    if (idx < 0 || idx >= steps.length) return;
    const step = steps[idx];
    // Normalize to array so we can always take a union.
    const selectors = Array.isArray(step.target) ? step.target : [step.target];
    const elements: Element[] = [];
    for (const sel of selectors) {
      const el = document.querySelector(sel);
      if (el) elements.push(el);
    }
    if (elements.length === 0) {
      // No targets in the DOM. Skip forward; if already at the last
      // step, close the tour.
      if (idx < steps.length - 1) setIdx((i) => i + 1);
      else onClose();
      return;
    }
    // Union bounding box — smallest rect that contains every matched
    // element. Single-target reduces to the element's own rect.
    let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
    for (const el of elements) {
      const r = el.getBoundingClientRect();
      if (r.left < minX) minX = r.left;
      if (r.top < minY) minY = r.top;
      if (r.right > maxX) maxX = r.right;
      if (r.bottom > maxY) maxY = r.bottom;
    }
    const rect = new DOMRect(minX, minY, maxX - minX, maxY - minY);
    const placement = step.placement && step.placement !== "auto"
      ? step.placement
      : chooseAutoPlacement(rect);
    setLayout({ rect, placement });
    // Scroll the first matched element into a comfortable reading
    // position. ``nearest`` avoids jump when already visible.
    elements[0].scrollIntoView({ block: "nearest", behavior: "smooth" });
  }, [idx, steps, onClose]);

  useLayoutEffect(() => {
    recompute();
    // Rerun on any viewport/scroll change — tour tooltip tracks the target.
    const onResize = () => recompute();
    window.addEventListener("resize", onResize);
    window.addEventListener("scroll", onResize, true);
    // MutationObserver catches cases where the target's size/position
    // changes from something OTHER than scroll/resize (e.g. the user
    // opens a dropdown that shifts the layout).
    const mo = new MutationObserver(() => recompute());
    mo.observe(document.body, { childList: true, subtree: true, attributes: true });
    return () => {
      window.removeEventListener("resize", onResize);
      window.removeEventListener("scroll", onResize, true);
      mo.disconnect();
    };
  }, [recompute]);

  // Keyboard nav. Esc dismisses at any time. Left arrow steps back.
  // Right arrow / Enter steps forward, and on the LAST step they close
  // the tour so Enter doubles as the Finish shortcut (matches the ↵
  // hint on the Finish button). Delegates to advance/retreat/finish so
  // the idxRef-based decision logic is shared with the button
  // handlers and can't drift out of sync.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        finish();
      } else if (e.key === "ArrowRight" || e.key === "Enter") {
        advance();
      } else if (e.key === "ArrowLeft") {
        retreat();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [advance, retreat, finish]);

  if (!layout) return null;

  const cur = steps[idx];
  const isFirst = idx === 0;
  const isLast = idx === steps.length - 1;
  const { rect } = layout;
  const ttPos = computeTooltipPos(layout);

  // Four overlay bands that leave a cutout for the target. Using four
  // rectangles rather than a CSS mask keeps this working in every
  // browser we care about without a polyfill.
  const overlayCls = "fixed bg-black/55 pointer-events-auto z-[49]";

  return createPortal(
    <div data-testid="guided-tour" aria-live="polite">
      {/* Top band */}
      <div
        className={overlayCls}
        style={{ top: 0, left: 0, right: 0, height: Math.max(0, rect.top - SPOTLIGHT_PAD) }}
        onClick={finish}
      />
      {/* Bottom band */}
      <div
        className={overlayCls}
        style={{ top: rect.bottom + SPOTLIGHT_PAD, left: 0, right: 0, bottom: 0 }}
        onClick={finish}
      />
      {/* Left band */}
      <div
        className={overlayCls}
        style={{
          top: Math.max(0, rect.top - SPOTLIGHT_PAD),
          left: 0,
          width: Math.max(0, rect.left - SPOTLIGHT_PAD),
          height: rect.height + SPOTLIGHT_PAD * 2,
        }}
        onClick={finish}
      />
      {/* Right band */}
      <div
        className={overlayCls}
        style={{
          top: Math.max(0, rect.top - SPOTLIGHT_PAD),
          left: rect.right + SPOTLIGHT_PAD,
          right: 0,
          height: rect.height + SPOTLIGHT_PAD * 2,
        }}
        onClick={finish}
      />

      {/* Spotlight outline — 2 px accent ring + glow that pulses so the
          user's eye lands on the highlighted element without us having
          to physically point at it. */}
      <div
        className="fixed rounded-lg pointer-events-none z-[50]"
        style={{
          top: rect.top - SPOTLIGHT_PAD,
          left: rect.left - SPOTLIGHT_PAD,
          width: rect.width + SPOTLIGHT_PAD * 2,
          height: rect.height + SPOTLIGHT_PAD * 2,
          boxShadow: "0 0 0 2px #3a6690, 0 0 20px 4px rgba(58, 102, 144, 0.5)",
          animation: "tour-pulse 2s ease-in-out infinite",
        }}
      />

      {/* Tooltip card */}
      <div
        className="fixed z-[51] bg-geo-bg border border-geo-border rounded-xl shadow-2xl overflow-hidden"
        style={{ top: ttPos.top, left: ttPos.left, width: TOOLTIP_WIDTH }}
        role="dialog"
        aria-modal="true"
        aria-labelledby="tour-step-title"
      >
        <div className="px-5 py-2.5 border-b border-geo-border bg-gradient-panel flex items-center justify-between">
          <span className="text-[12px] font-mono uppercase tracking-wider text-geo-muted">
            Step {idx + 1} of {steps.length}
          </span>
          <button
            type="button"
            onClick={finish}
            className="text-geo-dim hover:text-geo-text text-lg leading-none cursor-pointer"
            aria-label="Close tour"
          >
            ×
          </button>
        </div>
        <div className="px-5 py-4">
          <h3 id="tour-step-title" className="m-0 text-base font-semibold text-geo-text">
            {cur.title}
          </h3>
          <div className="mt-2 text-[14px] text-geo-text leading-relaxed">
            {cur.body}
          </div>
        </div>
        <div className="px-5 py-2.5 border-t border-geo-border bg-geo-surface flex items-center justify-between gap-3">
          <button
            type="button"
            onClick={finish}
            className="text-[12px] text-geo-dim hover:text-geo-text cursor-pointer"
          >
            Skip tour
          </button>
          {/* Arrow-key shortcut hint so users know Back/Next can be
              driven from the keyboard too. Uses real <kbd> elements so
              the icons read right with screen readers and match the
              bigger body font. */}
          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={retreat}
              disabled={isFirst}
              title="Back (left arrow key)"
              className={`inline-flex items-center gap-1.5 px-3 py-1.5 text-[13px] font-semibold rounded border border-geo-border ${
                isFirst
                  ? "text-geo-dim cursor-not-allowed"
                  : "text-geo-text hover:bg-geo-elevated cursor-pointer"
              }`}
            >
              <kbd
                aria-hidden="true"
                className={`font-mono text-[11px] px-1.5 py-0.5 rounded border ${
                  isFirst
                    ? "border-geo-border/60 text-geo-dim"
                    : "border-geo-border text-geo-muted bg-geo-bg"
                }`}
              >
                ←
              </kbd>
              Back
            </button>
            <button
              type="button"
              onClick={isLast ? finish : advance}
              title={isLast ? "Finish (Enter)" : "Next (right arrow key)"}
              data-testid={isLast ? "tour-finish-button" : "tour-next-button"}
              className="inline-flex items-center gap-1.5 px-3 py-1.5 text-[13px] font-semibold bg-gradient-primary text-white rounded shadow-sm hover:shadow cursor-pointer"
            >
              {isLast ? "Finish" : "Next"}
              <kbd
                aria-hidden="true"
                className="font-mono text-[11px] px-1.5 py-0.5 rounded border border-white/40 text-white/90 bg-white/10"
              >
                {isLast ? "↵" : "→"}
              </kbd>
            </button>
          </div>
        </div>
      </div>

      {/* Keyframes for the spotlight pulse. Scoped via style tag so we
          don't have to register a Tailwind custom animation. */}
      <style>{`@keyframes tour-pulse {
        0%, 100% { box-shadow: 0 0 0 2px #3a6690, 0 0 20px 4px rgba(58, 102, 144, 0.5); }
        50%      { box-shadow: 0 0 0 2px #3a6690, 0 0 28px 8px rgba(58, 102, 144, 0.7); }
      }`}</style>
    </div>,
    document.body,
  );
}
