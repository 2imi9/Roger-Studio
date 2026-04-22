import { useEffect, useRef, useState } from "react";

// Roger Studio mark — striped earth ring + viewfinder brackets in a
// refractive blue glass plate. Combines the V2 (cursor-tracked brackets,
// click-press shrink) and V3 (click-to-spin) interactions from the
// Claude Design logo prototype into a single "track + spin" component.
//
// Behavior:
//   - Hover: brackets glide toward the cursor (lerped, capped at 26px).
//   - Press (mousedown): brackets pinch inward to 0.72× scale.
//   - Click (mouseup): the earth ring spins 360° with cubic-out easing
//     and a stripe parallax wash; brackets spring back.
//
// The mark is purely decorative — there's no link/button semantics on
// the wrapper, so callers can wrap it in their own <a> or <button> if
// they want it to do something on click beyond the spin animation.

const PALETTE = {
  greens: ["#E3EEDD", "#C9DEBF", "#A9CA99", "#7FB178", "#4F8F5A", "#2F6B3F", "#1E4B2D", "#11331F"],
};

interface RogerLogoProps {
  size?: number;
  /** Disable all interactions (cursor tracking, click spin). Useful in
   *  a static loading splash. Default false. */
  static_?: boolean;
  /** Override the SVG element id prefix — needed only if you mount more
   *  than one logo on the same page (defs ids must be unique). */
  idPrefix?: string;
}

interface MarkProps {
  size: number;
  stripeOffset: number;  // 0–1, shifts stripe parallax
  ringRotate: number;    // deg, rotates the earth ring
  blueDx: number;        // px, brackets offset from anchor
  blueDy: number;
  frameScale: number;    // 0–1+, shrinks brackets on press
  shimmer: number;       // 0–1, highlight sweep across stripes
  id: string;
}

/** The actual SVG mark — pure render, all animation state passed in. */
function Mark({
  size,
  stripeOffset,
  ringRotate,
  blueDx,
  blueDy,
  frameScale,
  shimmer,
  id,
}: MarkProps) {
  const VB = 240;
  const cx = 110;
  const cy = 100;
  const ringOuter = 92;
  const ringInner = 42;
  const stripeCount = 12;

  // Ring stripe rects — vertical bands clipped to the ring annulus.
  const bandW = (ringOuter * 2) / stripeCount;
  const stripes: { x: number; y: number; w: number; h: number; color: string; i: number }[] = [];
  for (let i = 0; i < stripeCount; i++) {
    const x = cx - ringOuter + i * bandW;
    const t = i / (stripeCount - 1);
    const palIdx = Math.round(t * (PALETTE.greens.length - 1));
    stripes.push({
      x,
      y: -ringOuter,
      w: bandW,
      h: ringOuter * 2,
      color: PALETTE.greens[palIdx],
      i,
    });
  }

  // Viewfinder bracket — sits under-right of the ring, crossing its lower-right edge.
  const baseFrameSize = 104;
  const frameSize = baseFrameSize * frameScale;
  const frameR = frameSize / 2;
  const frameCx = cx + 46 + blueDx;
  const frameCy = cy + 58 + blueDy;
  const frameRadius = 26 * frameScale;
  const bracketLen = 38 * frameScale;
  const bracketStroke = 20 * frameScale;

  const ringGroup = (
    <g transform={`rotate(${ringRotate} ${cx} ${cy})`}>
      <g mask={`url(#${id}-ring-mask)`}>
        {stripes.map((s) => {
          const shift = Math.sin((s.i / stripeCount) * Math.PI * 2 + stripeOffset * Math.PI * 2) * 1.5;
          return (
            <rect
              key={s.i}
              x={s.x + shift * 0.6}
              y={cy + s.y}
              width={s.w + 0.5}
              height={s.h}
              fill={s.color}
            />
          );
        })}
        {shimmer > 0 && (
          <rect
            x={cx - ringOuter - 40 + shimmer * (ringOuter * 2 + 80)}
            y={cy - ringOuter}
            width="80"
            height={ringOuter * 2}
            fill={`url(#${id}-shim)`}
            opacity={Math.sin(shimmer * Math.PI)}
          />
        )}
      </g>
    </g>
  );

  return (
    <svg
      viewBox={`0 0 ${VB} 220`}
      width={size}
      height={(size * 220) / VB}
      style={{ display: "block", overflow: "visible" }}
    >
      <defs>
        {/* Annulus mask so stripes only paint inside the ring. */}
        <mask id={`${id}-ring-mask`}>
          <rect x="0" y="0" width={VB} height="220" fill="black" />
          <circle cx={cx} cy={cy} r={ringOuter} fill="white" />
          <circle cx={cx} cy={cy} r={ringInner} fill="black" />
        </mask>
        {/* Diagonal shimmer wash. */}
        <linearGradient id={`${id}-shim`} x1="0" y1="0" x2="1" y2="0">
          <stop offset="0%" stopColor="white" stopOpacity="0" />
          <stop offset="45%" stopColor="white" stopOpacity="0" />
          <stop offset="50%" stopColor="white" stopOpacity="0.55" />
          <stop offset="55%" stopColor="white" stopOpacity="0" />
          <stop offset="100%" stopColor="white" stopOpacity="0" />
        </linearGradient>
        {/* Refraction filter — distorts whatever's underneath the brackets. */}
        <filter id={`${id}-refract`} x="-20%" y="-20%" width="140%" height="140%">
          <feTurbulence type="fractalNoise" baseFrequency="0.02" numOctaves="2" seed="3" result="turb" />
          <feDisplacementMap in="SourceGraphic" in2="turb" scale="4" xChannelSelector="R" yChannelSelector="G" />
        </filter>
        {/* Subtle frost grain on the glass surface. */}
        <filter id={`${id}-grain`} x="-5%" y="-5%" width="110%" height="110%">
          <feTurbulence type="fractalNoise" baseFrequency="1.8" numOctaves="1" seed="7" result="n" />
          <feColorMatrix in="n" type="matrix" values="0 0 0 0 1  0 0 0 0 1  0 0 0 0 1  0 0 0 0.25 0" />
        </filter>
        {/* Blue glass tint. */}
        <radialGradient id={`${id}-tint`} cx="68%" cy="22%" r="95%">
          <stop offset="0%" stopColor="#7FDCFF" stopOpacity="0.42" />
          <stop offset="50%" stopColor="#2AA8F0" stopOpacity="0.55" />
          <stop offset="100%" stopColor="#1166C4" stopOpacity="0.68" />
        </radialGradient>
        {/* Hot specular spot. */}
        <radialGradient id={`${id}-spec`} cx="72%" cy="22%" r="18%">
          <stop offset="0%" stopColor="white" stopOpacity="0.65" />
          <stop offset="60%" stopColor="white" stopOpacity="0.1" />
          <stop offset="100%" stopColor="white" stopOpacity="0" />
        </radialGradient>
        {/* Clip the refracted ring copy to the bracket bbox. */}
        <clipPath id={`${id}-disc-clip`} clipPathUnits="userSpaceOnUse">
          <rect
            x={frameCx - frameR}
            y={frameCy - frameR}
            width={frameSize}
            height={frameSize}
            rx={frameRadius}
            ry={frameRadius}
          />
        </clipPath>
        {/* Mask that reveals only the four corner brackets. */}
        <mask
          id={`${id}-bracket-mask`}
          maskUnits="userSpaceOnUse"
          x={frameCx - frameR - 2}
          y={frameCy - frameR - 2}
          width={frameSize + 4}
          height={frameSize + 4}
        >
          <rect
            x={frameCx - frameR - 2}
            y={frameCy - frameR - 2}
            width={frameSize + 4}
            height={frameSize + 4}
            fill="black"
          />
          <g transform={`translate(${frameCx} ${frameCy})`}>
            {/* Outer rounded square — full silhouette */}
            <rect x={-frameR} y={-frameR} width={frameSize} height={frameSize} rx={frameRadius} ry={frameRadius} fill="white" />
            {/* Cut middle of each edge to leave only the 4 corner Ls */}
            <rect x={-frameR + bracketLen} y={-frameR - 1} width={frameSize - bracketLen * 2} height={bracketStroke + 2} fill="black" />
            <rect x={-frameR + bracketLen} y={frameR - bracketStroke - 1} width={frameSize - bracketLen * 2} height={bracketStroke + 2} fill="black" />
            <rect x={-frameR - 1} y={-frameR + bracketLen} width={bracketStroke + 2} height={frameSize - bracketLen * 2} fill="black" />
            <rect x={frameR - bracketStroke - 1} y={-frameR + bracketLen} width={bracketStroke + 2} height={frameSize - bracketLen * 2} fill="black" />
            {/* Cut interior */}
            <rect x={-frameR + bracketStroke} y={-frameR + bracketStroke} width={frameSize - bracketStroke * 2} height={frameSize - bracketStroke * 2} fill="black" />
          </g>
        </mask>
      </defs>

      {/* 1. Earth ring — stripes, behind everything */}
      {ringGroup}

      {/* 2. Refracted copy of the ring, revealed only through the bracket mask */}
      <g mask={`url(#${id}-bracket-mask)`}>
        <g clipPath={`url(#${id}-disc-clip)`}>
          <g filter={`url(#${id}-refract)`} style={{ transformOrigin: `${frameCx}px ${frameCy}px` }}>
            {ringGroup}
          </g>
        </g>
      </g>

      {/* 3. Glass tint, frost grain, specular hot-spot — all masked to brackets */}
      <g mask={`url(#${id}-bracket-mask)`}>
        <rect x={frameCx - frameR} y={frameCy - frameR} width={frameSize} height={frameSize} rx={frameRadius} ry={frameRadius} fill={`url(#${id}-tint)`} />
        <rect x={frameCx - frameR} y={frameCy - frameR} width={frameSize} height={frameSize} rx={frameRadius} ry={frameRadius} fill="white" opacity="0.03" filter={`url(#${id}-grain)`} />
        <rect x={frameCx - frameR} y={frameCy - frameR} width={frameSize} height={frameSize} rx={frameRadius} ry={frameRadius} fill={`url(#${id}-spec)`} opacity="0.5" />
        <rect x={frameCx - frameR + 0.6} y={frameCy - frameR + 0.6} width={frameSize - 1.2} height={frameSize - 1.2} rx={frameRadius - 0.5} ry={frameRadius - 0.5} fill="none" stroke="white" strokeWidth="0.8" opacity="0.35" />
      </g>

      {/* 4. Black outlines — outer + inner rounded squares + gap-end strokes */}
      <g mask={`url(#${id}-bracket-mask)`}>
        <rect x={frameCx - frameR} y={frameCy - frameR} width={frameSize} height={frameSize} rx={frameRadius} ry={frameRadius} fill="none" stroke="#0B2E5C" strokeWidth="5.5" strokeLinejoin="round" />
        <rect x={frameCx - frameR + bracketStroke} y={frameCy - frameR + bracketStroke} width={frameSize - bracketStroke * 2} height={frameSize - bracketStroke * 2} rx={Math.max(4, frameRadius - bracketStroke)} ry={Math.max(4, frameRadius - bracketStroke)} fill="none" stroke="#0B2E5C" strokeWidth="5.5" strokeLinejoin="round" />
        <line x1={frameCx - frameR + bracketLen} y1={frameCy - frameR} x2={frameCx - frameR + bracketLen} y2={frameCy - frameR + bracketStroke} stroke="#0B2E5C" strokeWidth="5.5" strokeLinecap="round" />
        <line x1={frameCx + frameR - bracketLen} y1={frameCy - frameR} x2={frameCx + frameR - bracketLen} y2={frameCy - frameR + bracketStroke} stroke="#0B2E5C" strokeWidth="5.5" strokeLinecap="round" />
        <line x1={frameCx - frameR + bracketLen} y1={frameCy + frameR - bracketStroke} x2={frameCx - frameR + bracketLen} y2={frameCy + frameR} stroke="#0B2E5C" strokeWidth="5.5" strokeLinecap="round" />
        <line x1={frameCx + frameR - bracketLen} y1={frameCy + frameR - bracketStroke} x2={frameCx + frameR - bracketLen} y2={frameCy + frameR} stroke="#0B2E5C" strokeWidth="5.5" strokeLinecap="round" />
        <line x1={frameCx - frameR} y1={frameCy - frameR + bracketLen} x2={frameCx - frameR + bracketStroke} y2={frameCy - frameR + bracketLen} stroke="#0B2E5C" strokeWidth="5.5" strokeLinecap="round" />
        <line x1={frameCx - frameR} y1={frameCy + frameR - bracketLen} x2={frameCx - frameR + bracketStroke} y2={frameCy + frameR - bracketLen} stroke="#0B2E5C" strokeWidth="5.5" strokeLinecap="round" />
        <line x1={frameCx + frameR - bracketStroke} y1={frameCy - frameR + bracketLen} x2={frameCx + frameR} y2={frameCy - frameR + bracketLen} stroke="#0B2E5C" strokeWidth="5.5" strokeLinecap="round" />
        <line x1={frameCx + frameR - bracketStroke} y1={frameCy + frameR - bracketLen} x2={frameCx + frameR} y2={frameCy + frameR - bracketLen} stroke="#0B2E5C" strokeWidth="5.5" strokeLinecap="round" />
      </g>
    </svg>
  );
}

/**
 * Roger Studio interactive mark.
 *
 * Combines V2 (cursor-track brackets) + V3 (click-spin earth ring) from
 * the Claude Design logo prototype.
 */
export function RogerLogo({ size = 40, static_ = false, idPrefix = "cv" }: RogerLogoProps) {
  // V2 state: bracket position lerped toward cursor.
  const [pos, setPos] = useState({ x: 0, y: 0 });
  const [active, setActive] = useState(false);
  const [pressed, setPressed] = useState(false);
  const [scale, setScale] = useState(1);

  // V3 state: ring rotation + stripe parallax sweep on click.
  const [rot, setRot] = useState(0);
  const [stripeOffset, setStripeOffset] = useState(0);

  const wrapperRef = useRef<HTMLDivElement>(null);
  const target = useRef({ x: 0, y: 0 });
  const targetScale = useRef(1);
  const trackRaf = useRef(0);
  const spinRaf = useRef(0);
  const [spinning, setSpinning] = useState(false);

  // Continuous lerp loop for cursor-tracked bracket position + scale.
  useEffect(() => {
    if (static_) return;
    const tick = () => {
      setPos((p) => ({
        x: p.x + (target.current.x - p.x) * 0.12,
        y: p.y + (target.current.y - p.y) * 0.12,
      }));
      setScale((s) => s + (targetScale.current - s) * 0.22);
      trackRaf.current = requestAnimationFrame(tick);
    };
    trackRaf.current = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(trackRaf.current);
  }, [static_]);

  // Update bracket scale target whenever pressed/active toggles.
  useEffect(() => {
    targetScale.current = pressed ? 0.72 : active ? 1.04 : 1;
  }, [pressed, active]);

  // Cleanup spin RAF on unmount.
  useEffect(() => () => cancelAnimationFrame(spinRaf.current), []);

  const onMove = (e: React.MouseEvent) => {
    if (static_ || !wrapperRef.current) return;
    const r = wrapperRef.current.getBoundingClientRect();
    const nx = ((e.clientX - r.left) / r.width - 0.5) * 2;
    const ny = ((e.clientY - r.top) / r.height - 0.5) * 2;
    const maxR = 26;
    const mag = Math.hypot(nx, ny);
    const k = mag > 1 ? 1 / mag : 1;
    target.current = { x: nx * k * maxR, y: ny * k * maxR };
  };

  const triggerSpin = () => {
    if (static_ || spinning) return;
    setSpinning(true);
    const start = performance.now();
    const from = rot;
    const to = rot + 360;
    const dur = 1400;
    const tick = (t: number) => {
      const p = Math.min(1, (t - start) / dur);
      const e = 1 - Math.pow(1 - p, 3); // ease-out cubic
      setRot(from + (to - from) * e);
      setStripeOffset(e * 2);
      if (p < 1) {
        spinRaf.current = requestAnimationFrame(tick);
      } else {
        setSpinning(false);
        // Reset stripe offset so it doesn't drift on subsequent spins.
        setStripeOffset(0);
        // Normalize rotation to [0, 360) so big multiples don't accumulate.
        setRot((r) => r % 360);
      }
    };
    spinRaf.current = requestAnimationFrame(tick);
  };

  return (
    <div
      ref={wrapperRef}
      onMouseMove={onMove}
      onMouseEnter={() => setActive(true)}
      onMouseLeave={() => {
        setActive(false);
        setPressed(false);
        target.current = { x: 0, y: 0 };
      }}
      onMouseDown={() => setPressed(true)}
      onMouseUp={() => setPressed(false)}
      onClick={triggerSpin}
      style={{
        cursor: static_ ? "default" : "pointer",
        display: "inline-block",
        userSelect: "none",
        // Hide the bottom 1/3 of the SVG (it's mostly empty space the
        // viewBox reserves for bracket overhang). Set in style not
        // className so callers can override via the size prop only.
        lineHeight: 0,
      }}
      title="Roger Studio"
      role="img"
      aria-label="Roger Studio"
    >
      <Mark
        size={size}
        stripeOffset={stripeOffset}
        ringRotate={rot}
        blueDx={pos.x}
        blueDy={pos.y}
        frameScale={scale}
        shimmer={0}
        id={idPrefix}
      />
    </div>
  );
}
