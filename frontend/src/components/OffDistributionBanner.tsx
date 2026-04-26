/**
 * Off-Distribution Warning Banner — shown above the Run button in
 * OlmoEarthImport when the user's AOI sits outside the selected FT
 * head's training region.
 *
 * Ported from Claude Design v2 component2 (2026-04-26). Replaces the
 * prior italic gray "FT heads run globally..." paragraph that existed
 * for every FT head regardless of AOI, with a precise, conditional
 * banner that shows only when there's an actual distribution mismatch
 * AND surfaces a small inline world map showing where the head was
 * trained vs where the user is asking it to predict.
 *
 * Visual highlights (ported from v2):
 *   * Bordered amber `rs-banner` chrome (warm-light, on-tone)
 *   * Headline "AOI is outside this model's training region"
 *   * Inline 480×220 SVG world map with:
 *       - Pixelated continent tiles (logo-gradient west→east)
 *       - Graticule + tick labels
 *       - Bracketed training-region rect (HUD-style)
 *       - Great-circle distance arc with km pill
 *       - Red crosshair AOI marker
 *   * Footer with disclosure toggle, "Pick a different head" / "Read
 *     the model card" CTAs, and an advisory "still can run" note.
 *
 * The continent polygons are from v2 — generalized silhouettes, not
 * cartographically accurate; they exist for visual orientation, not
 * navigation.
 */
import { useMemo, useState } from "react";
import type { BBox } from "../types";
import {
  HEAD_TRAINING_REGIONS,
  isAoiInsideTrainingRegion,
  aoiCenter,
  bboxCenter,
  greatCircleKm,
} from "../constants/headTrainingRegions";

interface Props {
  /** The user's drawn area, or null when nothing's selected. The banner
   *  hides while AOI is null — there's no off-distribution claim to make
   *  yet — and the existing required-AOI hint elsewhere covers it. */
  aoi: BBox | null;
  /** The selected FT head repo id. The banner hides for non-FT heads
   *  (base encoders are global) and for heads not in
   *  HEAD_TRAINING_REGIONS. */
  modelRepoId: string;
  /** Pretty model label for the body copy ("Mangrove" → "...EcosystemTypeMapping..."). */
  modelLabel: string;
  /** Click handler: switch to the head picker (focuses the model selector
   *  in OlmoEarthImport so the user can pick a head whose training region
   *  covers their AOI). Optional — button hides if not provided. */
  onPickAnotherHead?: () => void;
}

// Map projection: equirectangular trimmed to lat ∈ [-66, 84] so continents
// fill the frame instead of leaving big polar voids. View box is 480 × 220
// SVG units; consumers scale via CSS max-width.
const LAT_TOP = 84;
const LAT_BOT = -66;
const LAT_SPAN = LAT_TOP - LAT_BOT;
const VW = 480;
const VH = 220;
const STEP = 10; // tile size for the continent raster

function lonLatToXY(lon: number, lat: number): [number, number] {
  return [((lon + 180) / 360) * VW, ((LAT_TOP - lat) / LAT_SPAN) * VH];
}

// Generalized continent silhouettes. NOT cartographically accurate.
const CONTINENTS: [number, number][][] = [
  [[-168,66],[-159,71],[-141,70],[-128,70],[-110,72],[-95,73],[-82,73],[-72,68],
   [-62,60],[-52,52],[-58,46],[-66,44],[-71,41],[-76,38],[-80,32],[-82,27],
   [-80,25],[-87,30],[-94,29],[-97,26],[-105,22],[-107,24],[-110,30],[-117,32],
   [-124,40],[-124,48],[-130,54],[-138,59],[-150,60],[-157,58],[-165,62],[-168,66]],
  [[-90,16],[-83,15],[-77,8],[-78,1],[-80,-4],[-72,-12],[-71,-18],[-70,-23],
   [-71,-30],[-73,-37],[-73,-46],[-71,-54],[-66,-55],[-65,-52],[-58,-39],
   [-56,-34],[-48,-28],[-39,-15],[-35,-8],[-44,-2],[-50,1],[-60,5],[-72,9],
   [-77,8],[-82,9],[-85,11],[-90,16]],
  [[-10,36],[-9,43],[-2,43],[3,43],[8,44],[13,46],[14,40],[18,40],[23,38],
   [27,36],[32,36],[36,36],[36,40],[42,42],[50,40],[54,37],[57,30],[51,25],
   [44,12],[51,15],[58,23],[63,25],[72,18],[78,8],[80,16],[88,21],[92,22],
   [97,16],[103,11],[108,14],[110,21],[115,23],[122,30],[121,40],[129,42],
   [135,46],[140,52],[148,59],[156,64],[170,67],[178,69],[178,71],[160,72],
   [140,73],[110,74],[90,75],[70,75],[55,72],[40,68],[30,70],[20,69],[10,66],
   [5,60],[8,55],[2,52],[-5,49],[-9,44],[-10,36]],
  [[-17,21],[-16,15],[-12,10],[-7,5],[3,4],[10,3],[10,-3],[14,-12],[12,-17],
   [16,-29],[20,-35],[26,-34],[32,-29],[34,-21],[40,-17],[40,-7],[44,1],
   [51,11],[44,12],[37,17],[34,22],[31,29],[25,32],[15,32],[5,32],[-5,32],
   [-12,28],[-17,21]],
  [[114,-22],[122,-18],[131,-12],[136,-13],[141,-10],[145,-15],[151,-25],
   [153,-30],[150,-37],[143,-39],[133,-33],[126,-34],[114,-34],[114,-22]],
  [[-50,82],[-30,82],[-22,76],[-22,68],[-42,60],[-52,65],[-54,73],[-50,82]],
];

const MONO = 'ui-monospace, SFMono-Regular, Menlo, monospace';

interface Tile {
  x: number;
  y: number;
  fill: string;
}

/** Sample the continent polygons onto an offscreen canvas, then take an
 *  every-STEP-pixel grid of points that landed on land. Returns a flat
 *  array of {x, y, fill} where fill is the logo-gradient color at that
 *  longitudinal position. */
function buildLandTiles(): Tile[] {
  if (typeof document === "undefined") return [];
  const c = document.createElement("canvas");
  c.width = VW;
  c.height = VH;
  const ctx = c.getContext("2d");
  if (!ctx) return [];
  ctx.fillStyle = "#000";
  for (const cont of CONTINENTS) {
    ctx.beginPath();
    cont.forEach((p, i) => {
      const [x, y] = lonLatToXY(p[0], p[1]);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.closePath();
    ctx.fill();
  }
  const data = ctx.getImageData(0, 0, VW, VH).data;

  // Logo gradient (west → east): pale mint → deep forest. Six stops.
  const STOPS: { t: number; c: [number, number, number] }[] = [
    { t: 0.00, c: [220, 232, 208] },
    { t: 0.22, c: [184, 210, 162] },
    { t: 0.45, c: [138, 178, 116] },
    { t: 0.68, c: [92, 146, 84] },
    { t: 0.85, c: [56, 104, 60] },
    { t: 1.00, c: [30, 72, 44] },
  ];
  const lerpStop = (t: number): string => {
    for (let i = 1; i < STOPS.length; i++) {
      if (t <= STOPS[i].t) {
        const a = STOPS[i - 1];
        const b = STOPS[i];
        const k = (t - a.t) / (b.t - a.t);
        const r = Math.round(a.c[0] + (b.c[0] - a.c[0]) * k);
        const g = Math.round(a.c[1] + (b.c[1] - a.c[1]) * k);
        const bl = Math.round(a.c[2] + (b.c[2] - a.c[2]) * k);
        return `rgb(${r},${g},${bl})`;
      }
    }
    const last = STOPS[STOPS.length - 1].c;
    return `rgb(${last[0]},${last[1]},${last[2]})`;
  };

  const out: Tile[] = [];
  for (let y = 0; y < VH; y += STEP) {
    for (let x = 0; x < VW; x += STEP) {
      const idx = (y * VW + x) * 4 + 3;
      if (data[idx] > 64) {
        // Per-tile jitter so the gradient reads as sampled imagery, not a hard ramp.
        const seed = ((x * 928371) ^ (y * 17389)) >>> 0;
        const jitter = ((seed % 100) / 100 - 0.5) * 0.04;
        const t = Math.max(0, Math.min(1, x / VW + jitter));
        out.push({ x, y, fill: lerpStop(t) });
      }
    }
  }
  return out;
}

function WorldMap({
  trainingBbox,
  aoiBbox,
}: {
  trainingBbox: BBox;
  aoiBbox: BBox;
}) {
  const tiles = useMemo(buildLandTiles, []);

  const aoiC = bboxCenter(aoiBbox);
  const trainingC = bboxCenter(trainingBbox);
  const distKm = greatCircleKm(aoiC, trainingC);

  const [aoiX, aoiY] = lonLatToXY(...aoiC);
  const [trX, trY] = lonLatToXY(...trainingC);
  const ctrlX = (aoiX + trX) / 2;
  const ctrlY = Math.min(aoiY, trY) - 32;
  const mx = 0.25 * trX + 0.5 * ctrlX + 0.25 * aoiX;
  const my = 0.25 * trY + 0.5 * ctrlY + 0.25 * aoiY;

  // Training region screen rect.
  const [trxA, tryA] = lonLatToXY(trainingBbox.west, trainingBbox.north);
  const [trxB, tryB] = lonLatToXY(trainingBbox.east, trainingBbox.south);
  const tw = trxB - trxA;
  const th = tryB - tryA;
  const L = 6; // bracket arm length

  // Graticule lines at every 30°.
  const meridians: number[] = [];
  for (let lon = -150; lon <= 150; lon += 30) meridians.push(((lon + 180) / 360) * VW);
  const parallels: number[] = [];
  for (let lat = -60; lat <= 60; lat += 30) parallels.push(((LAT_TOP - lat) / LAT_SPAN) * VH);

  const distLabel = `${distKm.toLocaleString()} KM`;
  const distLabelW = distLabel.length * 5.6 + 14;

  return (
    <div className="flex justify-center w-full">
      <svg
        viewBox={`-26 -16 ${VW + 52} ${VH + 30}`}
        preserveAspectRatio="xMidYMid meet"
        role="img"
        aria-label={`World map: training coverage and AOI ${distKm} km apart`}
        className="block w-full max-w-[420px] h-auto bg-geo-surface rounded border border-geo-border"
      >
        <defs>
          <linearGradient id="off-dist-land-grad" x1="0" y1="0" x2="1" y2="1">
            <stop offset="0%" stopColor="#ffffff" stopOpacity="0.15" />
            <stop offset="100%" stopColor="#000000" stopOpacity="0.10" />
          </linearGradient>
        </defs>

        {/* Map frame */}
        <rect x={0} y={0} width={VW} height={VH} fill="none" stroke="var(--color-geo-border)" strokeWidth={1} shapeRendering="crispEdges" />

        {/* Graticule */}
        <g stroke="color-mix(in oklch, var(--color-geo-border) 70%, transparent)" strokeWidth={0.5} shapeRendering="crispEdges">
          {meridians.map((x, i) => <line key={`m${i}`} x1={x} y1={0} x2={x} y2={VH} />)}
          {parallels.map((y, i) => <line key={`p${i}`} x1={0} y1={y} x2={VW} y2={y} />)}
        </g>
        <g stroke="var(--color-geo-border)" strokeWidth={0.8} shapeRendering="crispEdges">
          <line x1={0} y1={(LAT_TOP / LAT_SPAN) * VH} x2={VW} y2={(LAT_TOP / LAT_SPAN) * VH} />
          <line x1={VW / 2} y1={0} x2={VW / 2} y2={VH} />
        </g>

        {/* Continent tiles */}
        <g shapeRendering="crispEdges">
          {tiles.map((t, i) => (
            <rect key={i} x={t.x + 0.5} y={t.y + 0.5} width={9} height={9} fill={t.fill} />
          ))}
        </g>
        <g shapeRendering="crispEdges" opacity={0.5}>
          {tiles.map((t, i) => (
            <rect key={`o${i}`} x={t.x + 0.5} y={t.y + 0.5} width={9} height={9} fill="url(#off-dist-land-grad)" />
          ))}
        </g>

        {/* Lon ticks */}
        <g fill="var(--color-geo-muted)" fontFamily={MONO} fontSize={7.5} textAnchor="middle">
          {[-120, -60, 0, 60, 120].map((lon) => {
            const x = ((lon + 180) / 360) * VW;
            const sign = lon < 0 ? "W" : lon > 0 ? "E" : "";
            return <text key={lon} x={x} y={-5}>{`${Math.abs(lon)}°${sign}`}</text>;
          })}
        </g>
        <g fill="var(--color-geo-muted)" fontFamily={MONO} fontSize={7.5} textAnchor="end">
          {[60, 30, 0, -30, -60].map((lat) => {
            const y = ((LAT_TOP - lat) / LAT_SPAN) * VH;
            const sign = lat > 0 ? "N" : lat < 0 ? "S" : "";
            return <text key={lat} x={-5} y={y + 2.5}>{`${Math.abs(lat)}°${sign}`}</text>;
          })}
        </g>

        {/* Training region — fill + bracketed corners */}
        <rect x={trxA} y={tryA} width={tw} height={th}
          fill="var(--color-geo-warn)" fillOpacity={0.22} stroke="none" />
        <g stroke="var(--color-geo-warn)" strokeWidth={1.4} fill="none" shapeRendering="crispEdges">
          <path d={`M${trxA} ${tryA + L} L${trxA} ${tryA} L${trxA + L} ${tryA}`} />
          <path d={`M${trxA + tw - L} ${tryA} L${trxA + tw} ${tryA} L${trxA + tw} ${tryA + L}`} />
          <path d={`M${trxA + tw} ${tryA + th - L} L${trxA + tw} ${tryA + th} L${trxA + tw - L} ${tryA + th}`} />
          <path d={`M${trxA + L} ${tryA + th} L${trxA} ${tryA + th} L${trxA} ${tryA + th - L}`} />
        </g>

        {/* Distance arc */}
        <path
          d={`M ${trX} ${trY} Q ${ctrlX} ${ctrlY} ${aoiX} ${aoiY}`}
          fill="none"
          stroke="var(--color-geo-warn)"
          strokeWidth={0.9}
          strokeDasharray="2 3"
          opacity={0.7}
        />

        {/* Distance pill */}
        <g transform={`translate(${mx - distLabelW / 2} ${my - 9})`}>
          <rect width={distLabelW} height={16} rx={2} ry={2}
            fill="#fbf9f5" stroke="var(--color-geo-warn)" strokeWidth={0.9} />
          <text x={distLabelW / 2} y={11} fontSize={9} fontFamily={MONO}
            fill="#7a4e0a" textAnchor="middle" fontWeight={700}
            letterSpacing="0.06em">
            {distLabel}
          </text>
        </g>

        {/* AOI crosshair */}
        <g stroke="var(--color-geo-danger)" strokeWidth={1.4} shapeRendering="crispEdges">
          <line x1={aoiX - 8} y1={aoiY} x2={aoiX - 3} y2={aoiY} />
          <line x1={aoiX + 3} y1={aoiY} x2={aoiX + 8} y2={aoiY} />
          <line x1={aoiX} y1={aoiY - 8} x2={aoiX} y2={aoiY - 3} />
          <line x1={aoiX} y1={aoiY + 3} x2={aoiX} y2={aoiY + 8} />
        </g>
        <circle cx={aoiX} cy={aoiY} r={2.4} fill="var(--color-geo-danger)" />
        <text x={aoiX} y={aoiY - 11} fontSize={8.5} fontFamily={MONO}
          fontWeight={700} fill="var(--color-geo-danger)" textAnchor="middle"
          letterSpacing="0.1em">AOI</text>

        {/* Training label pill */}
        {(() => {
          const label = "TRAINING";
          const w = label.length * 5.2 + 14;
          const cx = (trxA + trxB) / 2;
          const y = tryB + 6;
          return (
            <g transform={`translate(${cx - w / 2} ${y})`}>
              <rect width={w} height={16} rx={2} ry={2}
                fill="#fbf9f5" stroke="var(--color-geo-warn)" strokeWidth={0.9} />
              <text x={w / 2} y={11} fontSize={9} fontFamily={MONO}
                fill="#7a4e0a" textAnchor="middle" fontWeight={700}
                letterSpacing="0.06em">{label}</text>
            </g>
          );
        })()}
      </svg>
    </div>
  );
}

export function OffDistributionBanner({ aoi, modelRepoId, modelLabel, onPickAnotherHead }: Props) {
  const [open, setOpen] = useState<boolean>(true);

  const region = HEAD_TRAINING_REGIONS[modelRepoId];
  if (!region || !aoi) return null;
  if (isAoiInsideTrainingRegion(aoi, region.bbox)) return null;

  const center = aoiCenter(aoi);
  const trainingC = bboxCenter(region.bbox);
  const distKm = center ? greatCircleKm(center, trainingC) : null;

  return (
    <div
      data-testid="off-distribution-banner"
      className="rounded-md overflow-hidden border bg-geo-warn-soft border-geo-warn/50"
    >
      {/* Headline + body copy */}
      <div className="flex items-start gap-2.5 px-3.5 pt-2.5 pb-3">
        <span className="text-[14px] leading-none mt-0.5 text-geo-warn flex-shrink-0" aria-hidden="true">⚠</span>
        <div className="flex-1 min-w-0">
          <p className="font-display font-bold text-[13.5px] leading-tight tracking-tight text-geo-warn">
            AOI is outside this model's training region
          </p>
          <p className="mt-1 text-[12px] leading-snug text-geo-text/85">
            {region.copy.replace(/^[A-Z][A-Za-z]+/, modelLabel)}
            {distKm != null && (
              <>{" "}<span className="font-mono text-geo-muted">({distKm.toLocaleString()} km from {region.label})</span></>
            )}
          </p>
        </div>
      </div>

      {/* Inline world map */}
      {open && (
        <div className="px-3.5 pb-3 pt-3 border-t border-geo-warn/35">
          <WorldMap trainingBbox={region.bbox} aoiBbox={aoi} />
          <div className="flex justify-center gap-x-4 gap-y-2 flex-wrap mt-2.5 text-[11px] text-geo-muted">
            <span className="inline-flex items-center gap-1.5">
              <span
                className="inline-block w-3 h-2 rounded-[1px] border border-geo-warn"
                style={{ background: "color-mix(in oklch, var(--color-geo-warn) 60%, transparent)" }}
              />
              Training coverage
            </span>
            <span className="inline-flex items-center gap-1.5">
              <span
                className="inline-block w-2 h-2 rounded-full bg-geo-danger ring-[1.4px] ring-geo-bg shadow-[0_0_0_1px_var(--color-geo-danger)]"
              />
              Your AOI
            </span>
          </div>
        </div>
      )}

      {/* Footer with toggle + actions */}
      <div className="flex items-center flex-wrap gap-2 px-2.5 py-1.5 border-t border-geo-warn/35 bg-[color-mix(in_oklch,var(--color-geo-warn-soft)_85%,var(--color-geo-surface))]">
        <button
          type="button"
          onClick={() => setOpen(!open)}
          className="h-[26px] px-2 inline-flex items-center text-[12px] font-medium text-geo-warn hover:bg-geo-elevated/60 rounded transition-colors"
        >
          {open ? "▾ Hide training region" : "▸ Show training region"}
        </button>
        <span className="flex-1" />
        <div className="flex gap-1.5">
          {onPickAnotherHead && (
            <button
              type="button"
              onClick={onPickAnotherHead}
              className="h-[26px] px-2.5 inline-flex items-center text-[12px] font-medium border border-geo-border rounded bg-geo-surface text-geo-text hover:bg-geo-elevated transition-colors"
            >
              Pick a different head
            </button>
          )}
          <a
            href={`https://huggingface.co/${modelRepoId}`}
            target="_blank"
            rel="noreferrer"
            className="h-[26px] px-2.5 inline-flex items-center text-[12px] font-medium border border-geo-border rounded bg-geo-surface text-geo-text hover:bg-geo-elevated transition-colors no-underline"
          >
            Read the model card →
          </a>
        </div>
      </div>

      {/* Advisory */}
      <p className="px-3.5 py-1.5 text-center text-[11px] text-geo-muted bg-geo-bg/60 border-t border-geo-warn/20">
        The warning is advisory; you can still run the model.
      </p>
    </div>
  );
}
