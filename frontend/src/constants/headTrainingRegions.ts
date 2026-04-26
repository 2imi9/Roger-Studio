/**
 * Per-head approximate training-distribution regions, used to drive the
 * Off-Distribution Banner in OlmoEarthImport. These are intentionally
 * coarse (axis-aligned bboxes, not real polygons) — the goal is to flag
 * "your AOI is on a different continent than the data this head saw"
 * with high confidence, not to draw a publication-ready coverage map.
 *
 * If a head id isn't in this map, no banner is shown (the FT generic
 * coverage hint still applies for caveats; an explicit banner would be
 * misleading without grounded coverage data).
 *
 * Sources for the bboxes:
 *   * LFMC — chaparral / fire-prone CONUS west; demo AOI is Riverside, CA
 *   * Mangrove — tropical belt; the head was trained globally on tropical
 *     coastlines, so we treat the entire ±30° latitude band as in-distribution
 *   * AWF — southern Kenya only (head's actual training extent)
 *   * ForestLossDriver — pantropical broadleaf forest; ±23.5° latitude
 *   * EcosystemTypeMapping — north Africa (per the head's training docs;
 *     used as the demo example in Claude Design's v2 banner mock)
 */

import type { BBox } from "../types";

export interface HeadTrainingRegion {
  /** What to call this region in UI ("North Africa", "Southern Kenya", etc.) */
  label: string;
  /** Axis-aligned bbox covering the head's training distribution. */
  bbox: BBox;
  /** Tight one-sentence description of what was trained on, used as the
   * second line of the banner ("EcosystemTypeMapping was trained on
   * north Africa only; predictions over your AOI are exploratory."). */
  copy: string;
}

export const HEAD_TRAINING_REGIONS: Record<string, HeadTrainingRegion> = {
  "allenai/OlmoEarth-v1-FT-LFMC-Base": {
    label: "Western US (fire-prone)",
    bbox: { west: -125.0, south: 31.0, east: -103.0, north: 49.0 },
    copy:
      "LFMC was trained on the fire-prone western United States; predictions "
      + "outside that region reflect chaparral / Mediterranean fuel ecology and "
      + "may not transfer to wetter or non-fire regimes.",
  },
  "allenai/OlmoEarth-v1-FT-Mangrove-Base": {
    label: "Tropical belt (±30°)",
    // Mangrove training is global tropical coast — anything OUTSIDE
    // ±30° latitude is off-distribution. Lon span is the whole world.
    bbox: { west: -180.0, south: -30.0, east: 180.0, north: 30.0 },
    copy:
      "The Mangrove head was trained on tropical coastlines (±30° latitude). "
      + "Predictions outside that band cannot reflect real mangrove biology "
      + "and will surface as confident-looking false positives.",
  },
  "allenai/OlmoEarth-v1-FT-AWF-Base": {
    label: "Southern Kenya",
    bbox: { west: 33.5, south: -5.0, east: 42.0, north: -1.0 },
    copy:
      "AWF land-use was trained on a single southern-Kenya scene set "
      + "(Tsavo East and surrounds). Class identities reflect that "
      + "savanna / shrubland mix and will not transfer to other "
      + "biomes.",
  },
  "allenai/OlmoEarth-v1-FT-ForestLossDriver-Base": {
    label: "Pantropical broadleaf",
    bbox: { west: -180.0, south: -23.5, east: 180.0, north: 23.5 },
    copy:
      "ForestLossDriver was trained on pantropical broadleaf forest "
      + "(Amazon, Congo, SE Asia) within ±23.5° latitude. Outside the "
      + "tropics, the driver classes reflect tropical biology and won't "
      + "translate to boreal / temperate forest dynamics.",
  },
  "allenai/OlmoEarth-v1-FT-EcosystemTypeMapping-Base": {
    label: "North Africa",
    bbox: { west: -17.0, south: 14.0, east: 36.0, north: 36.0 },
    copy:
      "EcosystemTypeMapping was trained on north Africa only; predictions "
      + "outside the Saharan / Sahelian belt are exploratory. Class "
      + "identities reflect those ecosystems and will not transfer.",
  },
};

/** Return the AOI's center as [lon, lat], or null if no AOI. */
export function aoiCenter(aoi: BBox | null): [number, number] | null {
  if (!aoi) return null;
  return [(aoi.west + aoi.east) / 2, (aoi.south + aoi.north) / 2];
}

/** Return the bbox center as [lon, lat]. */
export function bboxCenter(b: BBox): [number, number] {
  return [(b.west + b.east) / 2, (b.south + b.north) / 2];
}

/** True iff every corner of the AOI bbox is inside the training bbox.
 * Conservative on purpose — even partial overlap can produce mostly-
 * off-distribution outputs, so we surface the warning unless the AOI is
 * fully inside. Returns false if either argument is null. */
export function isAoiInsideTrainingRegion(
  aoi: BBox | null,
  region: BBox | null,
): boolean {
  if (!aoi || !region) return false;
  return (
    aoi.west >= region.west
    && aoi.east <= region.east
    && aoi.south >= region.south
    && aoi.north <= region.north
  );
}

/** Great-circle distance in km between two [lon, lat] points. */
export function greatCircleKm(a: [number, number], b: [number, number]): number {
  const R = 6371;
  const toRad = (d: number) => (d * Math.PI) / 180;
  const lat1 = toRad(a[1]);
  const lat2 = toRad(b[1]);
  const dLat = toRad(b[1] - a[1]);
  const dLon = toRad(b[0] - a[0]);
  const x =
    Math.sin(dLat / 2) ** 2
    + Math.cos(lat1) * Math.cos(lat2) * Math.sin(dLon / 2) ** 2;
  return Math.round(2 * R * Math.atan2(Math.sqrt(x), Math.sqrt(1 - x)));
}
