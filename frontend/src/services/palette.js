/**
 * Colour palette shared with the backend.
 *
 * The backend (`src/utils/viz.py`) derives a colour from the class id with a
 * golden-ratio hue offset. The same formula is reproduced here so that the
 * canvas, the legend and the server-rendered PNG always use the exact same
 * colour for a given class.
 */

/** Port of Python's `colorsys.hsv_to_rgb` (the one used by the backend). */
export function hsvToRgb(h, s, v) {
  if (s === 0) {
    return { r: v, g: v, b: v };
  }
  const i = Math.floor(h * 6);
  const f = h * 6 - i;
  const p = v * (1 - s);
  const q = v * (1 - s * f);
  const t = v * (1 - s * (1 - f));
  const index = ((i % 6) + 6) % 6;

  switch (index) {
    case 0:
      return { r: v, g: t, b: p };
    case 1:
      return { r: q, g: v, b: p };
    case 2:
      return { r: p, g: v, b: t };
    case 3:
      return { r: p, g: q, b: v };
    case 4:
      return { r: t, g: p, b: v };
    default:
      return { r: v, g: p, b: q };
  }
}

/** @returns {[number, number, number]} the RGB triplet of a class. */
export function classColor(classId) {
  const id = Number.isFinite(classId) ? classId : 0;
  const hue = (id * 0.618033988749895) % 1;
  const { r, g, b } = hsvToRgb(hue, 0.85, 0.95);
  // Math.floor mirrors Python's int() truncation.
  return [Math.floor(r * 255), Math.floor(g * 255), Math.floor(b * 255)];
}

/** @returns {string} the CSS colour of a class. */
export function classColorCss(classId) {
  const [r, g, b] = classColor(classId);
  return `rgb(${r}, ${g}, ${b})`;
}

/** Fallback palette, used when the backend metadata could not be fetched. */
export const DEFAULT_CLASSES = [
  'Hardhat',
  'Mask',
  'NO-Hardhat',
  'NO-Mask',
  'NO-Safety Vest',
  'Person',
  'Safety Cone',
  'Safety Vest',
  'Machinery',
  'Vehicle',
];
