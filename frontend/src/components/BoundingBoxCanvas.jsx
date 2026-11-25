import { useEffect, useRef } from 'react';
import { classColorCss } from '../services/palette.js';

/**
 * Canvas that draws the detected bounding boxes on top of the source image.
 *
 * The backend returns `bbox` as an **object** (`{x1, y1, x2, y2, ...}`) in
 * pixels of the original image, so coordinates are read with
 * `detection.bbox.x1` - never array-destructured.
 *
 * The image is drawn into the canvas at its natural resolution and the
 * canvas is displayed with `width: 100%; height: auto`, so:
 *   - the aspect ratio is always preserved (no letterboxing), and
 *   - boxes stay aligned at every browser size (the whole canvas, image
 *     included, is scaled uniformly by CSS).
 *
 * `object-fit: contain` is handled by construction: the canvas box matches
 * the image aspect ratio, so there is no empty band where boxes could drift.
 */
export default function BoundingBoxCanvas({
  imageUrl,
  detections = [],
  imageSize = null,
  showLabels = true,
  hiddenClasses = [],
  canvasRef = null,
  onImageError = null,
  className = '',
}) {
  const internalRef = useRef(null);
  const ref = canvasRef ?? internalRef;

  // Stable primitive key so a new inline array does not restart the effect.
  const hiddenKey = hiddenClasses.join('\u0000');

  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return undefined;

    if (!imageUrl) {
      const ctx = canvas.getContext('2d');
      ctx?.clearRect(0, 0, canvas.width, canvas.height);
      return undefined;
    }

    const image = new Image();
    image.crossOrigin = 'anonymous';
    let cancelled = false;

    image.onload = () => {
      if (cancelled) return;
      const context = canvas.getContext('2d');
      if (!context) return;

      canvas.width = image.naturalWidth;
      canvas.height = image.naturalHeight;

      context.clearRect(0, 0, canvas.width, canvas.height);
      context.drawImage(image, 0, 0, canvas.width, canvas.height);

      const hidden = new Set(hiddenKey.split('\u0000').filter(Boolean));
      const visible = detections.filter((detection) => !hidden.has(detection.label));

      const scale = Math.max(1, canvas.width / 700);
      const fontSize = Math.round(16 * scale);
      context.font = `600 ${fontSize}px ui-sans-serif, system-ui, sans-serif`;
      context.textBaseline = 'top';
      context.lineJoin = 'round';

      visible.forEach((detection) => {
        const bbox = detection.bbox;
        if (!bbox || typeof bbox.x1 !== 'number') return;

        // `bbox` is an object: read the four coordinates explicitly.
        const x1 = bbox.x1;
        const y1 = bbox.y1;
        const x2 = bbox.x2;
        const y2 = bbox.y2;
        const width = x2 - x1;
        const height = y2 - y1;
        const color = classColorCss(detection.class_id);

        // Box
        context.lineWidth = Math.max(2, 3 * scale);
        context.strokeStyle = color;
        context.strokeRect(x1, y1, width, height);

        if (!showLabels) return;

        // Label background + text, flipped inside the image when too high.
        const text = `${detection.label} ${(detection.confidence * 100).toFixed(0)}%`;
        const textWidth = context.measureText(text).width;
        const boxHeight = fontSize + 8 * scale;
        const labelTop = Math.max(0, y1 - boxHeight);
        const labelLeft = Math.min(Math.max(0, x1), canvas.width - textWidth - 10 * scale);

        context.fillStyle = color;
        context.fillRect(labelLeft, labelTop, textWidth + 10 * scale, boxHeight);

        context.fillStyle = '#ffffff';
        context.fillText(text, labelLeft + 5 * scale, labelTop + 4 * scale);
      });
    };

    image.onerror = () => {
      if (!cancelled) onImageError?.('The image could not be decoded.');
    };

    image.src = imageUrl;

    return () => {
      cancelled = true;
    };
  }, [imageUrl, detections, hiddenKey, showLabels, ref, onImageError]);

  const hidden = new Set(hiddenKey.split('\u0000').filter(Boolean));
  const visibleCount = detections.filter((detection) => !hidden.has(detection.label)).length;

  return (
    <div
      className={`relative overflow-hidden rounded-2xl border border-line bg-black ${className}`}
      data-detection-count={detections.length}
      data-visible-count={visibleCount}
    >
      <canvas
        ref={ref}
        className="block h-auto w-full"
        role="img"
        aria-label={
          detections.length
            ? `Annotated image with ${detections.length} detected object(s)`
            : 'Analysed image'
        }
        data-image-size={imageSize ? `${imageSize.width}x${imageSize.height}` : undefined}
      />
      {detections.length === 0 && (
        <div className="pointer-events-none absolute inset-0 flex items-center justify-center p-4">
          <span className="rounded-full bg-black/75 px-4 py-2 text-center text-sm text-white/90">
            No object detected with the current thresholds
          </span>
        </div>
      )}
      {detections.length > 0 && visibleCount === 0 && (
        <div className="pointer-events-none absolute inset-0 flex items-center justify-center p-4">
          <span className="rounded-full bg-black/75 px-4 py-2 text-center text-sm text-white/90">
            All classes are hidden - enable them in Detection controls
          </span>
        </div>
      )}
    </div>
  );
}
