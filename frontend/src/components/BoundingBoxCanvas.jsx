import { useEffect, useRef } from 'react';
import { classColorCss } from '../services/palette.js';

/**
 * Canvas that draws the detected bounding boxes on top of the source image.
 *
 * The image is displayed with `object-contain`, so the drawing surface has to
 * be scaled back to the *intrinsic* size of the image before the boxes (which
 * are expressed in image pixels) can be drawn.
 */
export default function BoundingBoxCanvas({
  imageUrl,
  detections = [],
  imageSize = null,
  className = '',
}) {
  const canvasRef = useRef(null);
  const imageRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !imageUrl) return undefined;

    const image = new Image();
    image.crossOrigin = 'anonymous';
    let cancelled = false;

    image.onload = () => {
      if (cancelled) return;
      imageRef.current = image;

      const context = canvas.getContext('2d');
      if (!context) return;

      canvas.width = image.naturalWidth;
      canvas.height = image.naturalHeight;

      context.clearRect(0, 0, canvas.width, canvas.height);
      context.drawImage(image, 0, 0, canvas.width, canvas.height);

      const scale = Math.max(1, canvas.width / 700);
      const fontSize = Math.round(18 * scale);
      context.font = `600 ${fontSize}px ui-sans-serif, system-ui, sans-serif`;
      context.textBaseline = 'top';

      detections.forEach((detection) => {
        const [x1, y1, x2, y2] = detection.bbox;
        const width = x2 - x1;
        const height = y2 - y1;
        const color = classColorCss(detection.class_id);

        // Box
        context.lineWidth = Math.max(2, 3 * scale);
        context.strokeStyle = color;
        context.strokeRect(x1, y1, width, height);

        // Label background
        const text = `${detection.label} ${(detection.confidence * 100).toFixed(0)}%`;
        const textWidth = context.measureText(text).width;
        const boxHeight = fontSize + 8 * scale;
        const labelTop = Math.max(0, y1 - boxHeight);

        context.fillStyle = color;
        context.fillRect(x1, labelTop, textWidth + 10 * scale, boxHeight);

        // Label text
        context.fillStyle = '#ffffff';
        context.fillText(text, x1 + 5 * scale, labelTop + 4 * scale);
      });
    };

    image.src = imageUrl;

    return () => {
      cancelled = true;
    };
  }, [imageUrl, detections]);

  return (
    <div className={`relative overflow-hidden rounded-2xl bg-slate-950 ${className}`}>
      <canvas
        ref={canvasRef}
        className="block h-auto w-full"
        aria-label="Detected objects"
        data-image-size={
          imageSize ? `${imageSize.width}x${imageSize.height}` : undefined
        }
      />
      {detections.length === 0 && (
        <div className="pointer-events-none absolute inset-0 flex items-center justify-center">
          <span className="rounded-full bg-slate-900/80 px-4 py-2 text-sm text-slate-300">
            No object detected with the current thresholds
          </span>
        </div>
      )}
    </div>
  );
}
