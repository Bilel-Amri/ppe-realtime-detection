import { classColorCss } from '../services/palette.js';

/**
 * Animated confidence bars, one per detected class.
 *
 * The bar width uses a CSS transition so the value slides smoothly instead of
 * jumping - this is what the brief asks for ("scores de probabilités sous
 * forme de barres de progression animées").
 */
export default function ConfidenceBars({ detections = [], classes = [] }) {
  if (!detections.length) {
    return (
      <p className="rounded-xl border border-slate-700/60 bg-slate-900/50 p-4 text-sm text-slate-400">
        No detection yet. Upload an image or start the webcam to see the per-class confidence.
      </p>
    );
  }

  // Aggregate the detections by class, keeping the best score of each.
  const byClass = new Map();
  detections.forEach((detection) => {
    const current = byClass.get(detection.class_id);
    if (!current || detection.confidence > current.confidence) {
      byClass.set(detection.class_id, {
        classId: detection.class_id,
        label: detection.label,
        confidence: detection.confidence,
        count: (current?.count ?? 0) + 1,
      });
    } else {
      current.count += 1;
    }
  });

  const rows = Array.from(byClass.values()).sort((a, b) => b.confidence - a.confidence);

  return (
    <ul className="space-y-3">
      {rows.map((row) => {
        const percent = Math.round(row.confidence * 100);
        const name =
          classes.find((item) => item.id === row.classId)?.name ?? row.label ?? 'Unknown';
        return (
          <li key={row.classId} className="space-y-1.5">
            <div className="flex items-baseline justify-between text-sm">
              <span className="flex items-center gap-2 font-medium text-slate-200">
                <span
                  className="inline-block h-3 w-3 rounded-sm"
                  style={{ backgroundColor: classColorCss(row.classId) }}
                  aria-hidden="true"
                />
                {name}
                {row.count > 1 && (
                  <span className="rounded-full bg-slate-700 px-2 py-0.5 text-xs text-slate-300">
                    x{row.count}
                  </span>
                )}
              </span>
              <span className="font-mono text-xs text-slate-400">{percent}%</span>
            </div>
            <div className="h-2.5 w-full overflow-hidden rounded-full bg-slate-800">
              <div
                className="h-full rounded-full transition-all duration-700 ease-out"
                style={{
                  width: `${percent}%`,
                  backgroundColor: classColorCss(row.classId),
                }}
              />
            </div>
          </li>
        );
      })}
    </ul>
  );
}
