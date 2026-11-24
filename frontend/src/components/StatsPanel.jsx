import { classColorCss } from '../services/palette.js';

/**
 * Compact summary of the last inference:
 * number of objects, per-class counters and the metadata of the model.
 */
export default function StatsPanel({ result = null, metadata = null, health = null }) {
  const counts = new Map();
  if (result?.detections) {
    result.detections.forEach((detection) => {
      const key = detection.label ?? `class ${detection.class_id}`;
      counts.set(key, (counts.get(key) ?? 0) + 1);
    });
  }

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-2 gap-3">
        <div className="rounded-xl border border-slate-700/60 bg-slate-900/50 p-3">
          <p className="text-xs uppercase tracking-wide text-slate-500">Objects</p>
          <p className="mt-1 text-2xl font-bold text-slate-100">{result?.count ?? 0}</p>
        </div>
        <div className="rounded-xl border border-slate-700/60 bg-slate-900/50 p-3">
          <p className="text-xs uppercase tracking-wide text-slate-500">Model</p>
          <p className="mt-1 truncate text-sm font-semibold text-slate-100">
            {metadata?.base_model ?? '--'}
          </p>
          <p className="truncate text-xs text-slate-500">
            {metadata?.backend ?? '--'}
            {metadata?.model_kind === 'coco-fallback'
              ? ' - COCO fallback (not PPE-trained)'
              : ''}
          </p>
        </div>
      </div>

      {counts.size > 0 && (
        <ul className="space-y-1.5 text-sm">
          {Array.from(counts.entries())
            .sort((a, b) => b[1] - a[1])
            .map(([label, count]) => {
              const classId =
                result.detections.find((detection) => detection.label === label)?.class_id ?? 0;
              return (
                <li key={label} className="flex items-center justify-between">
                  <span className="flex items-center gap-2 text-slate-300">
                    <span
                      className="inline-block h-2.5 w-2.5 rounded-sm"
                      style={{ backgroundColor: classColorCss(classId) }}
                      aria-hidden="true"
                    />
                    {label}
                  </span>
                  <span className="font-mono text-slate-400">{count}</span>
                </li>
              );
            })}
        </ul>
      )}

      {metadata?.metrics && Object.keys(metadata.metrics).length > 0 && (
        <div className="rounded-xl border border-slate-700/60 bg-slate-900/50 p-3 text-xs">
          <p className="mb-2 font-semibold uppercase tracking-wide text-slate-500">
            Measured test metrics
          </p>
          <dl className="grid grid-cols-2 gap-x-3 gap-y-1 font-mono text-slate-300">
            {['test_mAP50', 'test_mAP50_95', 'precision', 'recall']
              .filter((key) => typeof metadata.metrics[key] === 'number')
              .map((key) => (
                <div key={key} className="flex justify-between">
                  <dt className="text-slate-500">{key}</dt>
                  <dd>{metadata.metrics[key].toFixed(3)}</dd>
                </div>
              ))}
          </dl>
        </div>
      )}

      {health && (
        <p className="text-xs text-slate-500">
          API {health.status} &middot; GPU{' '}
          {health.gpu_available ? (health.gpu_name ?? 'available') : 'CPU only'} &middot; uptime{' '}
          {Math.round(health.uptime_s)}s
        </p>
      )}
    </div>
  );
}
