import Sparkline from './Sparkline.jsx';
import { classColorCss } from '../services/palette.js';

/**
 * Session metrics panel:
 *   - server inference latency vs client round-trip latency,
 *   - latency history of the last 30 requests (inline-SVG sparkline),
 *   - FPS during webcam mode,
 *   - detection count and per-class distribution.
 */
export default function StatsPanel({
  result = null,
  metadata = null,
  health = null,
  serverMs = null,
  clientMs = null,
  history = [],
  fps = null,
}) {
  const counts = new Map();
  if (result?.detections) {
    result.detections.forEach((detection) => {
      const key = detection.label ?? `class ${detection.class_id}`;
      counts.set(key, (counts.get(key) ?? 0) + 1);
    });
  }
  const maxCount = Math.max(1, ...counts.values());

  const serverHistory = history.map((entry) => entry.server).filter((v) => typeof v === 'number');

  const tone = (value) => {
    if (value === null || value === undefined) return 'text-muted';
    if (value < 50) return 'text-emerald-400';
    if (value < 150) return 'text-amber-400';
    return 'text-rose-400';
  };

  const format = (value) => (value === null || value === undefined ? '--' : value.toFixed(1));

  return (
    <div className="space-y-4">
      {/* ------------------------------------------------------- metrics */}
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
        <div className="rounded-xl border border-line bg-raised p-3">
          <p className="text-[11px] uppercase tracking-wide text-faint">Server</p>
          <p className={`mt-1 font-mono text-lg font-bold ${tone(serverMs)}`}>
            {format(serverMs)}
            <span className="ml-0.5 text-xs font-normal text-faint">ms</span>
          </p>
        </div>
        <div className="rounded-xl border border-line bg-raised p-3">
          <p className="text-[11px] uppercase tracking-wide text-faint">Client</p>
          <p className={`mt-1 font-mono text-lg font-bold ${tone(clientMs)}`}>
            {format(clientMs)}
            <span className="ml-0.5 text-xs font-normal text-faint">ms</span>
          </p>
        </div>
        <div className="rounded-xl border border-line bg-raised p-3">
          <p className="text-[11px] uppercase tracking-wide text-faint">Webcam FPS</p>
          <p className="mt-1 font-mono text-lg font-bold text-ink">
            {fps === null || fps === undefined ? '--' : fps.toFixed(1)}
          </p>
        </div>
        <div className="rounded-xl border border-line bg-raised p-3">
          <p className="text-[11px] uppercase tracking-wide text-faint">Detections</p>
          <p className="mt-1 font-mono text-lg font-bold text-ink">{result?.count ?? 0}</p>
        </div>
      </div>

      {/* ----------------------------------------------------- sparkline */}
      <div className="rounded-xl border border-line bg-raised p-3">
        <div className="mb-1 flex items-baseline justify-between">
          <p className="text-[11px] uppercase tracking-wide text-faint">
            Server latency - last {serverHistory.length}/30
          </p>
          {serverHistory.length > 0 && (
            <p className="font-mono text-[11px] text-muted">
              avg{' '}
              {(
                serverHistory.reduce((a, b) => a + b, 0) / serverHistory.length
              ).toFixed(1)}{' '}
              ms
            </p>
          )}
        </div>
        <Sparkline values={serverHistory} />
      </div>

      {/* ------------------------------------------- per-class counts */}
      {counts.size > 0 ? (
        <div className="rounded-xl border border-line bg-raised p-3">
          <p className="mb-2 text-[11px] uppercase tracking-wide text-faint">
            Class distribution
          </p>
          <ul className="space-y-1.5">
            {Array.from(counts.entries())
              .sort((a, b) => b[1] - a[1])
              .map(([label, count]) => {
                const classId =
                  result?.detections?.find((detection) => detection.label === label)?.class_id ?? 0;
                return (
                  <li key={label} className="flex items-center gap-2 text-sm">
                    <span
                      className="inline-block h-2.5 w-2.5 shrink-0 rounded-sm"
                      style={{ backgroundColor: classColorCss(classId) }}
                      aria-hidden="true"
                    />
                    <span className="truncate text-muted">{label}</span>
                    <span className="relative h-2 flex-1 overflow-hidden rounded-full bg-panel">
                      <span
                        className="absolute inset-y-0 left-0 rounded-full"
                        style={{
                          width: `${(count / maxCount) * 100}%`,
                          backgroundColor: classColorCss(classId),
                        }}
                      />
                    </span>
                    <span className="w-6 shrink-0 text-right font-mono text-xs text-ink">{count}</span>
                  </li>
                );
              })}
          </ul>
        </div>
      ) : (
        <div className="rounded-xl border border-line bg-raised p-3">
          <p className="text-[11px] uppercase tracking-wide text-faint">Class distribution</p>
          <p className="mt-2 text-sm text-muted">
            No detection yet - upload an image or start the webcam.
          </p>
        </div>
      )}

      {/* ------------------------------------------------ model + health */}
      <div className="rounded-xl border border-line bg-raised p-3 text-xs text-muted">
        <p className="flex flex-wrap items-center gap-x-3 gap-y-1">
          <span className="font-semibold text-ink">
            {metadata?.base_model === 'yolo11n.pt' ? 'YOLO11n' : (metadata?.base_model ?? '--')}
          </span>
          <span>{metadata?.architecture ?? '--'}</span>
          <span>
            {metadata?.backend === 'onnx' ? 'ONNX Runtime (CPU)' : (metadata?.backend ?? '--')}
            {metadata?.model_kind === 'coco-fallback' ? ' - COCO fallback' : ''}
          </span>
        </p>
        {health && (
          <p className="mt-1 text-faint">
            uptime {Math.round(health.uptime_s)}s &middot;{' '}
            {health.gpu_available ? (health.gpu_name ?? 'GPU') : 'CPU only'}
          </p>
        )}
      </div>
    </div>
  );
}
