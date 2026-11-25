import { classColorCss } from '../services/palette.js';

/**
 * Model dashboard tab - all figures come from `GET /model/metadata`
 * (which reads `artifacts/metrics.json` produced by `src/models/evaluate.py`).
 *
 * Required wording is preserved verbatim: **YOLO11n** and
 * **ONNX Runtime (CPU)**.
 */

const SWAGGER_URL = 'http://localhost:8000/docs';
const MLFLOW_URL = 'http://localhost:5000';

function MetricTile({ label, value }) {
  const display = typeof value === 'number' ? value.toFixed(4) : '--';
  const percent = typeof value === 'number' ? `${(value * 100).toFixed(1)}%` : null;
  return (
    <div className="rounded-xl border border-line bg-raised p-3">
      <p className="text-xs uppercase tracking-wide text-faint">{label}</p>
      <p className="mt-1 font-mono text-xl font-bold text-ink">{display}</p>
      {percent && <p className="text-xs text-muted">{percent}</p>}
    </div>
  );
}

function SkeletonTile() {
  return (
    <div className="rounded-xl border border-line bg-raised p-3">
      <div className="skeleton h-3 w-20 rounded" />
      <div className="skeleton mt-2 h-6 w-24 rounded" />
    </div>
  );
}

export default function ModelDashboard({ metadata = null, loading = false, health = null }) {
  if (loading && !metadata) {
    return (
      <div className="space-y-4" aria-busy="true" aria-label="Loading model metadata">
        <div className="skeleton h-24 w-full rounded-2xl" />
        <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
          {[0, 1, 2, 3].map((i) => (
            <SkeletonTile key={i} />
          ))}
        </div>
        <div className="skeleton h-64 w-full rounded-2xl" />
      </div>
    );
  }

  if (!metadata) {
    return (
      <div className="rounded-2xl border border-line bg-panel p-8 text-center">
        <p className="font-semibold text-ink">Model metadata unavailable</p>
        <p className="mt-1 text-sm text-muted">
          The backend did not answer <code className="font-mono">GET /model/metadata</code>.
        </p>
        <p className="mt-3 text-xs text-faint">Check that the API is running on port 8000.</p>
      </div>
    );
  }

  const metrics = metadata.metrics ?? {};
  const perClass = metrics.per_class_ap50_95 ?? {};
  const classRows = (metadata.classes ?? []).map((cls) => ({
    ...cls,
    ap: typeof perClass[cls.name] === 'number' ? perClass[cls.name] : null,
  }));
  const maxAp = Math.max(1, ...classRows.map((row) => row.ap ?? 0));

  const engine =
    metadata.backend === 'onnx'
      ? health?.gpu_available
        ? 'ONNX Runtime (GPU)'
        : 'ONNX Runtime (CPU)'
      : 'Ultralytics (PyTorch)';

  return (
    <div className="space-y-4">
      {/* ------------------------------------------------------- identity */}
      <div className="rounded-2xl border border-line bg-panel p-5">
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div>
            <h2 className="text-lg font-bold text-ink">
              YOLO11n
              <span className="ml-2 align-middle rounded-md bg-sky-500/15 px-2 py-0.5 text-xs font-semibold text-sky-400">
                {metadata.model_kind === 'coco-fallback' ? 'COCO fallback' : 'PPE fine-tuned'}
              </span>
            </h2>
            <dl className="mt-2 grid gap-x-6 gap-y-1 text-sm sm:grid-cols-2">
              <div className="flex gap-2">
                <dt className="text-muted">Architecture</dt>
                <dd className="font-medium text-ink">{metadata.architecture}</dd>
              </div>
              <div className="flex gap-2">
                <dt className="text-muted">Inference backend</dt>
                <dd className="font-medium text-ink">{engine}</dd>
              </div>
              <div className="flex gap-2">
                <dt className="text-muted">Classes</dt>
                <dd className="font-medium text-ink">{metadata.num_classes}</dd>
              </div>
              <div className="flex gap-2">
                <dt className="text-muted">Input size</dt>
                <dd className="font-medium text-ink">{metadata.imgsz} px</dd>
              </div>
            </dl>
          </div>

          <div className="flex flex-col gap-2">
            <a
              href={SWAGGER_URL}
              target="_blank"
              rel="noreferrer"
              className="inline-flex items-center gap-1.5 rounded-xl border border-line bg-raised px-3 py-2 text-xs font-semibold text-ink transition hover:border-sky-500/60"
            >
              <svg
                className="h-3.5 w-3.5"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
                viewBox="0 0 24 24"
                aria-hidden="true"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14"
                />
              </svg>
              FastAPI Swagger
            </a>
            <a
              href={MLFLOW_URL}
              target="_blank"
              rel="noreferrer"
              className="inline-flex items-center gap-1.5 rounded-xl border border-line bg-raised px-3 py-2 text-xs font-semibold text-ink transition hover:border-sky-500/60"
            >
              <svg
                className="h-3.5 w-3.5"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
                viewBox="0 0 24 24"
                aria-hidden="true"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14"
                />
              </svg>
              MLflow
            </a>
          </div>
        </div>

        {metrics.measured_at && (
          <p className="mt-3 border-t border-line pt-3 text-xs text-faint">
            Measured {metrics.measured_at} &middot; {metrics.dataset ?? 'test split'}
          </p>
        )}
      </div>

      {/* -------------------------------------------------------- metrics */}
      <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
        <MetricTile label="mAP50" value={metrics.test_mAP50} />
        <MetricTile label="mAP50-95" value={metrics.test_mAP50_95} />
        <MetricTile label="Precision" value={metrics.precision} />
        <MetricTile label="Recall" value={metrics.recall} />
      </div>

      {/* ------------------------------------------- per-class AP chart */}
      <div className="rounded-2xl border border-line bg-panel p-5">
        <h3 className="mb-1 text-sm font-semibold text-ink">Per-class AP50-95</h3>
        <p className="mb-4 text-xs text-faint">Average precision at IoU 0.50:0.95 on the test split</p>

        {classRows.length === 0 ? (
          <p className="text-sm text-muted">No per-class metrics available.</p>
        ) : (
          <ul className="space-y-2.5">
            {classRows.map((row) => (
              <li key={row.id} className="flex items-center gap-3 text-sm">
                <span
                  className="inline-block h-3 w-3 shrink-0 rounded-sm"
                  style={{ backgroundColor: classColorCss(row.id) }}
                  aria-hidden="true"
                />
                <span className="w-32 shrink-0 truncate text-muted" title={row.name}>
                  {row.name}
                </span>
                <span className="relative h-4 flex-1 overflow-hidden rounded bg-raised">
                  <span
                    className="absolute inset-y-0 left-0 rounded transition-all duration-700"
                    style={{
                      width: `${((row.ap ?? 0) / maxAp) * 100}%`,
                      backgroundColor: classColorCss(row.id),
                      opacity: row.ap === null ? 0.25 : 0.9,
                    }}
                  />
                </span>
                <span className="w-12 shrink-0 text-right font-mono text-xs text-ink">
                  {row.ap === null ? '--' : row.ap.toFixed(3)}
                </span>
              </li>
            ))}
          </ul>
        )}
      </div>

      {/* ----------------------------------------------------- legend */}
      <div className="rounded-2xl border border-line bg-panel p-5">
        <h3 className="mb-3 text-sm font-semibold text-ink">Class colour legend</h3>
        <ul className="flex flex-wrap gap-x-4 gap-y-2">
          {(metadata.classes ?? []).map((cls) => (
            <li key={cls.id} className="flex items-center gap-1.5 text-xs text-muted">
              <span
                className="inline-block h-2.5 w-2.5 rounded-sm"
                style={{ backgroundColor: classColorCss(cls.id) }}
                aria-hidden="true"
              />
              <span className="font-mono text-faint">{cls.id}</span> {cls.name}
            </li>
          ))}
        </ul>
      </div>
    </div>
  );
}
