import { classColorCss } from '../services/palette.js';

/**
 * Detection controls panel.
 *
 * Only `conf` and `iou` are sent to the API - they are the only optional
 * parameters accepted by `POST /predict` (verified against
 * `backend/app/main.py`). Class visibility, labels and the export actions
 * are purely client-side.
 */
export default function DetectionControls({
  conf = 0.25,
  iou = 0.45,
  onConfChange,
  onIouChange,
  onReset,
  classes = [],
  hiddenClasses = [],
  onToggleClass,
  showLabels = true,
  onToggleLabels,
  onDownload,
  onCopyJson,
  canExport = false,
  busy = false,
}) {
  const hidden = new Set(hiddenClasses);

  return (
    <div className="rounded-2xl border border-line bg-panel p-4">
      <div className="mb-3 flex items-center justify-between gap-2">
        <h2 className="text-sm font-semibold text-ink">Detection controls</h2>
        <button
          type="button"
          onClick={onReset}
          className="rounded-lg border border-line bg-raised px-2.5 py-1 text-xs font-medium text-muted transition hover:text-ink"
        >
          Reset
        </button>
      </div>

      <div className="space-y-3">
        {/* ---------------------------------------------------- thresholds */}
        <label className="block">
          <span className="flex items-baseline justify-between text-xs text-muted">
            Confidence threshold
            <span className="font-mono font-semibold text-ink">{conf.toFixed(2)}</span>
          </span>
          <input
            type="range"
            min="0.05"
            max="0.95"
            step="0.05"
            value={conf}
            disabled={busy}
            onChange={(event) => onConfChange?.(Number(event.target.value))}
            className="mt-1.5 w-full accent-sky-500 disabled:opacity-50"
            aria-label="Confidence threshold"
          />
        </label>

        <label className="block">
          <span className="flex items-baseline justify-between text-xs text-muted">
            NMS IoU threshold
            <span className="font-mono font-semibold text-ink">{iou.toFixed(2)}</span>
          </span>
          <input
            type="range"
            min="0.10"
            max="0.90"
            step="0.05"
            value={iou}
            disabled={busy}
            onChange={(event) => onIouChange?.(Number(event.target.value))}
            className="mt-1.5 w-full accent-sky-500 disabled:opacity-50"
            aria-label="NMS IoU threshold"
          />
        </label>

        <p className="text-[11px] text-faint">
          Applied on the next analysis (sent as <code className="font-mono">conf</code> /{' '}
          <code className="font-mono">iou</code> query parameters).
        </p>

        {/* -------------------------------------------------- class toggles */}
        <div>
          <p className="mb-1.5 text-xs text-muted">Class visibility</p>
          <div className="flex flex-wrap gap-1.5">
            {classes.map((cls) => {
              const isHidden = hidden.has(cls.name);
              return (
                <button
                  key={cls.id}
                  type="button"
                  aria-pressed={!isHidden}
                  onClick={() => onToggleClass?.(cls.name)}
                  className={`inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-xs transition ${
                    isHidden
                      ? 'border-line bg-raised text-faint opacity-60'
                      : 'border-line bg-raised text-ink hover:border-sky-500/60'
                  }`}
                  title={
                    isHidden
                      ? `Show ${cls.name}`
                      : `Hide ${cls.name}`
                  }
                >
                  <span
                    className="inline-block h-2.5 w-2.5 rounded-sm"
                    style={{
                      backgroundColor: classColorCss(cls.id),
                      opacity: isHidden ? 0.3 : 1,
                    }}
                    aria-hidden="true"
                  />
                  <span className={isHidden ? 'line-through' : ''}>{cls.name}</span>
                </button>
              );
            })}
          </div>
        </div>

        {/* ------------------------------------------------------ label toggle */}
        <label className="flex cursor-pointer items-center justify-between gap-2 text-xs text-muted">
          Show labels on boxes
          <span className="relative inline-flex">
            <input
              type="checkbox"
              checked={showLabels}
              onChange={(event) => onToggleLabels?.(event.target.checked)}
              className="peer sr-only"
            />
            <span className="h-5 w-9 rounded-full bg-raised ring-1 ring-line transition peer-checked:bg-sky-600" />
            <span className="absolute left-0.5 top-0.5 h-4 w-4 rounded-full bg-white shadow transition peer-checked:translate-x-4" />
          </span>
        </label>

        {/* --------------------------------------------------------- exports */}
        <div className="grid grid-cols-2 gap-2 pt-1">
          <button
            type="button"
            onClick={onDownload}
            disabled={!canExport}
            className="rounded-xl border border-line bg-raised px-3 py-2 text-xs font-semibold text-ink transition hover:border-sky-500/60 disabled:cursor-not-allowed disabled:opacity-40"
          >
            Download PNG
          </button>
          <button
            type="button"
            onClick={onCopyJson}
            disabled={!canExport}
            className="rounded-xl border border-line bg-raised px-3 py-2 text-xs font-semibold text-ink transition hover:border-sky-500/60 disabled:cursor-not-allowed disabled:opacity-40"
          >
            Copy JSON
          </button>
        </div>
      </div>
    </div>
  );
}
