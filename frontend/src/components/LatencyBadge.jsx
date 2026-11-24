/**
 * Latency badge.
 *
 * Two numbers are displayed:
 *   - the **server** latency returned by the API (`inference_time_ms`);
 *   - the **client** latency measured in the browser (upload + round trip).
 *
 * The brief explicitly asks for both.
 */
export default function LatencyBadge({ serverMs = null, clientMs = null, busy = false }) {
  const format = (value) =>
    value === null || value === undefined ? '--' : `${value.toFixed(1)} ms`;

  const tone = (value) => {
    if (value === null || value === undefined) return 'text-slate-400';
    if (value < 50) return 'text-emerald-400';
    if (value < 150) return 'text-amber-400';
    return 'text-rose-400';
  };

  return (
    <div className="flex flex-wrap items-center gap-2 text-xs">
      <span
        className={`inline-flex items-center gap-2 rounded-full border border-slate-700 bg-slate-900/70 px-3 py-1.5 font-mono ${tone(
          serverMs,
        )}`}
      >
        <span className="text-slate-500">server</span>
        {format(serverMs)}
      </span>
      <span
        className={`inline-flex items-center gap-2 rounded-full border border-slate-700 bg-slate-900/70 px-3 py-1.5 font-mono ${tone(
          clientMs,
        )}`}
      >
        <span className="text-slate-500">client</span>
        {format(clientMs)}
      </span>
      {busy && (
        <span className="inline-flex items-center gap-2 rounded-full bg-sky-500/10 px-3 py-1.5 text-sky-300">
          <span className="h-2 w-2 animate-ping rounded-full bg-sky-400" />
          analysing
        </span>
      )}
    </div>
  );
}
