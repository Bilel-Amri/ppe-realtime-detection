/**
 * Latency badge.
 *
 * Two numbers are displayed:
 *   - the **server** latency returned by the API (`inference_time_ms`);
 *   - the **client** latency measured in the browser (upload + round trip).
 */
export default function LatencyBadge({ serverMs = null, clientMs = null, busy = false }) {
  const format = (value) =>
    value === null || value === undefined ? '--' : `${value.toFixed(1)} ms`;

  const tone = (value) => {
    if (value === null || value === undefined) return 'text-muted';
    if (value < 50) return 'text-emerald-400';
    if (value < 150) return 'text-amber-400';
    return 'text-rose-400';
  };

  return (
    <div className="flex flex-wrap items-center gap-2 text-xs">
      <span
        className={`inline-flex items-center gap-2 rounded-full border border-line bg-raised px-3 py-1.5 font-mono ${tone(
          serverMs,
        )}`}
      >
        <span className="text-faint">server</span>
        {format(serverMs)}
      </span>
      <span
        className={`inline-flex items-center gap-2 rounded-full border border-line bg-raised px-3 py-1.5 font-mono ${tone(
          clientMs,
        )}`}
      >
        <span className="text-faint">client</span>
        {format(clientMs)}
      </span>
      {busy && (
        <span className="inline-flex items-center gap-2 rounded-full bg-sky-500/10 px-3 py-1.5 text-sky-400">
          <span className="h-2 w-2 animate-ping rounded-full bg-sky-400" />
          analysing
        </span>
      )}
    </div>
  );
}
