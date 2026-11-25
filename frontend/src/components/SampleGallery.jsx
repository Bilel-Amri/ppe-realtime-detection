import { useCallback, useEffect, useState } from 'react';
import { fetchSamples, sampleUrl } from '../services/api.js';

/**
 * Gallery of pre-loaded images: one click runs the inference.
 * Shows a skeleton while loading and a useful empty/error state.
 */
export default function SampleGallery({ onSelect, disabled = false }) {
  const [samples, setSamples] = useState([]);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(true);
  const [attempt, setAttempt] = useState(0);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    fetchSamples()
      .then((payload) => {
        if (!cancelled) {
          setSamples(payload.samples ?? []);
          setError(null);
        }
      })
      .catch((exception) => {
        if (!cancelled) setError(exception.message ?? 'The gallery is unavailable.');
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [attempt]);

  const retry = useCallback(() => setAttempt((value) => value + 1), []);

  if (loading) {
    return (
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-3" aria-busy="true" aria-label="Loading samples">
        {[0, 1, 2, 3, 4, 5].map((i) => (
          <div key={i} className="skeleton h-24 rounded-xl" />
        ))}
      </div>
    );
  }

  if (error) {
    return (
      <div className="rounded-xl border border-line bg-raised p-4 text-center">
        <p className="text-sm text-amber-400">Gallery unavailable: {error}</p>
        <button
          type="button"
          onClick={retry}
          className="mt-3 rounded-lg border border-line bg-panel px-3 py-1.5 text-xs font-semibold text-ink transition hover:border-sky-500/60"
        >
          Retry
        </button>
      </div>
    );
  }

  if (!samples.length) {
    return (
      <p className="text-sm text-muted">
        No sample image found. Add files to <code className="font-mono">backend/app/samples/</code>{' '}
        or run <code className="font-mono">python scripts/extract_samples.py</code>.
      </p>
    );
  }

  return (
    <div className="grid grid-cols-2 gap-3 sm:grid-cols-3">
      {samples.map((sample) => (
        <button
          key={sample.name}
          type="button"
          disabled={disabled}
          onClick={() => onSelect(sampleUrl(sample.name), sample.name)}
          className="group relative overflow-hidden rounded-xl border border-line bg-raised transition hover:border-sky-500 disabled:opacity-40"
          title={`Analyse ${sample.name}`}
        >
          <img
            src={sampleUrl(sample.name)}
            alt={sample.name}
            loading="lazy"
            className="h-24 w-full object-cover transition-transform duration-300 group-hover:scale-105"
          />
          <span className="absolute inset-x-0 bottom-0 truncate bg-black/70 px-2 py-1 text-[11px] text-white">
            {sample.name}
          </span>
        </button>
      ))}
    </div>
  );
}
