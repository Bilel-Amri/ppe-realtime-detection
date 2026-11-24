import { useEffect, useState } from 'react';
import { fetchSamples, sampleUrl } from '../services/api.js';

/**
 * Gallery of pre-loaded images: one click runs the inference.
 */
export default function SampleGallery({ onSelect, disabled = false }) {
  const [samples, setSamples] = useState([]);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;
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
  }, []);

  if (loading) {
    return <p className="text-sm text-slate-400">Loading the sample gallery...</p>;
  }

  if (error) {
    return <p className="text-sm text-amber-400">Gallery unavailable: {error}</p>;
  }

  if (!samples.length) {
    return (
      <p className="text-sm text-slate-400">
        No sample image found. Add files to <code>backend/app/samples/</code> or run{' '}
        <code>python scripts/extract_samples.py</code>.
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
          className="group relative overflow-hidden rounded-xl border border-slate-700 bg-slate-900 transition hover:border-sky-500 disabled:opacity-40"
          title={`Analyse ${sample.name}`}
        >
          <img
            src={sampleUrl(sample.name)}
            alt={sample.name}
            loading="lazy"
            className="h-24 w-full object-cover transition-transform duration-300 group-hover:scale-105"
          />
          <span className="absolute inset-x-0 bottom-0 truncate bg-slate-950/80 px-2 py-1 text-[11px] text-slate-300">
            {sample.name}
          </span>
        </button>
      ))}
    </div>
  );
}
