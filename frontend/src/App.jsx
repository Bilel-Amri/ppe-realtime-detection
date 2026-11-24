import { useCallback, useEffect, useRef, useState } from 'react';
import BoundingBoxCanvas from './components/BoundingBoxCanvas.jsx';
import ConfidenceBars from './components/ConfidenceBars.jsx';
import Dropzone from './components/Dropzone.jsx';
import LatencyBadge from './components/LatencyBadge.jsx';
import SampleGallery from './components/SampleGallery.jsx';
import StatsPanel from './components/StatsPanel.jsx';
import WebcamPanel from './components/WebcamPanel.jsx';
import { fetchHealth, fetchMetadata, predict } from './services/api.js';

const TABS = [
  { id: 'upload', label: 'Upload' },
  { id: 'webcam', label: 'Webcam' },
  { id: 'samples', label: 'Samples' },
];

export default function App() {
  const [metadata, setMetadata] = useState(null);
  const [health, setHealth] = useState(null);
  const [tab, setTab] = useState('upload');
  const [live, setLive] = useState(false);

  const [result, setResult] = useState(null);
  const [imageUrl, setImageUrl] = useState(null);
  const [imageSize, setImageSize] = useState(null);

  const [serverMs, setServerMs] = useState(null);
  const [clientMs, setClientMs] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);

  const [conf, setConf] = useState(0.25);
  const [iou, setIou] = useState(0.45);

  const urlRef = useRef(null);

  // -- bootstrap ---------------------------------------------------------- //
  useEffect(() => {
    let cancelled = false;
    Promise.allSettled([fetchMetadata(), fetchHealth()]).then(([metaResult, healthResult]) => {
      if (cancelled) return;
      if (metaResult.status === 'fulfilled') {
        setMetadata(metaResult.value);
        if (typeof metaResult.value.conf_threshold === 'number') {
          setConf(metaResult.value.conf_threshold);
        }
        if (typeof metaResult.value.iou_threshold === 'number') {
          setIou(metaResult.value.iou_threshold);
        }
      }
      if (healthResult.status === 'fulfilled') setHealth(healthResult.value);
    });
    return () => {
      cancelled = true;
    };
  }, []);

  // -- inference ---------------------------------------------------------- //
  const runInference = useCallback(
    async (file, options = {}) => {
      setBusy(true);
      setError(null);
      const startedAt = performance.now();
      try {
        const payload = await predict(file, {
          conf: options.conf ?? conf,
          iou: options.iou ?? iou,
        });
        const elapsed = performance.now() - startedAt;

        setResult(payload);
        setServerMs(payload.inference_time_ms ?? null);
        setClientMs(elapsed);
        setImageSize(payload.image_size ?? null);

        // Display the analysed image (a blob URL keeps the original file intact).
        if (urlRef.current) URL.revokeObjectURL(urlRef.current);
        const url = URL.createObjectURL(file);
        urlRef.current = url;
        setImageUrl(url);
      } catch (exception) {
        setError(exception?.message ?? 'The inference failed.');
        setResult(null);
        setServerMs(null);
        setClientMs(null);
      } finally {
        setBusy(false);
      }
    },
    [conf, iou],
  );

  const handleSample = useCallback(
    async (url, name) => {
      setBusy(true);
      setError(null);
      const startedAt = performance.now();
      try {
        const response = await fetch(url);
        const blob = await response.blob();
        const file = new File([blob], name, { type: blob.type || 'image/jpeg' });
        await runInference(file);
      } catch (exception) {
        setError(exception?.message ?? 'The sample could not be loaded.');
        setBusy(false);
      }
    },
    [runInference],
  );

  // Release the blob URL when the component unmounts.
  useEffect(
    () => () => {
      if (urlRef.current) URL.revokeObjectURL(urlRef.current);
    },
    [],
  );

  // -- rendering ---------------------------------------------------------- //
  const apiOffline = health && health.status !== 'ok';

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100">
      <header className="border-b border-slate-800 bg-slate-900/60 backdrop-blur">
        <div className="mx-auto flex max-w-7xl flex-wrap items-center justify-between gap-4 px-6 py-4">
          <div>
            <h1 className="text-xl font-bold tracking-tight">
              PPE Detection
              <span className="ml-2 rounded-md bg-sky-500/15 px-2 py-0.5 text-xs font-semibold text-sky-300">
                YOLO11 &middot; real time
              </span>
            </h1>
            <p className="text-sm text-slate-400">
              Personal Protective Equipment detection on construction sites
            </p>
          </div>
          <div className="flex flex-col items-end gap-2">
            <LatencyBadge serverMs={serverMs} clientMs={clientMs} busy={busy} />
            <span
              className={`text-xs font-medium ${
                apiOffline ? 'text-amber-400' : 'text-emerald-400'
              }`}
            >
              {health
                ? `API ${health.status} - ${health.backend}`
                : 'API unreachable'}
            </span>
          </div>
        </div>
      </header>

      <main className="mx-auto grid max-w-7xl gap-6 px-6 py-6 lg:grid-cols-2">
        {/* ---------------------------- inputs --------------------------- */}
        <section className="space-y-4">
          <div className="flex gap-1 rounded-xl border border-slate-800 bg-slate-900/60 p-1">
            {TABS.map((item) => (
              <button
                key={item.id}
                type="button"
                onClick={() => setTab(item.id)}
                className={`flex-1 rounded-lg px-4 py-2 text-sm font-semibold transition ${
                  tab === item.id
                    ? 'bg-sky-600 text-white'
                    : 'text-slate-400 hover:bg-slate-800 hover:text-slate-200'
                }`}
              >
                {item.label}
              </button>
            ))}
          </div>

          {tab === 'upload' && (
            <Dropzone onFile={(file) => runInference(file)} disabled={busy} />
          )}

          {tab === 'webcam' && (
            <WebcamPanel
              onFrame={(file) => runInference(file)}
              live={live}
              onLiveChange={setLive}
              onError={setError}
            />
          )}

          {tab === 'samples' && (
            <SampleGallery onSelect={handleSample} disabled={busy} />
          )}

          {/* thresholds */}
          <div className="rounded-2xl border border-slate-800 bg-slate-900/40 p-4">
            <p className="mb-3 text-sm font-semibold text-slate-200">Detection thresholds</p>
            <label className="mb-3 block text-xs text-slate-400">
              Confidence threshold: <span className="font-mono text-slate-200">{conf}</span>
              <input
                type="range"
                min="0.05"
                max="0.95"
                step="0.05"
                value={conf}
                onChange={(event) => setConf(Number(event.target.value))}
                className="mt-2 w-full accent-sky-500"
              />
            </label>
            <label className="block text-xs text-slate-400">
              NMS IoU threshold: <span className="font-mono text-slate-200">{iou}</span>
              <input
                type="range"
                min="0.1"
                max="0.9"
                step="0.05"
                value={iou}
                onChange={(event) => setIou(Number(event.target.value))}
                className="mt-2 w-full accent-sky-500"
              />
            </label>
          </div>

          {error && (
            <div className="rounded-xl border border-rose-500/40 bg-rose-500/10 p-4 text-sm text-rose-200">
              {error}
            </div>
          )}
        </section>

        {/* ---------------------------- results -------------------------- */}
        <section className="space-y-4">
          {imageUrl ? (
            <BoundingBoxCanvas
              imageUrl={imageUrl}
              detections={result?.detections ?? []}
              imageSize={imageSize}
            />
          ) : (
            <div className="flex min-h-[320px] items-center justify-center rounded-2xl border border-slate-800 bg-slate-900/40 p-8 text-center text-sm text-slate-500">
              The annotated image will appear here.
            </div>
          )}

          <div className="rounded-2xl border border-slate-800 bg-slate-900/40 p-4">
            <p className="mb-3 text-sm font-semibold text-slate-200">Confidence per class</p>
            <ConfidenceBars
              detections={result?.detections ?? []}
              classes={metadata?.classes ?? []}
            />
          </div>

          <div className="rounded-2xl border border-slate-800 bg-slate-900/40 p-4">
            <p className="mb-3 text-sm font-semibold text-slate-200">Session</p>
            <StatsPanel result={result} metadata={metadata} health={health} />
          </div>
        </section>
      </main>

      <footer className="border-t border-slate-800 px-6 py-4 text-center text-xs text-slate-600">
        Deep Learning Avance - Projet de module | Axe 3 : detection d'objets en temps reel
      </footer>
    </div>
  );
}
