import { useCallback, useEffect, useRef, useState } from 'react';
import BoundingBoxCanvas from './components/BoundingBoxCanvas.jsx';
import ComplianceBanner from './components/ComplianceBanner.jsx';
import ConfidenceBars from './components/ConfidenceBars.jsx';
import DetectionControls from './components/DetectionControls.jsx';
import Dropzone from './components/Dropzone.jsx';
import LatencyBadge from './components/LatencyBadge.jsx';
import ModelDashboard from './components/ModelDashboard.jsx';
import SampleGallery from './components/SampleGallery.jsx';
import StatsPanel from './components/StatsPanel.jsx';
import WebcamPanel from './components/WebcamPanel.jsx';
import { ApiError, fetchHealth, fetchMetadata, predict } from './services/api.js';

const TABS = [
  { id: 'upload', label: 'Upload' },
  { id: 'webcam', label: 'Webcam' },
  { id: 'samples', label: 'Samples' },
  { id: 'model', label: 'Model' },
];

const DEFAULT_CONF = 0.25;
const DEFAULT_IOU = 0.45;
const HISTORY_LIMIT = 30;
const RERUN_DEBOUNCE_MS = 350;

/** Turn any thrown value into a user-facing message. */
function describeError(exception) {
  if (exception instanceof ApiError) {
    if (exception.status === 400) return `Invalid image or request - ${exception.message}`;
    if (exception.status === 503) return `Model or backend unavailable - ${exception.message}`;
    if (exception.status === 0) return exception.message;
    return exception.message;
  }
  return exception?.message ?? 'Unexpected error while contacting the API.';
}

function ToastStack({ toasts, onDismiss }) {
  if (!toasts.length) return null;
  const tone = {
    error: 'border-rose-500/50 bg-rose-500/10',
    success: 'border-emerald-500/50 bg-emerald-500/10',
    info: 'border-sky-500/50 bg-sky-500/10',
  };
  return (
    <div
      className="pointer-events-none fixed right-4 top-4 z-50 flex w-[min(360px,calc(100vw-2rem))] flex-col gap-2"
      aria-live="polite"
      aria-label="Notifications"
    >
      {toasts.map((toast) => (
        <div
          key={toast.id}
          className={`pointer-events-auto flex items-start gap-2 rounded-xl border px-3 py-2.5 text-sm shadow-xl animate-toast-in ${tone[toast.kind] ?? tone.info}`}
        >
          <span className="flex-1 text-ink">{toast.message}</span>
          <button
            type="button"
            onClick={() => onDismiss(toast.id)}
            className="shrink-0 rounded p-0.5 text-muted transition hover:text-ink"
            aria-label="Dismiss notification"
          >
            <svg
              className="h-3.5 w-3.5"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              viewBox="0 0 24 24"
              aria-hidden="true"
            >
              <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>
      ))}
    </div>
  );
}

function EmptyResult({ onBrowseSamples }) {
  return (
    <div className="flex min-h-[320px] flex-col items-center justify-center gap-3 rounded-2xl border border-dashed border-line bg-panel p-8 text-center">
      <svg
        className="h-10 w-10 text-faint"
        fill="none"
        stroke="currentColor"
        strokeWidth="1.5"
        viewBox="0 0 24 24"
        aria-hidden="true"
      >
        <path
          strokeLinecap="round"
          strokeLinejoin="round"
          d="M2.25 15.75l5.159-5.159a2.25 2.25 0 013.182 0l5.159 5.159m-1.5-1.5l1.409-1.409a2.25 2.25 0 013.182 0l2.909 2.909M3.75 21h16.5A2.25 2.25 0 0022.5 18.75V5.25A2.25 2.25 0 0020.25 3H3.75A2.25 2.25 0 001.5 5.25v13.5A2.25 2.25 0 003.75 21zm12-6.75a.75.75 0 11-1.5 0 .75.75 0 011.5 0z"
        />
      </svg>
      <p className="font-semibold text-ink">The annotated image will appear here</p>
      <p className="max-w-sm text-sm text-muted">
        Upload a photo, paste one with Ctrl+V, or analyse a sample to see bounding boxes, the
        compliance verdict and per-class metrics.
      </p>
      <button
        type="button"
        onClick={onBrowseSamples}
        className="rounded-xl bg-sky-600 px-4 py-2 text-sm font-semibold text-white transition hover:bg-sky-500"
      >
        Browse sample images
      </button>
    </div>
  );
}

export default function App() {
  const [metadata, setMetadata] = useState(null);
  const [health, setHealth] = useState(null);
  const [loading, setLoading] = useState(true);
  const [tab, setTab] = useState('upload');
  const [theme, setTheme] = useState(() => {
    try {
      return localStorage.getItem('ppe-theme') === 'light' ? 'light' : 'dark';
    } catch {
      return 'dark';
    }
  });

  const [result, setResult] = useState(null);
  const [origin, setOrigin] = useState(null); // upload | sample | webcam
  const [imageUrl, setImageUrl] = useState(null);
  const [imageDetections, setImageDetections] = useState([]);
  const [imageSize, setImageSize] = useState(null);
  const [fileName, setFileName] = useState(null);

  const [serverMs, setServerMs] = useState(null);
  const [clientMs, setClientMs] = useState(null);
  const [history, setHistory] = useState([]);
  const [fps, setFps] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);
  const [toasts, setToasts] = useState([]);

  const [conf, setConf] = useState(DEFAULT_CONF);
  const [iou, setIou] = useState(DEFAULT_IOU);
  const [hiddenClasses, setHiddenClasses] = useState([]);
  const [showLabels, setShowLabels] = useState(true);
  const [live, setLive] = useState(false);
  const [violationHistory, setViolationHistory] = useState([]);

  const urlRef = useRef(null);
  const canvasRef = useRef(null);
  const lastRunRef = useRef(null); // { file, origin } of the last non-webcam run
  const tabRef = useRef(tab);
  const tabButtonsRef = useRef([]);

  // ------------------------------------------------------------- theme
  useEffect(() => {
    const root = document.documentElement;
    root.dataset.theme = theme;
    root.classList.toggle('dark', theme === 'dark');
    try {
      localStorage.setItem('ppe-theme', theme);
    } catch {
      /* storage may be unavailable */
    }
  }, [theme]);

  useEffect(() => {
    tabRef.current = tab;
  }, [tab]);

  // ----------------------------------------------------------- toasts
  const pushToast = useCallback((kind, message) => {
    const id = `${Date.now()}-${Math.random().toString(36).slice(2, 7)}`;
    setToasts((current) =>
      [...current.filter((toast) => toast.message !== message), { id, kind, message }].slice(-4),
    );
    window.setTimeout(() => {
      setToasts((current) => current.filter((toast) => toast.id !== id));
    }, 5000);
  }, []);

  const dismissToast = useCallback((id) => {
    setToasts((current) => current.filter((toast) => toast.id !== id));
  }, []);

  // --------------------------------------------------------- bootstrap
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
      } else {
        pushToast('error', `Model metadata unavailable: ${metaResult.reason?.message ?? 'error'}`);
      }
      if (healthResult.status === 'fulfilled') {
        setHealth(healthResult.value);
      } else {
        setHealth({ status: 'down' });
        pushToast(
          'error',
          'Backend unreachable - start it with `docker compose up` or uvicorn on port 8000.',
        );
      }
      setLoading(false);
    });
    return () => {
      cancelled = true;
    };
  }, [pushToast]);

  // Keep polling /health so the online/offline pill stays truthful.
  useEffect(() => {
    const id = window.setInterval(() => {
      fetchHealth()
        .then(setHealth)
        .catch(() => setHealth({ status: 'down' }));
    }, 30000);
    return () => window.clearInterval(id);
  }, []);

  // ------------------------------------------------------- inference
  const runInference = useCallback(
    async (file, options = {}) => {
      const runOrigin = options.origin ?? 'upload';
      setBusy(true);
      setError(null);
      const startedAt = performance.now();
      try {
        const payload = await predict(file, { conf, iou });
        const elapsed = performance.now() - startedAt;

        setResult(payload);
        setOrigin(runOrigin);
        setServerMs(payload.inference_time_ms ?? null);
        setClientMs(elapsed);
        setHistory((current) =>
          [
            ...current,
            { server: payload.inference_time_ms ?? null, client: elapsed },
          ].slice(-HISTORY_LIMIT),
        );

        if (runOrigin !== 'webcam') {
          lastRunRef.current = { file, origin: runOrigin };
          if (urlRef.current) URL.revokeObjectURL(urlRef.current);
          const url = URL.createObjectURL(file);
          urlRef.current = url;
          setImageUrl(url);
          setImageDetections(payload.detections ?? []);
          setImageSize(payload.image_size ?? null);
          setFileName(file.name ?? 'image');
        }

        if (payload.count === 0 && runOrigin !== 'webcam') {
          pushToast('info', 'No object detected at the current thresholds.');
        }
      } catch (exception) {
        const message = describeError(exception);
        setError(message);
        pushToast('error', message);
        setResult(null);
        setOrigin(null);
        setServerMs(null);
        setClientMs(null);
      } finally {
        setBusy(false);
      }
    },
    [conf, iou, pushToast],
  );

  const handleSample = useCallback(
    async (url, name) => {
      setError(null);
      try {
        // `no-store`: the gallery <img> tags cache this URL *without* the
        // CORS headers (the response has no `Vary: Origin`), so a plain fetch
        // can be answered from that poisoned entry and fail with a CORS
        // error. Bypassing the cache always re-sends the `Origin` header.
        const response = await fetch(url, { cache: 'no-store' });
        if (!response.ok) throw new Error(`sample download failed (${response.status})`);
        const blob = await response.blob();
        const file = new File([blob], name, { type: blob.type || 'image/jpeg' });
        await runInference(file, { origin: 'sample' });
      } catch (exception) {
        const message = `The sample could not be loaded - ${exception.message}`;
        setError(message);
        pushToast('error', message);
      }
    },
    [runInference, pushToast],
  );

  // Re-run automatically when a threshold changes (debounced), so the
  // sliders visibly affect the result within the supported API parameters.
  useEffect(() => {
    const last = lastRunRef.current;
    if (!last || tabRef.current === 'webcam') return undefined;
    const timer = window.setTimeout(() => {
      runInference(last.file, { origin: last.origin });
    }, RERUN_DEBOUNCE_MS);
    return () => window.clearTimeout(timer);
  }, [conf, iou, runInference]);

  // Release the blob URL when the component unmounts.
  useEffect(
    () => () => {
      if (urlRef.current) URL.revokeObjectURL(urlRef.current);
    },
    [],
  );

  // ------------------------------------------------------ violations
  const handleViolation = useCallback((event) => {
    setViolationHistory((current) => [event, ...current].slice(0, 8));
  }, []);

  // ------------------------------------- stable callbacks for the webcam
  // (an inline arrow would restart the throttled live loop on every render,
  // so these are memoised).
  const handleWebcamFrame = useCallback(
    (file) => runInference(file, { origin: 'webcam' }),
    [runInference],
  );
  const handleWebcamError = useCallback((message) => pushToast('error', message), [pushToast]);
  const handleWebcamClearError = useCallback(() => setError(null), []);
  const handleImageError = useCallback(
    (message) => pushToast('error', message),
    [pushToast],
  );
  const handleUploadFile = useCallback(
    (file) => runInference(file),
    [runInference],
  );
  const browseSamples = useCallback(() => setTab('samples'), []);

  // -------------------------------------------------------- controls
  const handleReset = useCallback(() => {
    setConf(metadata?.conf_threshold ?? DEFAULT_CONF);
    setIou(metadata?.iou_threshold ?? DEFAULT_IOU);
    setHiddenClasses([]);
    setShowLabels(true);
    pushToast('info', 'Detection controls reset to defaults.');
  }, [metadata, pushToast]);

  const toggleClass = useCallback((name) => {
    setHiddenClasses((current) =>
      current.includes(name) ? current.filter((item) => item !== name) : [...current, name],
    );
  }, []);

  const downloadPng = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas || !imageUrl) {
      pushToast('error', 'Nothing to download yet - analyse an image first.');
      return;
    }
    try {
      const link = document.createElement('a');
      link.download = `ppe-annotated-${Date.now()}.png`;
      link.href = canvas.toDataURL('image/png');
      link.click();
      pushToast('success', 'Annotated image downloaded as PNG.');
    } catch {
      pushToast('error', 'The annotated image could not be exported.');
    }
  }, [imageUrl, pushToast]);

  const copyJson = useCallback(async () => {
    if (!result) {
      pushToast('error', 'No prediction to copy yet - run a detection first.');
      return;
    }
    const text = JSON.stringify(result, null, 2);
    try {
      await navigator.clipboard.writeText(text);
      pushToast('success', 'Prediction JSON copied to the clipboard.');
    } catch {
      try {
        const area = document.createElement('textarea');
        area.value = text;
        area.style.position = 'fixed';
        area.style.opacity = '0';
        document.body.appendChild(area);
        area.select();
        document.execCommand('copy');
        area.remove();
        pushToast('success', 'Prediction JSON copied to the clipboard.');
      } catch {
        pushToast('error', 'Clipboard access was denied by the browser.');
      }
    }
  }, [result, pushToast]);

  // ---------------------------------------------------------- render
  const apiOffline = Boolean(health && health.status !== 'ok');
  const engineLabel =
    health?.backend === 'onnx'
      ? health.gpu_available
        ? 'ONNX Runtime (GPU)'
        : 'ONNX Runtime (CPU)'
      : 'ONNX Runtime (CPU)';
  const statusLabel = loading
    ? 'Connecting to backend...'
    : health?.status === 'ok'
      ? `Backend online - ${engineLabel}`
      : 'Backend offline - check the API connection';

  // The canvas always shows the detections of the *uploaded image*, so live
  // webcam frames never clobber the annotated picture.
  const showCanvas = tab !== 'webcam' && imageUrl && origin !== 'webcam';

  const onTabKeyDown = (event) => {
    if (event.key !== 'ArrowRight' && event.key !== 'ArrowLeft') return;
    event.preventDefault();
    const index = TABS.findIndex((item) => item.id === tab);
    const delta = event.key === 'ArrowRight' ? 1 : -1;
    const next = (index + delta + TABS.length) % TABS.length;
    setTab(TABS[next].id);
    tabButtonsRef.current[next]?.focus();
  };

  return (
    <div className="min-h-screen bg-bg text-ink">
      {/* ------------------------------------------------------- header */}
      <header className="sticky top-0 z-40 border-b border-line bg-panel/90 backdrop-blur">
        <div className="mx-auto flex max-w-shell flex-wrap items-center justify-between gap-3 px-4 py-3 sm:px-6">
          <div className="flex items-center gap-3">
            <span
              className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-sky-500/15 text-sky-400"
              aria-hidden="true"
            >
              <svg
                className="h-6 w-6"
                fill="none"
                stroke="currentColor"
                strokeWidth="1.8"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  d="M9 12.75L11.25 15 15 9.75m-3-7.036A11.959 11.959 0 013.598 6 11.99 11.99 0 003 9.749c0 5.592 3.824 10.29 9 11.623 5.176-1.332 9-6.03 9-11.622 0-1.31-.21-2.571-.598-3.751h-.152c-3.196 0-6.1-1.248-8.25-3.285z"
                />
              </svg>
            </span>
            <div>
              <h1 className="flex flex-wrap items-center gap-2 text-lg font-bold leading-tight tracking-tight">
                PPE Detection
                <span className="rounded-md bg-sky-500/15 px-2 py-0.5 text-xs font-semibold text-sky-400">
                  YOLO11n
                </span>
              </h1>
              <p className="text-xs text-muted">
                Real-time Personal Protective Equipment monitoring on construction sites
              </p>
            </div>
          </div>

          <div className="flex flex-wrap items-center justify-end gap-2">
            <LatencyBadge serverMs={serverMs} clientMs={clientMs} busy={busy} />

            <span
              className={`inline-flex items-center gap-1.5 rounded-full border px-3 py-1.5 text-xs font-semibold ${
                loading
                  ? 'border-line bg-raised text-muted'
                  : apiOffline
                    ? 'border-amber-500/50 bg-amber-500/10 text-amber-500'
                    : 'border-emerald-500/50 bg-emerald-500/10 text-emerald-500'
              }`}
              role="status"
            >
              <span
                className={`h-2 w-2 rounded-full ${
                  loading ? 'animate-ping bg-sky-400' : apiOffline ? 'bg-amber-500' : 'bg-emerald-500'
                }`}
                aria-hidden="true"
              />
              {statusLabel}
            </span>

            <button
              type="button"
              onClick={() => setTheme(theme === 'dark' ? 'light' : 'dark')}
              className="rounded-xl border border-line bg-raised p-2 text-muted transition hover:text-ink"
              aria-label={theme === 'dark' ? 'Switch to light theme' : 'Switch to dark theme'}
              title={theme === 'dark' ? 'Light theme' : 'Dark theme'}
            >
              {theme === 'dark' ? (
                <svg
                  className="h-4 w-4"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2"
                  viewBox="0 0 24 24"
                  aria-hidden="true"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    d="M12 3v2.25m6.364.386l-1.591 1.591M21 12h-2.25m-.386 6.364l-1.591-1.591M12 18.75V21m-4.773-4.227l-1.591 1.591M5.25 12H3m4.227-4.773L5.636 5.636M15.75 12a3.75 3.75 0 11-7.5 0 3.75 3.75 0 017.5 0z"
                  />
                </svg>
              ) : (
                <svg
                  className="h-4 w-4"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2"
                  viewBox="0 0 24 24"
                  aria-hidden="true"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    d="M21.752 15.002A9.718 9.718 0 0118 15.75c-5.385 0-9.75-4.365-9.75-9.75 0-1.33.266-2.597.748-3.752A9.753 9.753 0 003 11.25C3 16.635 7.365 21 12.75 21a9.753 9.753 0 009.002-5.998z"
                  />
                </svg>
              )}
            </button>
          </div>
        </div>
      </header>

      {/* ---------------------------------------------------------- tabs */}
      <nav className="border-b border-line bg-panel" aria-label="Sections">
        <div className="mx-auto max-w-shell px-4 sm:px-6">
          <div
            className="flex w-fit gap-1 py-2"
            role="tablist"
            aria-label="Dashboard sections"
            onKeyDown={onTabKeyDown}
          >
            {TABS.map((item, index) => (
              <button
                key={item.id}
                ref={(element) => {
                  tabButtonsRef.current[index] = element;
                }}
                type="button"
                role="tab"
                aria-selected={tab === item.id}
                tabIndex={tab === item.id ? 0 : -1}
                onClick={() => setTab(item.id)}
                className={`rounded-lg px-4 py-2 text-sm font-semibold transition ${
                  tab === item.id
                    ? 'bg-sky-600 text-white'
                    : 'text-muted hover:bg-raised hover:text-ink'
                }`}
              >
                {item.label}
              </button>
            ))}
          </div>
        </div>
      </nav>

      {/* ---------------------------------------------------------- main */}
      <main className="mx-auto max-w-shell px-4 py-6 sm:px-6">
        {tab === 'model' ? (
          <ModelDashboard metadata={metadata} loading={loading} health={health} />
        ) : (
          <div className="grid gap-6 lg:grid-cols-[minmax(0,1fr)_minmax(0,1.2fr)]">
            {/* --------------------------------------------- inputs */}
            <section className="space-y-4" aria-label="Image input">
              <div
                role="tabpanel"
                aria-label={`${tab} input`}
                className="rounded-2xl border border-line bg-panel p-4"
              >
                {tab === 'upload' && <Dropzone onFile={handleUploadFile} disabled={busy} />}

                {tab === 'webcam' && (
                  <WebcamPanel
                    onFrame={handleWebcamFrame}
                    onError={handleWebcamError}
                    onClearError={handleWebcamClearError}
                    live={live}
                    onLiveChange={setLive}
                    detections={origin === 'webcam' ? (result?.detections ?? []) : []}
                    showLabels={showLabels}
                    hiddenClasses={hiddenClasses}
                    onFps={setFps}
                  />
                )}

                {tab === 'samples' && (
                  <SampleGallery onSelect={handleSample} disabled={busy} />
                )}
              </div>

              <DetectionControls
                conf={conf}
                iou={iou}
                onConfChange={setConf}
                onIouChange={setIou}
                onReset={handleReset}
                classes={metadata?.classes ?? []}
                hiddenClasses={hiddenClasses}
                onToggleClass={toggleClass}
                showLabels={showLabels}
                onToggleLabels={setShowLabels}
                onDownload={downloadPng}
                onCopyJson={copyJson}
                canExport={Boolean(showCanvas && result)}
                busy={busy && origin !== 'webcam'}
              />

              {error && (
                <div
                  role="alert"
                  className="rounded-xl border border-rose-500/50 bg-rose-500/10 p-4 text-sm text-ink"
                >
                  <p className="font-semibold text-rose-500">Request failed</p>
                  <p className="mt-1 text-muted">{error}</p>
                </div>
              )}
            </section>

            {/* --------------------------------------------- results */}
            <section className="space-y-4" aria-label="Detection results">
              <ComplianceBanner
                detections={result?.detections ?? []}
                history={violationHistory}
                onViolation={handleViolation}
              />

              {showCanvas ? (
                <div className="space-y-2">
                  <div className="flex items-baseline justify-between gap-2">
                    <p className="truncate text-sm font-semibold text-ink">
                      {fileName ?? 'Analysed image'}
                    </p>
                    <p className="shrink-0 text-xs text-faint">
                      {imageSize ? `${imageSize.width} x ${imageSize.height}` : ''}
                      {result ? ` - ${result.count} detection(s)` : ''}
                    </p>
                  </div>
                  <BoundingBoxCanvas
                    imageUrl={imageUrl}
                    detections={imageDetections}
                    imageSize={imageSize}
                    showLabels={showLabels}
                    hiddenClasses={hiddenClasses}
                    canvasRef={canvasRef}
                    onImageError={handleImageError}
                  />
                </div>
              ) : tab !== 'webcam' ? (
                <EmptyResult onBrowseSamples={browseSamples} />
              ) : (
                <div className="rounded-2xl border border-dashed border-line bg-panel p-6 text-center text-sm text-muted">
                  Live boxes are drawn directly on the webcam preview. Results, compliance and
                  metrics update below as frames are analysed.
                </div>
              )}

              <div className="rounded-2xl border border-line bg-panel p-4">
                <h2 className="mb-3 text-sm font-semibold text-ink">Confidence per class</h2>
                <ConfidenceBars
                  detections={result?.detections ?? []}
                  classes={metadata?.classes ?? []}
                />
              </div>

              <div className="rounded-2xl border border-line bg-panel p-4">
                <h2 className="mb-3 text-sm font-semibold text-ink">Session metrics</h2>
                <StatsPanel
                  result={result}
                  metadata={metadata}
                  health={health}
                  serverMs={serverMs}
                  clientMs={clientMs}
                  history={history}
                  fps={fps}
                />
              </div>
            </section>
          </div>
        )}
      </main>

      <footer className="border-t border-line px-6 py-4 text-center text-xs text-faint">
        Deep Learning Avance - Projet de module | Axe 3 : detection d'objets en temps reel
      </footer>

      <ToastStack toasts={toasts} onDismiss={dismissToast} />
    </div>
  );
}
