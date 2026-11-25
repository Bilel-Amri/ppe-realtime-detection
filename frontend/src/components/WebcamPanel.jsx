import { useCallback, useEffect, useRef, useState } from 'react';
import { classColorCss } from '../services/palette.js';

/**
 * Webcam panel with explicit lifecycle states:
 *
 *   idle -> requesting -> live -> (error | idle)
 *
 * Frames are captured from the <video> element and sent to the existing
 * `POST /predict` endpoint through `onFrame` (the parent owns inference
 * state). In live mode the loop is throttled: a new frame is only grabbed
 * after the previous request settled, with a minimum gap of `INTERVAL_MS`,
 * so the backend is never flooded.
 *
 * Detected boxes from the last response are drawn on a canvas overlay that
 * tracks the video's `object-fit: contain` rectangle, so boxes stay aligned
 * at any window size.
 */

const INTERVAL_MS = 600;
const BLACK_FRAME_LUMA = 12; // average 0-255 below this = effectively black
const BLACK_FRAME_STRIKES = 3;

export default function WebcamPanel({
  onFrame,
  onError,
  onClearError,
  live = false,
  onLiveChange,
  detections = [],
  showLabels = true,
  hiddenClasses = [],
  onFps,
}) {
  const videoRef = useRef(null);
  const frameCanvasRef = useRef(null);
  const overlayRef = useRef(null);
  const streamRef = useRef(null);
  const busyRef = useRef(false);
  const stopRef = useRef(false);
  const blackStrikesRef = useRef(0);

  const [state, setState] = useState('idle'); // idle | requesting | live | error
  const [message, setMessage] = useState(null);
  const [devices, setDevices] = useState([]);
  const [deviceId, setDeviceId] = useState(null);
  const [blackWarning, setBlackWarning] = useState(false);
  const [fps, setFps] = useState(null);

  const hiddenKey = hiddenClasses.join('\u0000');

  const stop = useCallback(() => {
    stopRef.current = true;
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }
    if (videoRef.current) videoRef.current.srcObject = null;
    setState('idle');
    setFps(null);
    onFps?.(null);
    onLiveChange?.(false);
  }, [onFps, onLiveChange]);

  // Unmount always releases the camera.
  useEffect(() => stop, [stop]);

  const start = useCallback(
    async (selectedDeviceId) => {
      setState('requesting');
      setMessage(null);
      setBlackWarning(false);
      blackStrikesRef.current = 0;
      onClearError?.();

      if (!navigator.mediaDevices?.getUserMedia) {
        setState('error');
        setMessage(
          'Camera access is not available in this browser context. ' +
            'The webcam requires HTTPS or localhost.',
        );
        return;
      }

      try {
        const constraints = {
          video: selectedDeviceId
            ? { deviceId: { exact: selectedDeviceId } }
            : { facingMode: 'environment', width: { ideal: 1280 }, height: { ideal: 720 } },
          audio: false,
        };
        const stream = await navigator.mediaDevices.getUserMedia(constraints);
        streamRef.current = stream;
        stopRef.current = false;

        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          await videoRef.current.play();
        }
        setState('live');

        const allDevices = await navigator.mediaDevices.enumerateDevices();
        const cameras = allDevices.filter((device) => device.kind === 'videoinput');
        setDevices(cameras);
        const current = stream.getVideoTracks()[0]?.getSettings()?.deviceId;
        if (current) setDeviceId(current);
      } catch (exception) {
        const name = exception?.name ?? '';
        const friendly =
          name === 'NotAllowedError' || name === 'SecurityError'
            ? 'Camera permission denied. Allow camera access in your browser settings and try again.'
            : name === 'NotFoundError' || name === 'OverconstrainedError'
              ? 'No camera was found on this device.'
              : name === 'NotReadableError'
                ? 'The camera is already in use by another application.'
                : `Cannot access the webcam: ${exception?.message ?? 'unknown error'}.`;
        setState('error');
        setMessage(friendly);
        onError?.(friendly);
      }
    },
    [onClearError, onError],
  );

  /** Grab the current video frame as a JPEG File, or null if unavailable. */
  const grabFrame = useCallback(async () => {
    const video = videoRef.current;
    const canvas = frameCanvasRef.current;
    if (!video || !canvas || !video.videoWidth) return null;

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const context = canvas.getContext('2d');
    if (!context) return null;
    context.drawImage(video, 0, 0, canvas.width, canvas.height);

    // Black-frame detection on a tiny downsample: cheap and reliable.
    try {
      const probe = document.createElement('canvas');
      probe.width = 16;
      probe.height = 16;
      const probeCtx = probe.getContext('2d');
      probeCtx.drawImage(video, 0, 0, 16, 16);
      const pixels = probeCtx.getImageData(0, 0, 16, 16).data;
      let sum = 0;
      for (let i = 0; i !== pixels.length; i += 4) {
        sum += (pixels[i] + pixels[i + 1] + pixels[i + 2]) / 3;
      }
      const luma = sum / (pixels.length / 4);
      if (BLACK_FRAME_LUMA > luma) {
        blackStrikesRef.current += 1;
        if (blackStrikesRef.current >= BLACK_FRAME_STRIKES) setBlackWarning(true);
      } else {
        blackStrikesRef.current = 0;
        setBlackWarning(false);
      }
    } catch {
      /* pixel probe is best-effort only */
    }

    return new Promise((resolve) => {
      canvas.toBlob(
        (blob) => {
          resolve(blob ? new File([blob], 'webcam-frame.jpg', { type: 'image/jpeg' }) : null);
        },
        'image/jpeg',
        0.92,
      );
    });
  }, []);

  const capture = useCallback(async () => {
    const file = await grabFrame();
    if (file) await onFrame?.(file);
  }, [grabFrame, onFrame]);

  // ------------------------------------------------- throttled live loop
  useEffect(() => {
    if (!live || state !== 'live') {
      setFps(null);
      onFps?.(null);
      return undefined;
    }

    let cancelled = false;
    const intervals = [];

    const tick = async () => {
      if (cancelled || stopRef.current) return;
      const startedAt = performance.now();
      if (!busyRef.current) {
        busyRef.current = true;
        try {
          const file = await grabFrame();
          if (file && !cancelled && !stopRef.current) {
            await onFrame?.(file);
            const now = performance.now();
            intervals.push(now - startedAt);
            if (intervals.length > 5) intervals.shift();
            if (intervals.length >= 2) {
              const avg = intervals.reduce((a, b) => a + b, 0) / intervals.length;
              const value = Math.min(30, 1000 / Math.max(avg, 1));
              setFps(value);
              onFps?.(value);
            }
          }
        } catch {
          /* parent reports inference errors itself */
        } finally {
          busyRef.current = false;
        }
      }
      if (!cancelled && !stopRef.current) {
        const elapsed = performance.now() - startedAt;
        const delay = Math.max(50, INTERVAL_MS - elapsed);
        window.setTimeout(tick, delay);
      }
    };

    const timer = window.setTimeout(tick, INTERVAL_MS);
    return () => {
      cancelled = true;
      window.clearTimeout(timer);
    };
  }, [live, state, grabFrame, onFrame, onFps]);

  // -------------------------------------------- overlay box drawing
  useEffect(() => {
    const canvas = overlayRef.current;
    const video = videoRef.current;
    if (!canvas) return undefined;

    const draw = () => {
      const context = canvas.getContext('2d');
      if (!context) return;
      const parent = canvas.parentElement;
      if (!parent) return;

      const rect = parent.getBoundingClientRect();
      const dpr = window.devicePixelRatio || 1;
      canvas.width = Math.max(1, Math.round(rect.width * dpr));
      canvas.height = Math.max(1, Math.round(rect.height * dpr));
      context.setTransform(dpr, 0, 0, dpr, 0, 0);
      context.clearRect(0, 0, rect.width, rect.height);

      if (state !== 'live' || !video || !video.videoWidth || detections.length === 0) return;

      // object-fit: contain rectangle of the video inside the container.
      const scale = Math.min(rect.width / video.videoWidth, rect.height / video.videoHeight);
      const offsetX = (rect.width - video.videoWidth * scale) / 2;
      const offsetY = (rect.height - video.videoHeight * scale) / 2;

      const hidden = new Set(hiddenKey.split('\u0000').filter(Boolean));
      const visible = detections.filter((detection) => !hidden.has(detection.label));

      const fontSize = 12;
      context.font = `600 ${fontSize}px ui-sans-serif, system-ui, sans-serif`;
      context.textBaseline = 'top';

      visible.forEach((detection) => {
        const bbox = detection.bbox;
        if (!bbox || typeof bbox.x1 !== 'number') return;
        const x1 = offsetX + bbox.x1 * scale;
        const y1 = offsetY + bbox.y1 * scale;
        const x2 = offsetX + bbox.x2 * scale;
        const y2 = offsetY + bbox.y2 * scale;
        const color = classColorCss(detection.class_id);

        context.lineWidth = 2;
        context.strokeStyle = color;
        context.strokeRect(x1, y1, x2 - x1, y2 - y1);

        if (!showLabels) return;
        const text = `${detection.label} ${(detection.confidence * 100).toFixed(0)}%`;
        const textWidth = context.measureText(text).width;
        const boxHeight = fontSize + 6;
        const labelTop = Math.max(0, y1 - boxHeight);
        context.fillStyle = color;
        context.fillRect(x1, labelTop, textWidth + 8, boxHeight);
        context.fillStyle = '#ffffff';
        context.fillText(text, x1 + 4, labelTop + 3);
      });
    };

    draw();

    const parent = canvas.parentElement;
    const observer = typeof ResizeObserver !== 'undefined' ? new ResizeObserver(draw) : null;
    if (parent && observer) observer.observe(parent);

    const onVideoResize = () => draw();
    video?.addEventListener('loadedmetadata', onVideoResize);

    return () => {
      observer?.disconnect();
      video?.removeEventListener('loadedmetadata', onVideoResize);
    };
  }, [detections, showLabels, hiddenKey, state, videoRef]);

  const toggleLive = () => onLiveChange?.(!live);

  return (
    <div className="space-y-3">
      {/* ------------------------------------------------------- preview */}
      <div className="relative aspect-video w-full overflow-hidden rounded-2xl border border-line bg-black">
        <video
          ref={videoRef}
          playsInline
          muted
          autoPlay
          className="absolute inset-0 h-full w-full object-contain"
          aria-label="Webcam preview"
        />
        <canvas
          ref={overlayRef}
          className="pointer-events-none absolute inset-0 h-full w-full"
          aria-hidden="true"
        />

        {state === 'idle' && (
          <div className="absolute inset-0 flex flex-col items-center justify-center gap-2 p-6 text-center">
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
                d="M15.75 10.5l4.72-4.72a.75.75 0 011.28.53v11.38a.75.75 0 01-1.28.53l-4.72-4.72M4.5 18.75h8.25a2.25 2.25 0 002.25-2.25v-9a2.25 2.25 0 00-2.25-2.25H4.5A2.25 2.25 0 002.25 7.5v9a2.25 2.25 0 002.25 2.25z"
              />
            </svg>
            <p className="text-sm text-muted">The webcam is off</p>
          </div>
        )}

        {state === 'requesting' && (
          <div
            className="absolute inset-0 flex flex-col items-center justify-center gap-3 bg-black/70 p-6 text-center"
            role="status"
          >
            <span className="h-8 w-8 animate-spin rounded-full border-2 border-sky-400 border-t-transparent" />
            <p className="text-sm text-white">Requesting camera permission...</p>
          </div>
        )}

        {state === 'error' && (
          <div className="absolute inset-0 flex flex-col items-center justify-center gap-3 bg-black/80 p-6 text-center">
            <svg
              className="h-9 w-9 text-rose-400"
              fill="none"
              stroke="currentColor"
              strokeWidth="1.5"
              viewBox="0 0 24 24"
              aria-hidden="true"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                d="M12 9v3.75m-9.303 3.376c-.866 1.5.217 3.374 1.948 3.374h14.71c1.73 0 2.813-1.874 1.948-3.374L13.949 3.378c-.866-1.5-3.032-1.5-3.898 0L2.697 16.126zM12 15.75h.007v.008H12v-.008z"
              />
            </svg>
            <p className="max-w-sm text-sm text-white">{message}</p>
            <button
              type="button"
              onClick={() => start(deviceId)}
              className="rounded-xl bg-sky-600 px-4 py-2 text-sm font-semibold text-white transition hover:bg-sky-500"
            >
              Try again
            </button>
          </div>
        )}

        {/* status chips */}
        {state === 'live' && (
          <div className="absolute left-2 top-2 flex gap-2">
            <span className="inline-flex items-center gap-1.5 rounded-full bg-black/70 px-2.5 py-1 text-[11px] font-semibold text-white">
              <span className="h-2 w-2 animate-pulse rounded-full bg-rose-500" aria-hidden="true" />
              LIVE
            </span>
            {fps !== null && (
              <span className="rounded-full bg-black/70 px-2.5 py-1 font-mono text-[11px] text-sky-300">
                {fps.toFixed(1)} FPS
              </span>
            )}
          </div>
        )}

        {blackWarning && state === 'live' && (
          <div
            className="absolute inset-x-2 bottom-2 rounded-lg bg-amber-500/90 px-3 py-1.5 text-center text-xs font-semibold text-black"
            role="alert"
          >
            The camera image looks black - check lens coverage or lighting.
          </div>
        )}

        <canvas ref={frameCanvasRef} className="hidden" />
      </div>

      {/* ------------------------------------------------- camera picker */}
      {devices.length > 1 && state === 'live' && (
        <select
          value={deviceId ?? ''}
          onChange={(event) => {
            setDeviceId(event.target.value);
            if (state === 'live') start(event.target.value);
          }}
          className="w-full rounded-lg border border-line bg-raised px-3 py-2 text-sm text-ink"
          aria-label="Camera selection"
        >
          {devices.map((device, index) => (
            <option key={device.deviceId} value={device.deviceId}>
              {device.label || `Camera ${index + 1}`}
            </option>
          ))}
        </select>
      )}

      {/* ------------------------------------------------------ controls */}
      <div className="flex flex-wrap gap-2">
        {state !== 'live' && state !== 'requesting' && (
          <button
            type="button"
            onClick={() => start(deviceId)}
            className="flex-1 rounded-xl bg-sky-600 px-4 py-2.5 text-sm font-semibold text-white transition hover:bg-sky-500"
          >
            {state === 'error' ? 'Enable the webcam' : 'Enable the webcam'}
          </button>
        )}

        {state === 'requesting' && (
          <button
            type="button"
            disabled
            className="flex-1 cursor-wait rounded-xl bg-raised px-4 py-2.5 text-sm font-semibold text-muted"
          >
            Requesting permission...
          </button>
        )}

        {state === 'live' && (
          <>
            <button
              type="button"
              onClick={capture}
              className="flex-1 rounded-xl bg-sky-600 px-4 py-2.5 text-sm font-semibold text-white transition hover:bg-sky-500"
            >
              Capture a frame
            </button>
            <button
              type="button"
              onClick={toggleLive}
              aria-pressed={live}
              className={`flex-1 rounded-xl px-4 py-2.5 text-sm font-semibold transition ${
                live
                  ? 'bg-rose-600 text-white hover:bg-rose-500'
                  : 'border border-line bg-raised text-ink hover:bg-panel'
              }`}
            >
              {live ? 'Stop live analysis' : 'Start live analysis'}
            </button>
            <button
              type="button"
              onClick={stop}
              className="rounded-xl border border-line px-4 py-2.5 text-sm text-muted transition hover:text-ink"
            >
              Turn off
            </button>
          </>
        )}
      </div>

      {live && state === 'live' && (
        <p className="text-xs text-muted">
          Live mode: one frame every {INTERVAL_MS} ms (requests are queued so the backend is never
          overloaded). Boxes are drawn on the video as results arrive.
        </p>
      )}
    </div>
  );
}
