import { useCallback, useEffect, useRef, useState } from 'react';

/**
 * Webcam panel: live preview, single capture and continuous (live) inference.
 *
 * The panel only produces image frames - the API calls are made by the parent
 * component, which owns the detection state.
 */
export default function WebcamPanel({ onFrame, onError, live = false, onLiveChange }) {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const streamRef = useRef(null);
  const timerRef = useRef(null);
  const [active, setActive] = useState(false);
  const [devices, setDevices] = useState([]);
  const [deviceId, setDeviceId] = useState(null);

  const stop = useCallback(() => {
    if (timerRef.current) {
      window.clearInterval(timerRef.current);
      timerRef.current = null;
    }
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }
    setActive(false);
  }, []);

  useEffect(() => stop, [stop]);

  const start = useCallback(
    async (selectedDeviceId) => {
      try {
        const constraints = {
          video: selectedDeviceId
            ? { deviceId: { exact: selectedDeviceId } }
            : { facingMode: 'environment', width: { ideal: 1280 }, height: { ideal: 720 } },
          audio: false,
        };
        const stream = await navigator.mediaDevices.getUserMedia(constraints);
        streamRef.current = stream;
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          await videoRef.current.play();
        }
        setActive(true);

        const allDevices = await navigator.mediaDevices.enumerateDevices();
        const cameras = allDevices.filter((device) => device.kind === 'videoinput');
        setDevices(cameras);
        const current = stream.getVideoTracks()[0]?.getSettings()?.deviceId;
        if (current) setDeviceId(current);
      } catch (exception) {
        setActive(false);
        onError?.(
          `Cannot access the webcam: ${exception?.message ?? 'permission denied'}. ` +
            'The browser only exposes the camera on HTTPS or on localhost.',
        );
      }
    },
    [onError],
  );

  /** Grab the current video frame as a JPEG blob. */
  const grabFrame = useCallback(async () => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas || !video.videoWidth) return null;

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const context = canvas.getContext('2d');
    context.drawImage(video, 0, 0, canvas.width, canvas.height);

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
    if (file) onFrame?.(file);
  }, [grabFrame, onFrame]);

  // Continuous inference: one frame every 600 ms.
  useEffect(() => {
    if (!live || !active) {
      if (timerRef.current) {
        window.clearInterval(timerRef.current);
        timerRef.current = null;
      }
      return undefined;
    }
    timerRef.current = window.setInterval(async () => {
      const file = await grabFrame();
      if (file) onFrame?.(file);
    }, 600);
    return () => {
      if (timerRef.current) {
        window.clearInterval(timerRef.current);
        timerRef.current = null;
      }
    };
  }, [live, active, grabFrame, onFrame]);

  const toggleLive = () => onLiveChange?.(!live);

  return (
    <div className="space-y-3">
      <div className="relative overflow-hidden rounded-2xl bg-slate-950">
        <video
          ref={videoRef}
          playsInline
          muted
          className="h-full w-full object-contain"
          aria-label="Webcam preview"
        />
        {!active && (
          <div className="absolute inset-0 flex flex-col items-center justify-center gap-2 p-6 text-center">
            <svg
              className="h-10 w-10 text-slate-600"
              fill="none"
              stroke="currentColor"
              strokeWidth="1.5"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                d="M15.75 10.5l4.72-4.72a.75.75 0 011.28.53v11.38a.75.75 0 01-1.28.53l-4.72-4.72M4.5 18.75h8.25a2.25 2.25 0 002.25-2.25v-9a2.25 2.25 0 00-2.25-2.25H4.5A2.25 2.25 0 002.25 7.5v9a2.25 2.25 0 002.25 2.25z"
              />
            </svg>
            <p className="text-sm text-slate-400">The webcam is off</p>
          </div>
        )}
        <canvas ref={canvasRef} className="hidden" />
      </div>

      {devices.length > 1 && (
        <select
          value={deviceId ?? ''}
          onChange={(event) => {
            setDeviceId(event.target.value);
            if (active) start(event.target.value);
          }}
          className="w-full rounded-lg border border-slate-700 bg-slate-900 px-3 py-2 text-sm text-slate-200"
          aria-label="Camera selection"
        >
          {devices.map((device, index) => (
            <option key={device.deviceId} value={device.deviceId}>
              {device.label || `Camera ${index + 1}`}
            </option>
          ))}
        </select>
      )}

      <div className="flex flex-wrap gap-2">
        {!active ? (
          <button
            type="button"
            onClick={() => start(deviceId)}
            className="flex-1 rounded-xl bg-sky-600 px-4 py-2.5 text-sm font-semibold text-white transition hover:bg-sky-500"
          >
            Enable the webcam
          </button>
        ) : (
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
              className={`flex-1 rounded-xl px-4 py-2.5 text-sm font-semibold transition ${
                live
                  ? 'bg-rose-600 text-white hover:bg-rose-500'
                  : 'border border-slate-600 bg-slate-800 text-slate-200 hover:bg-slate-700'
              }`}
            >
              {live ? 'Stop the live stream' : 'Start the live stream'}
            </button>
            <button
              type="button"
              onClick={stop}
              className="rounded-xl border border-slate-600 px-4 py-2.5 text-sm text-slate-300 transition hover:bg-slate-700"
            >
              Off
            </button>
          </>
        )}
      </div>

      {live && (
        <p className="text-xs text-rose-300">
          Live mode: one frame is analysed every 600 ms. Use the stop button to leave.
        </p>
      )}
    </div>
  );
}
