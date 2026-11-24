import { useCallback, useRef, useState } from 'react';

/**
 * Drag & drop zone.
 *
 * Three ways to provide an image:
 *   - drag & drop a file;
 *   - click to open the file picker;
 *   - paste (Ctrl+V) an image from the clipboard.
 */
export default function Dropzone({ onFile, disabled = false, accept = 'image/*' }) {
  const inputRef = useRef(null);
  const [isOver, setIsOver] = useState(false);

  const handleFiles = useCallback(
    (fileList) => {
      const file = fileList?.[0];
      if (file && file.type.startsWith('image/')) {
        onFile(file);
      }
    },
    [onFile],
  );

  const handleDrop = useCallback(
    (event) => {
      event.preventDefault();
      setIsOver(false);
      if (disabled) return;
      handleFiles(event.dataTransfer.files);
    },
    [disabled, handleFiles],
  );

  const handlePaste = useCallback(
    (event) => {
      if (disabled) return;
      const items = event.clipboardData?.items;
      if (!items) return;
      for (const item of items) {
        if (item.type.startsWith('image/')) {
          const file = item.getAsFile();
          if (file) {
            onFile(file);
            return;
          }
        }
      }
    },
    [disabled, onFile],
  );

  return (
    <div
      onDrop={handleDrop}
      onDragOver={(event) => {
        event.preventDefault();
        if (!disabled) setIsOver(true);
      }}
      onDragLeave={() => setIsOver(false)}
      onPaste={handlePaste}
      onClick={() => !disabled && inputRef.current?.click()}
      role="button"
      tabIndex={0}
      onKeyDown={(event) => {
        if (event.key === 'Enter' || event.key === ' ') {
          event.preventDefault();
          if (!disabled) inputRef.current?.click();
        }
      }}
      className={[
        'group relative flex min-h-[220px] cursor-pointer flex-col items-center justify-center gap-3',
        'rounded-2xl border-2 border-dashed p-8 text-center transition-all duration-200',
        isOver
          ? 'border-sky-400 bg-sky-500/10 scale-[1.01]'
          : 'border-slate-600 bg-slate-900/40 hover:border-slate-500 hover:bg-slate-900/70',
        disabled ? 'pointer-events-none opacity-50' : '',
      ].join(' ')}
    >
      <input
        ref={inputRef}
        type="file"
        accept={accept}
        className="hidden"
        onChange={(event) => {
          handleFiles(event.target.files);
          event.target.value = '';
        }}
      />

      <div className="rounded-full bg-slate-800 p-4 transition-transform duration-200 group-hover:scale-110">
        <svg
          className="h-7 w-7 text-sky-400"
          fill="none"
          stroke="currentColor"
          strokeWidth="1.8"
          viewBox="0 0 24 24"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5m-13.5-9L12 3m0 0l4.5 4.5M12 3v13.5"
          />
        </svg>
      </div>

      <div className="space-y-1">
        <p className="text-base font-semibold text-slate-100">
          Drag &amp; drop a construction site photo
        </p>
        <p className="text-sm text-slate-400">
          or <span className="text-sky-400 underline">browse your files</span> &mdash; you can
          also paste an image with Ctrl&nbsp;+&nbsp;V
        </p>
      </div>

      <p className="text-xs uppercase tracking-wide text-slate-500">
        JPEG &middot; PNG &middot; WEBP &mdash; max 15 MB
      </p>
    </div>
  );
}
