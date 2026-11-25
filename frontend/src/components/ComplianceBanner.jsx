import { useEffect, useRef, useState } from 'react';

/**
 * PPE compliance indicator.
 *
 * States:
 *   - red     : at least one violation class is detected
 *               (NO-Hardhat, NO-Safety Vest, NO-Mask)
 *   - green   : relevant PPE detections exist and none of them is a violation
 *   - grey    : no relevant detection at all - the scene is *not assessed*
 *               (an empty or machinery-only frame must never be called safe)
 *
 * A timestamped history of violation events is kept, and the banner briefly
 * flashes when a new violation appears. This is a model-based indication,
 * not a certified safety inspection.
 */

const VIOLATION_CLASSES = ['NO-Hardhat', 'NO-Safety Vest', 'NO-Mask'];

/** Classes that make a scene assessable for PPE compliance. */
const RELEVANT_CLASSES = [
  ...VIOLATION_CLASSES,
  'Hardhat',
  'Mask',
  'Safety Vest',
  'Person',
];

const HISTORY_LIMIT = 8;

function assess(detections) {
  const relevant = detections.filter((d) => RELEVANT_CLASSES.includes(d.label));
  const violations = detections.filter((d) => VIOLATION_CLASSES.includes(d.label));

  if (violations.length > 0) {
    const byType = new Map();
    violations.forEach((d) => byType.set(d.label, (byType.get(d.label) ?? 0) + 1));
    return {
      status: 'violation',
      relevantCount: relevant.length,
      violationCount: violations.length,
      violationTypes: Array.from(byType.entries()).sort((a, b) => b[1] - a[1]),
    };
  }
  if (relevant.length > 0) {
    return {
      status: 'compliant',
      relevantCount: relevant.length,
      violationCount: 0,
      violationTypes: [],
    };
  }
  return { status: 'unassessed', relevantCount: 0, violationCount: 0, violationTypes: [] };
}

function formatTime(date) {
  return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
}

const STYLES = {
  violation: {
    card: 'border-rose-500/60 bg-rose-500/10',
    icon: 'bg-rose-500 text-white',
    title: 'text-rose-200',
    body: 'text-rose-100/80',
    chip: 'bg-rose-500/20 text-rose-100 border-rose-500/40',
    svg: 'M12 9v4m0 4h.01M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z',
  },
  compliant: {
    card: 'border-emerald-500/50 bg-emerald-500/10',
    icon: 'bg-emerald-500 text-white',
    title: 'text-emerald-200',
    body: 'text-emerald-100/80',
    chip: 'bg-emerald-500/20 text-emerald-100 border-emerald-500/40',
    svg: 'M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z',
  },
  unassessed: {
    card: 'border-line bg-raised',
    icon: 'bg-faint text-white',
    title: 'text-ink',
    body: 'text-muted',
    chip: 'bg-panel text-muted border-line',
    svg: 'M8.228 9c.549-1.165 2.03-2 3.772-2 2.21 0 4 1.343 4 3 0 1.4-1.278 2.575-3.006 2.907-.542.104-.994.54-.994 1.093m0 3h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z',
  },
};

const TITLES = {
  violation: 'PPE violation detected',
  compliant: 'No violations detected',
  unassessed: 'Not assessed',
};

export default function ComplianceBanner({
  detections = [],
  history = [],
  onViolation = null,
  className = '',
}) {
  const assessment = assess(detections);
  const style = STYLES[assessment.status];
  const [flash, setFlash] = useState(false);
  const lastSignatureRef = useRef('');

  // Log a new history entry (and flash) when a violation appears for the
  // first time or when the set of violation types changes.
  useEffect(() => {
    if (assessment.status !== 'violation') {
      lastSignatureRef.current = '';
      return;
    }
    const signature = assessment.violationTypes.map(([type, n]) => `${type}:${n}`).join('|');
    if (signature === lastSignatureRef.current) return;
    lastSignatureRef.current = signature;
    onViolation?.({
      id: `${Date.now()}-${Math.random().toString(36).slice(2, 7)}`,
      at: Date.now(),
      violationCount: assessment.violationCount,
      types: assessment.violationTypes,
    });
    setFlash(true);
    const timer = window.setTimeout(() => setFlash(false), 900);
    return () => window.clearTimeout(timer);
  }, [assessment, onViolation]);

  return (
    <div className={`space-y-3 ${className}`}>
      <div
        role="status"
        aria-live="polite"
        className={`flex items-start gap-3 rounded-2xl border p-4 transition-colors ${style.card} ${
          flash ? 'animate-compliance-flash' : ''
        }`}
        data-testid="compliance-banner"
        data-status={assessment.status}
      >
        <span
          className={`flex h-9 w-9 shrink-0 items-center justify-center rounded-full ${style.icon}`}
          aria-hidden="true"
        >
          <svg
            className="h-5 w-5"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
            viewBox="0 0 24 24"
          >
            <path d={style.svg} />
          </svg>
        </span>

        <div className="min-w-0 flex-1">
          <p className={`text-sm font-bold ${style.title}`}>{TITLES[assessment.status]}</p>

          {assessment.status === 'violation' && (
            <>
              <p className={`mt-0.5 text-sm ${style.body}`}>
                {assessment.violationCount} violation
                {assessment.violationCount > 1 ? 's' : ''} across{' '}
                {assessment.violationTypes.length} categorie
                {assessment.violationTypes.length > 1 ? 's' : ''}
              </p>
              <ul className="mt-2 flex flex-wrap gap-1.5">
                {assessment.violationTypes.map(([type, count]) => (
                  <li
                    key={type}
                    className={`rounded-full border px-2 py-0.5 text-xs font-semibold ${style.chip}`}
                  >
                    {type} &times;{count}
                  </li>
                ))}
              </ul>
            </>
          )}

          {assessment.status === 'compliant' && (
            <p className={`mt-0.5 text-sm ${style.body}`}>
              {assessment.relevantCount} PPE-relevant detection
              {assessment.relevantCount > 1 ? 's' : ''}, none of them a violation class
            </p>
          )}

          {assessment.status === 'unassessed' && (
            <p className={`mt-0.5 text-sm ${style.body}`}>
              No PPE-relevant detection in this frame - compliance cannot be evaluated
            </p>
          )}

          <p className={`mt-2 text-[11px] ${style.body} opacity-70`}>
            Model-based indication, not a certified safety inspection.
          </p>
        </div>
      </div>

      {history.length > 0 && (
        <div className="rounded-2xl border border-line bg-panel p-3">
          <p className="mb-2 text-xs font-semibold uppercase tracking-wide text-faint">
            Violation history
          </p>
          <ul className="space-y-1.5">
            {history.slice(0, HISTORY_LIMIT).map((event) => (
              <li
                key={event.id}
                className="flex items-center justify-between gap-2 rounded-lg bg-raised px-2.5 py-1.5 text-xs"
              >
                <span className="truncate text-muted">
                  <time dateTime={new Date(event.at).toISOString()} className="font-mono">
                    {formatTime(new Date(event.at))}
                  </time>
                  <span className="mx-1.5 text-faint">|</span>
                  {event.types.map(([type, count]) => `${type} x${count}`).join(', ')}
                </span>
                <span className="shrink-0 rounded-full border border-rose-500/40 bg-rose-500/15 px-2 py-0.5 font-semibold text-rose-200">
                  {event.violationCount}
                </span>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}
