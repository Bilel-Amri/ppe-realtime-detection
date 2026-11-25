/**
 * Small inline-SVG sparkline of the latency history (last N requests).
 * No chart library: just a polyline over a fixed viewBox, scaled with
 * `preserveAspectRatio="none"` so it stretches with its container.
 */
export default function Sparkline({ values = [], width = 120, height = 32, stroke = '#38bdf8' }) {
  if (values.length < 2) {
    return (
      <svg
        viewBox={`0 0 ${width} ${height}`}
        className="h-8 w-full"
        preserveAspectRatio="none"
        aria-hidden="true"
      >
        <line
          x1="0"
          y1={height / 2}
          x2={width}
          y2={height / 2}
          stroke="currentColor"
          strokeOpacity="0.25"
          strokeDasharray="3 3"
        />
      </svg>
    );
  }

  const min = Math.min(...values);
  const max = Math.max(...values);
  const range = max - min || 1;
  const stepX = width / (values.length - 1);
  const points = values
    .map((value, index) => {
      const x = index * stepX;
      const y = height - 2 - ((value - min) / range) * (height - 4);
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .join(' ');

  const lastX = (values.length - 1) * stepX;
  const lastY = height - 2 - ((values[values.length - 1] - min) / range) * (height - 4);

  return (
    <svg
      viewBox={`0 0 ${width} ${height}`}
      className="h-8 w-full"
      preserveAspectRatio="none"
      role="img"
      aria-label={`Latency trend: last ${values.length} requests, from ${min.toFixed(
        0,
      )} to ${max.toFixed(0)} milliseconds`}
    >
      <polyline
        points={points}
        fill="none"
        stroke={stroke}
        strokeWidth="1.5"
        strokeLinejoin="round"
        strokeLinecap="round"
        vectorEffect="non-scaling-stroke"
      />
      <circle cx={lastX} cy={lastY} r="2" fill={stroke} />
    </svg>
  );
}
