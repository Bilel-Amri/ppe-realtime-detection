/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx}'],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        // Semantic design tokens: values come from CSS variables so that the
        // light and dark themes swap without touching the components.
        bg: 'rgb(var(--c-bg) / <alpha-value>)',
        panel: 'rgb(var(--c-panel) / <alpha-value>)',
        raised: 'rgb(var(--c-raised) / <alpha-value>)',
        line: 'rgb(var(--c-line) / <alpha-value>)',
        ink: 'rgb(var(--c-ink) / <alpha-value>)',
        muted: 'rgb(var(--c-muted) / <alpha-value>)',
        faint: 'rgb(var(--c-faint) / <alpha-value>)',
      },
      maxWidth: {
        shell: '1440px',
      },
      keyframes: {
        'toast-in': {
          from: { opacity: '0', transform: 'translateY(-8px)' },
          to: { opacity: '1', transform: 'translateY(0)' },
        },
        'compliance-flash': {
          '0%': { boxShadow: '0 0 0 0 rgb(244 63 94 / 0.65)' },
          '100%': { boxShadow: '0 0 0 16px rgb(244 63 94 / 0)' },
        },
      },
      animation: {
        'toast-in': 'toast-in 0.18s ease-out',
        'compliance-flash': 'compliance-flash 0.8s ease-out',
      },
    },
  },
  plugins: [],
};
