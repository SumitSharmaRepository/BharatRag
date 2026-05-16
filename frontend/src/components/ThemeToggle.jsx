export default function ThemeToggle({ dark, onToggle }) {
  return (
    <button
      onClick={onToggle}
      aria-label="Toggle dark mode"
      title={dark ? "Switch to light mode" : "Switch to dark mode"}
      className="
        w-8 h-8 rounded-lg
        flex items-center justify-center
        border border-slate-200 dark:border-slate-700
        bg-slate-50 dark:bg-slate-800
        text-slate-600 dark:text-slate-300
        hover:bg-slate-100 dark:hover:bg-slate-700
        transition-colors text-base
      "
    >
      {dark ? "☀️" : "🌙"}
    </button>
  )
}