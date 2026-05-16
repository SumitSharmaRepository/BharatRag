// ============================================
// Header — app title + language selector
// ============================================

const LANGUAGES = [
  "English",
  "Hindi / हिंदी",
  "Hinglish",
  "Arabic / عربي",
]

export default function Header({
  language, onLanguageChange, apiStatus, darkToggle
}) {
  return (
    <header className="
      bg-white border-b border-slate-200
      px-4 py-3 flex items-center
      justify-between sticky top-0 z-10
    ">
      <div className="flex items-center gap-3">
        <div className="
          w-8 h-8 rounded-lg bg-brand
          flex items-center justify-center
          text-white font-bold text-sm
        ">
          B
        </div>
        <div>
          <p className="font-semibold text-slate-800 text-sm leading-none">
            BharatRAG
          </p>
          <p className="text-xs text-slate-400 mt-0.5">
            AI document assistant
          </p>
        </div>
      </div>

      <div className="flex items-center gap-3">
        <select
          value={language}
          onChange={e => onLanguageChange(e.target.value)}
          className="
            text-xs border border-slate-200
            rounded-lg px-2 py-1.5
            bg-slate-50 text-slate-700
            focus:outline-none focus:ring-2
            focus:ring-brand/30
          "
        >
          {LANGUAGES.map(l => (
            <option key={l} value={l}>{l}</option>
          ))}
        </select>

        <div className="flex items-center gap-1.5">
          <div className={`
            w-2 h-2 rounded-full
            ${apiStatus === "healthy"
              ? "bg-green-400"
              : apiStatus === "checking"
              ? "bg-amber-400"
              : "bg-red-400"}
          `} />
          <span className="text-xs text-slate-400">
            {apiStatus === "healthy" ? "connected" : apiStatus}
          </span>
        </div>
        {darkToggle}
      </div>
    </header>
  )
}