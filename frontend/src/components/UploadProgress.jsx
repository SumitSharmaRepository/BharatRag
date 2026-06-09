import { useEffect, useState } from "react"

const STAGES = [
  { label: "Waking up server...",    pct: 10 },
  { label: "Uploading document...",  pct: 28 },
  { label: "Extracting text...",     pct: 50 },
  { label: "Creating embeddings...", pct: 72 },
  { label: "Indexing...",            pct: 88 },
]

// Cumulative ms from upload start to enter each stage after stage 0
// Spread across ~90s to match Render cold-start (30-60s) + embedding time
const ADVANCE_MS = [12000, 28000, 50000, 72000]

export default function UploadProgress({ uploadState }) {
  const [stageIdx, setStageIdx] = useState(0)

  useEffect(() => {
    if (uploadState !== "progress") return
    setStageIdx(0)
    let idx = 0
    const timers = ADVANCE_MS.map(ms =>
      setTimeout(() => setStageIdx(++idx), ms)
    )
    return () => timers.forEach(clearTimeout)
  }, [uploadState])

  const stage = STAGES[Math.min(stageIdx, STAGES.length - 1)]
  const pct   = uploadState === "done" ? 100 : stage.pct
  const label = uploadState === "done"   ? "Ready ✓"
              : uploadState === "failed" ? "Upload failed"
              : stage.label

  const barColor = uploadState === "failed" ? "bg-red-500"
                 : uploadState === "done"   ? "bg-green-500 dark:bg-green-400"
                 : "bg-brand"

  const textColor = uploadState === "failed" ? "text-red-500"
                  : uploadState === "done"   ? "text-green-600 dark:text-green-400"
                  : "text-brand"

  return (
    <div className="w-full space-y-2 py-1">
      <p className={`text-xs font-medium text-center leading-snug transition-colors duration-300 ${textColor}`}>
        {label}
      </p>
      <div className="w-full h-1.5 rounded-full overflow-hidden bg-slate-200 dark:bg-slate-700">
        <div
          className={`h-full rounded-full transition-all duration-700 ease-out ${barColor}`}
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  )
}
