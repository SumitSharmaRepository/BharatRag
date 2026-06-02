import { useState, useRef } from "react"
import { uploadDocument } from "../api/bharatrag"
import UploadProgress from "./UploadProgress"

const AGENT_COLORS = {
  TechAgent:      "bg-teal-100 text-teal-800 dark:bg-teal-900 dark:text-teal-200",
  ResearchAgent:  "bg-purple-100 text-purple-800 dark:bg-purple-900 dark:text-purple-200",
  LogisticsAgent: "bg-orange-100 text-orange-800 dark:bg-orange-900 dark:text-orange-200",
  GeneralAgent:   "bg-slate-100 text-slate-700 dark:bg-slate-700 dark:text-slate-200",
  Cache:          "bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200",
  RAGPipeline:    "bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200",
}

export function AgentBadge({ agent }) {
  const color = AGENT_COLORS[agent] || AGENT_COLORS.GeneralAgent
  return (
    <span className={`
      text-xs font-medium px-2 py-0.5
      rounded-full ${color}
    `}>
      {agent}
    </span>
  )
}

export default function Sidebar({ documents, onUpload, onDelete, onDeleteAll }) {
  const [uploadState, setUploadState] = useState("idle") // "idle"|"progress"|"done"|"failed"
  const [dragOver,    setDragOver]    = useState(false)
  const [error,       setError]       = useState("")
  const inputRef = useRef()

  async function handleFile(file) {
    if (!file || !file.name.endsWith(".pdf")) {
      setError("Only PDF files accepted")
      return
    }
    setError("")
    if (inputRef.current) inputRef.current.value = ""
    setUploadState("progress")
    try {
      const result = await uploadDocument(file)
      setUploadState("done")
      setTimeout(() => {
        onUpload(result)
        setUploadState("idle")
      }, 1500)
    } catch (e) {
      setUploadState("failed")
      setTimeout(() => {
        setError(e.message)
        setUploadState("idle")
      }, 1500)
    }
  }

  function onDrop(e) {
    e.preventDefault()
    setDragOver(false)
    if (uploadState === "idle") handleFile(e.dataTransfer.files[0])
  }

  return (
    <aside className="
      w-64 shrink-0 flex flex-col gap-4 p-4
      bg-white dark:bg-slate-900
      border-r border-slate-200 dark:border-slate-800
      transition-colors duration-200
    ">
      {/* Section title */}
      <div>
        <p className="
          text-xs font-semibold uppercase
          tracking-wide mb-3
          text-slate-500 dark:text-slate-400
          flex items-center justify-between
        ">
          Documents
          {documents.length > 0 && (
            <span className="
              bg-brand text-white text-xs
              px-1.5 py-0.5 rounded-full
              font-normal normal-case
            ">
              {documents.length}
            </span>
          )}
        </p>

        {documents.length === 0 ? (
          <p className="text-xs italic
            text-slate-400 dark:text-slate-500">
            No documents yet
          </p>
        ) : (
          <ul className="flex flex-col gap-2">
            {documents.map(doc => (
              <li key={doc} className="
                flex items-center gap-2
                rounded-lg px-3 py-2
                bg-slate-50 dark:bg-slate-800
                group
              ">
                <span className="text-brand text-sm shrink-0">
                  &#128196;
                </span>
                <div className="flex-1 min-w-0">
                  <p className="
                    text-xs font-medium truncate
                    text-slate-700 dark:text-slate-200
                  ">
                    {doc}
                  </p>
                </div>
                <button
                  onClick={() => onDelete(doc)}
                  className="
                    opacity-0 group-hover:opacity-100
                    text-slate-400 hover:text-red-500
                    transition-all text-xs
                    shrink-0 px-1
                  "
                  title="Remove document"
                  aria-label={`Remove ${doc}`}
                >
                  ✕
                </button>
              </li>
            ))}
          </ul>
        )}

        {documents.length > 1 && (
          <button
            onClick={onDeleteAll}
            className="
              text-xs text-slate-400
              hover:text-red-500
              transition-colors mt-1
            "
          >
            Clear all documents
          </button>
        )}
      </div>

      {/* Upload zone */}
      <div
        onDrop={onDrop}
        onDragOver={e => { e.preventDefault(); uploadState === "idle" && setDragOver(true) }}
        onDragLeave={() => setDragOver(false)}
        onClick={() => uploadState === "idle" && inputRef.current?.click()}
        className={`
          border-2 border-dashed rounded-xl
          p-4 text-center
          transition-colors duration-150
          ${uploadState !== "idle"
            ? "cursor-not-allowed opacity-75"
            : "cursor-pointer"}
          ${dragOver
            ? "border-brand bg-brand-light dark:bg-brand/10"
            : "border-slate-200 dark:border-slate-700 hover:border-brand/50"}
        `}
      >
        <input
          ref={inputRef}
          type="file"
          accept=".pdf"
          className="hidden"
          onChange={e => handleFile(e.target.files[0])}
        />
        {uploadState !== "idle" ? (
          <UploadProgress uploadState={uploadState} />
        ) : (
          <>
            <p className="text-2xl mb-1">&#8679;</p>
            <p className="text-xs text-slate-500 dark:text-slate-400">
              Drop PDF or click
            </p>
          </>
        )}
      </div>

      {error && (
        <p className="text-xs text-red-500 px-1">{error}</p>
      )}

      {/* Stats */}
      <div className="mt-auto pt-4
        border-t border-slate-100 dark:border-slate-800">
        <p className="text-xs
          text-slate-400 dark:text-slate-500">
          {documents.length} document(s) indexed
        </p>
      </div>
    </aside>
  )
}