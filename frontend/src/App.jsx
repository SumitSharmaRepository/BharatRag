// ============================================
// App.jsx — root component
// ============================================
import { useState, useEffect, useRef } from "react"
import Header from "./components/Header"
import Sidebar from "./components/Sidebar"
import Chat from "./components/Chat"
import ThemeToggle from "./components/ThemeToggle"
import DeleteModal from "./components/DeleteModal"
import {
  checkHealth,
  listDocuments,
  deleteDocument,
  restoreDocument,
} from "./api/bharatrag"

export default function App() {
  // ── User identity ──────────────────────────
  // UUID lives in a ref: survives re-renders, lost on tab close (no localStorage).
  const userIdRef = useRef(null)
  if (userIdRef.current === null) {
    userIdRef.current = crypto.randomUUID()
  }

  // ── State ──────────────────────────────────
  const [language,   setLanguage]   = useState("English")
  const [apiStatus,  setApiStatus]  = useState("checking")
  const [documents,  setDocuments]  = useState({ active: [], archived: [] })
  const [dark,       setDark]       = useState(false)
  const [deleteTarget, setDeleteTarget] = useState(null)  // doc name string | null

  // ── Dark mode ──────────────────────────────
  useEffect(() => {
    document.documentElement.classList.toggle("dark", dark)
  }, [dark])

  // ── Bootstrap ──────────────────────────────
  useEffect(() => {
    // Pre-wake the Render server while the user is still reading the UI
    const BASE_URL = import.meta.env.VITE_API_URL || ""
    fetch(`${BASE_URL}/health`).catch(() => {})

    async function init() {
      try {
        await checkHealth()
        setApiStatus("healthy")
        const res = await listDocuments(userIdRef.current)
        setDocuments({ active: res.active || [], archived: res.archived || [] })
      } catch {
        setApiStatus("offline")
      }
    }
    init()
  }, [])

  // ── Document helpers ───────────────────────
  async function refreshDocuments() {
    try {
      const res = await listDocuments(userIdRef.current)
      setDocuments({ active: res.active || [], archived: res.archived || [] })
    } catch (e) {
      console.error("Failed to refresh documents:", e)
    }
  }

  function handleUpload() {
    refreshDocuments()
  }

  // ── Delete modal handlers ──────────────────
  function handleDeleteRequest(doc) {
    setDeleteTarget(doc)
  }

  async function handleArchive(doc) {
    setDeleteTarget(null)
    try {
      await deleteDocument(doc, userIdRef.current, "archive")
      await refreshDocuments()
    } catch (e) {
      console.error("Archive failed:", e)
    }
  }

  async function handlePermanentDelete(doc) {
    setDeleteTarget(null)
    try {
      await deleteDocument(doc, userIdRef.current, "permanent")
      await refreshDocuments()
    } catch (e) {
      console.error("Permanent delete failed:", e)
    }
  }

  async function handleRestore(doc) {
    try {
      await restoreDocument(doc, userIdRef.current)
      await refreshDocuments()
    } catch (e) {
      console.error("Restore failed:", e)
    }
  }

  return (
    <div className="h-screen flex flex-col bg-slate-50">
      <Header
        language={language}
        onLanguageChange={setLanguage}
        apiStatus={apiStatus}
        darkToggle={
          <ThemeToggle
            dark={dark}
            onToggle={() => setDark(d => !d)}
          />
        }
      />
      <div className="flex flex-1 min-h-0">
        <Sidebar
          documents={documents}
          userId={userIdRef.current}
          onUpload={handleUpload}
          onDeleteRequest={handleDeleteRequest}
          onRestore={handleRestore}
        />
        <main className="flex-1 flex flex-col bg-white min-h-0">
          <Chat language={language} userId={userIdRef.current} />
        </main>
      </div>

      {deleteTarget && (
        <DeleteModal
          doc={deleteTarget}
          dark={dark}
          onArchive={() => handleArchive(deleteTarget)}
          onPermanent={() => handlePermanentDelete(deleteTarget)}
          onCancel={() => setDeleteTarget(null)}
        />
      )}
    </div>
  )
}
