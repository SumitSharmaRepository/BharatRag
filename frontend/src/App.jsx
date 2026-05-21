// ============================================
// App.jsx — root component
// ============================================
import { useState, useEffect } from "react"
import Header from "./components/Header"
import Sidebar from "./components/Sidebar"
import Chat from "./components/Chat"
import ThemeToggle from "./components/ThemeToggle"
import { checkHealth, listDocuments, deleteDocument } from "./api/bharatrag"

export default function App() {
  const [language, setLanguage] = useState("English")
  const [apiStatus, setApiStatus] = useState("checking")
  const [documents, setDocuments] = useState([])
  const [dark, setDark] = useState(false)

  // Apply dark class to html element
  useEffect(() => {
    if (dark) {
      document.documentElement.classList.add("dark")
    } else {
      document.documentElement.classList.remove("dark")
    }
  }, [dark])

  useEffect(() => {
    async function init() {
      try {
        await checkHealth()
        setApiStatus("healthy")
        const res = await listDocuments()
        setDocuments(res.documents || [])
      } catch {
        setApiStatus("offline")
      }
    }
    init()
  }, [])

  function handleUpload(result) {
    setDocuments(prev =>
      prev.includes(result.filename)
        ? prev
        : [...prev, result.filename]
    )
  }

  async function handleDelete(filename) {
    try {
      await deleteDocument(filename)
      setDocuments(prev => prev.filter(d => d !== filename))
    } catch (e) {
      console.error("Delete failed:", e)
    }
  }

  function handleDeleteAll() {
    documents.forEach(doc => deleteDocument(doc))
    setDocuments([])
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
          onUpload={handleUpload}
          onDelete={handleDelete}
          onDeleteAll={handleDeleteAll}
        />
        <main className="
          flex-1 flex flex-col
          bg-white min-h-0
        ">
          <Chat language={language} />
        </main>
      </div>
    </div>
  )
}