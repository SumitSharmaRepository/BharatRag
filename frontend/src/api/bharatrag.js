const BASE_URL = import.meta.env.VITE_API_URL || ""

export async function checkHealth() {
  const res = await fetch(`${BASE_URL}/health`)
  if (!res.ok) throw new Error("API not reachable")
  return res.json()
}

export async function listDocuments(userId) {
  const res = await fetch(
    `${BASE_URL}/documents?user_id=${encodeURIComponent(userId)}`
  )
  if (!res.ok) throw new Error("Failed to list documents")
  return res.json()  // { active: [...], archived: [...], total_chunks: N }
}

export async function uploadDocument(file, userId) {
  const form = new FormData()
  form.append("file", file)
  form.append("user_id", userId)
  const res = await fetch(`${BASE_URL}/upload`, {
    method: "POST",
    body:   form,
  })
  if (!res.ok) {
    const err = await res.json()
    throw new Error(err.detail || "Upload failed")
  }
  return res.json()
}

export async function queryDocuments(
  question, language = "English",
  userId = "default_user", docFilter = null
) {
  const res = await fetch(`${BASE_URL}/query`, {
    method:  "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      question,
      language,
      user_id:    userId,
      doc_filter: docFilter,
    }),
  })
  if (!res.ok) {
    const err = await res.json()
    throw new Error(err.detail || "Query failed")
  }
  return res.json()
}

export async function deleteDocument(filename, userId, mode = "archive") {
  const encoded = encodeURIComponent(filename)
  const res = await fetch(
    `${BASE_URL}/documents/${encoded}?user_id=${encodeURIComponent(userId)}&mode=${mode}`,
    { method: "DELETE" }
  )
  if (!res.ok) throw new Error("Delete failed")
  return res.json()
}

export async function restoreDocument(filename, userId) {
  const encoded = encodeURIComponent(filename)
  const res = await fetch(
    `${BASE_URL}/documents/${encoded}/restore?user_id=${encodeURIComponent(userId)}`,
    { method: "POST" }
  )
  if (!res.ok) throw new Error("Restore failed")
  return res.json()
}

export async function resetDatabase() {
  const res = await fetch(`${BASE_URL}/reset`, { method: "DELETE" })
  if (!res.ok) throw new Error("Reset failed")
  return res.json()
}

export async function getChatHistory(userId, limit = 50) {
  const res = await fetch(
    `${BASE_URL}/chat/history?user_id=${encodeURIComponent(userId)}&limit=${limit}`
  )
  if (!res.ok) throw new Error("Failed to fetch chat history")
  return res.json()
}

export async function saveChatMessage(userId, role, content, sources = [], agentUsed = "") {
  const res = await fetch(`${BASE_URL}/chat/save`, {
    method:  "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      user_id:    userId,
      role,
      content,
      sources,
      agent_used: agentUsed,
    }),
  })
  if (!res.ok) throw new Error("Failed to save message")
  return res.json()
}

export async function clearChatHistory(userId) {
  const res = await fetch(
    `${BASE_URL}/chat/clear?user_id=${encodeURIComponent(userId)}`,
    { method: "DELETE" }
  )
  if (!res.ok) throw new Error("Failed to clear chat history")
  return res.json()
}

export async function exportChatPDF(userId, messages) {
  const res = await fetch(`${BASE_URL}/chat/export`, {
    method:  "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ user_id: userId, messages }),
  })
  if (!res.ok) throw new Error("Export failed")
  const date = new Date().toISOString().slice(0, 10).replace(/-/g, "")
  const blob = await res.blob()
  const url  = URL.createObjectURL(blob)
  const a    = document.createElement("a")
  a.href     = url
  a.download = `BharatRAG_Chat_${date}.pdf`
  a.click()
  setTimeout(() => URL.revokeObjectURL(url), 1000)
}
