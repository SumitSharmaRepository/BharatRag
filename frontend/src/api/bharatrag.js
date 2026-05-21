// Use VITE_API_URL in production
// Fall back to localhost in development
const BASE_URL = import.meta.env.VITE_API_URL || ""

export async function checkHealth() {
  const res = await fetch(`${BASE_URL}/health`)
  if (!res.ok) throw new Error("API not reachable")
  return res.json()
}

export async function listDocuments() {
  const res = await fetch(`${BASE_URL}/documents`)
  if (!res.ok) throw new Error("Failed to list documents")
  return res.json()
}

export async function uploadDocument(file) {
  const form = new FormData()
  form.append("file", file)
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

export async function deleteDocument(filename) {
  const encoded = encodeURIComponent(filename)
  const res = await fetch(`${BASE_URL}/documents/${encoded}`, {
    method: "DELETE",
  })
  if (!res.ok) throw new Error("Delete failed")
  return res.json()
}

export async function resetDatabase() {
  const res = await fetch(`${BASE_URL}/reset`, {
    method: "DELETE",
  })
  if (!res.ok) throw new Error("Reset failed")
  return res.json()
}