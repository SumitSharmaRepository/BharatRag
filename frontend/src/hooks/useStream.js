// ============================================
// src/hooks/useStream.js
// ============================================
// Custom hook for streaming LLM responses.
//
// Uses browser's built-in EventSource API.
// No extra library needed.
//
// Usage:
// const { streamMessage } = useStream()
// await streamMessage(question, language, onChunk)
// ============================================

import { useRef } from "react"

//const BASE_URL = ""
// Empty = uses Vite proxy to localhost:8000

// Direct URL bypasses Vite proxy
// Vite buffers SSE responses — kills streaming
const STREAM_URL = "https://bharatrag-api.onrender.com"//"http://172.25.210.165:8000"
// ↑ Replace with your actual IP from hostname -I


export function useStream() {
  const sourceRef = useRef(null)

  function stopStream() {
    if (sourceRef.current) {
      sourceRef.current.close()
      sourceRef.current = null
    }
  }

  function streamMessage({
    question,
    language   = "English",
    userId     = "bharatrag_user",
    docFilter  = "",
    onChunk,    // called with each text chunk
    onSources,  // called with sources when done
    onDone,     // called when stream complete
    onError,    // called on error
  }) {
    // Close any existing stream
    stopStream()

    // Build query string
    // EventSource only supports GET + query params
    const params = new URLSearchParams({
      question,
      language,
      user_id:    userId,
      doc_filter: docFilter,
    })

    const url = `${STREAM_URL}/stream?${params}`

    // Open SSE connection
    const source = new EventSource(url)
    sourceRef.current = source

    source.onmessage = (event) => {
      // Stream complete signal
      if (event.data === "[DONE]") {
        source.close()
        sourceRef.current = null
        onDone?.()
        return
      }

      try {
        const data = JSON.parse(event.data)

        if (data.error) {
          source.close()
          onError?.(data.error)
          return
        }

        if (data.chunk) {
          // New text chunk — append to message
          onChunk?.(data.chunk)
        }

        if (data.done) {
          // Stream finished — update sources (may be empty)
          onSources?.(data.sources || [], data.agent_used || null)
        }

      } catch (e) {
        console.error("Stream parse error:", e)
      }
    }

    source.onerror = (e) => {
      source.close()
      sourceRef.current = null
      onError?.("Stream connection failed")
    }

    // Return stop function so component can cancel
    return stopStream
  }

  return { streamMessage, stopStream }
}