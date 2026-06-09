// ============================================
// Chat.jsx — with streaming responses
// ============================================
import { useState, useRef, useEffect, startTransition  } from "react"
import Message from "./Message"
import { useStream } from "../hooks/useStream"
import { getChatHistory, saveChatMessage, exportChatPDF, clearChatHistory } from "../api/bharatrag"

export default function Chat({ language, userId = "default_user" }) {
  const [messages, setMessages] = useState([
    {
      id:      "welcome",
      role:    "assistant",
      content: "🙏 Namaste! I am BharatRAG — your AI " +
               "document assistant.\n\n" +
               "Upload a PDF from the sidebar and ask " +
               "me anything in English, Hindi, " +
               "Hinglish, or Arabic.\n\n" +
               "Built with Claude AI + LangGraph.",
      agent:   null,
      sources: [],
      streaming: false,
    }
  ])
  const [input,     setInput]     = useState("")
  const [loading,   setLoading]   = useState(false)
  const [exporting, setExporting] = useState(false)
  const [clearing,  setClearing]  = useState(false)
  const bottomRef      = useRef()
  const stopRef        = useRef()
  const pendingContent = useRef("")
  const pendingSources = useRef([])
  const pendingAgent   = useRef(null)

  const { streamMessage } = useStream()

  // Load history on mount
  useEffect(() => {
    getChatHistory(userId).then(history => {
      if (!history.length) return
      const loaded = history.map(m => ({
        id:        m.id,
        role:      m.role,
        content:   m.content,
        sources:   m.sources || [],
        agent:     m.agent_used || null,
        timestamp: m.timestamp || "",
        streaming: false,
        loading:   false,
      }))
      setMessages(prev => {
        const existingIds = new Set(prev.map(m => m.id))
        const newMsgs = loaded.filter(m => !existingIds.has(m.id))
        return newMsgs.length ? [...prev, ...newMsgs] : prev
      })
    }).catch(() => {})
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  useEffect(() => {
    bottomRef.current?.scrollIntoView({
      behavior: "smooth"
    })
  }, [messages])

  async function send() {
    const q = input.trim()
    if (!q || loading) return

    const msgId     = Date.now()
    const streamId  = msgId + 1

    // Reset accumulators for this turn
    pendingContent.current = ""
    pendingSources.current = []
    pendingAgent.current   = null

    const now = new Date().toISOString()

    // Add user message + empty streaming assistant message
    setMessages(prev => [
      ...prev,
      {
        id:        msgId,
        role:      "user",
        content:   q,
        sources:   [],
        timestamp: now,
        streaming: false,
      },
      {
        id:        streamId,
        role:      "assistant",
        content:   "",
        sources:   [],
        agent:     null,
        timestamp: now,
        streaming: true,
        loading:   true,
      },
    ])
    setInput("")
    setLoading(true)

    // Start streaming
    const stop = streamMessage({
      question: q,
      language,
      userId,

      onChunk: (chunk) => {
        pendingContent.current += chunk
        startTransition(() => {
            setMessages(prev =>
              prev.map(msg =>
                msg.id === streamId
                  ? {
                      ...msg,
                      content: msg.content + chunk,
                      loading: false,
                    }
                  : msg
              )
            )
          })
        },

      onSources: (sources, agent) => {
        pendingSources.current = sources
        pendingAgent.current   = agent
        setMessages(prev =>
          prev.map(msg =>
            msg.id === streamId
              ? {
                  ...msg,
                  sources,
                  agent,
                  streaming: false,
                }
              : msg
          )
        )
      },

      onDone: () => {
        // Ensure streaming flag is cleared even if onSources was never fired
        setMessages(prev =>
          prev.map(msg =>
            msg.id === streamId
              ? { ...msg, streaming: false, loading: false }
              : msg
          )
        )
        setLoading(false)
        stopRef.current = null
        // Save both messages to history (fire-and-forget)
        saveChatMessage(userId, "user", q).catch(() => {})
        saveChatMessage(
          userId, "assistant",
          pendingContent.current,
          pendingSources.current,
          pendingAgent.current || "",
        ).catch(() => {})
      },

      onError: (err) => {
        setMessages(prev =>
          prev.map(msg =>
            msg.id === streamId
              ? {
                  ...msg,
                  content:   `Error: ${err}`,
                  streaming: false,
                  loading:   false,
                }
              : msg
          )
        )
        setLoading(false)
      },
    })

    stopRef.current = stop
  }

  async function handleExport() {
    setExporting(true)
    try {
      const exportable = messages
        .filter(m => !m.streaming && !m.loading && m.content)
        .map(m => ({
          role:       m.role,
          content:    m.content,
          sources:    m.sources || [],
          agent_used: m.agent   || "",
          timestamp:  m.timestamp || "",
        }))
      await exportChatPDF(userId, exportable)
    } catch (e) {
      console.error("Export failed:", e)
    } finally {
      setExporting(false)
    }
  }

  async function handleClear() {
    setClearing(true)
    try {
      await clearChatHistory(userId)
      setMessages([{
        id:      "welcome",
        role:    "assistant",
        content: "🙏 Namaste! I am BharatRAG — your AI " +
                 "document assistant.\n\n" +
                 "Upload a PDF from the sidebar and ask " +
                 "me anything in English, Hindi, " +
                 "Hinglish, or Arabic.\n\n" +
                 "Built with Claude AI + LangGraph.",
        agent:   null,
        sources: [],
        streaming: false,
      }])
    } catch (e) {
      console.error("Clear failed:", e)
    } finally {
      setClearing(false)
    }
  }

  function stopStream() {
    stopRef.current?.()
    setLoading(false)
  }

  function onKeyDown(e) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault()
      send()
    }
  }

  return (
    <div className="flex-1 flex flex-col min-h-0">

      {/* Export bar */}
      {messages.length > 1 && (
        <div className="
          px-4 py-2 flex justify-end
          border-b border-slate-100 dark:border-slate-800
          bg-white dark:bg-slate-900
          transition-colors duration-200
        ">
          <button
            onClick={handleClear}
            disabled={clearing || loading}
            className="
              flex items-center gap-1.5
              text-xs px-3 py-1.5 rounded-lg
              border border-slate-200 dark:border-slate-700
              text-slate-500 dark:text-slate-400
              hover:border-red-400 hover:text-red-500
              disabled:opacity-40 disabled:cursor-not-allowed
              transition-colors duration-150
            "
          >
            <svg xmlns="http://www.w3.org/2000/svg" className="w-3.5 h-3.5" viewBox="0 0 20 20" fill="currentColor">
              <path fillRule="evenodd" d="M9 2a1 1 0 00-.894.553L7.382 4H4a1 1 0 000 2v10a2 2 0 002 2h8a2 2 0 002-2V6a1 1 0 100-2h-3.382l-.724-1.447A1 1 0 0011 2H9zM7 8a1 1 0 012 0v6a1 1 0 11-2 0V8zm5-1a1 1 0 00-1 1v6a1 1 0 102 0V8a1 1 0 00-1-1z" clipRule="evenodd" />
            </svg>
            {clearing ? "Clearing…" : "Clear chat"}
          </button>

          <button
            onClick={handleExport}
            disabled={exporting || loading}
            className="
              flex items-center gap-1.5
              text-xs px-3 py-1.5 rounded-lg
              border border-slate-200 dark:border-slate-700
              text-slate-500 dark:text-slate-400
              hover:border-brand hover:text-brand
              disabled:opacity-40 disabled:cursor-not-allowed
              transition-colors duration-150
            "
          >
            <svg xmlns="http://www.w3.org/2000/svg" className="w-3.5 h-3.5" viewBox="0 0 20 20" fill="currentColor">
              <path fillRule="evenodd" d="M3 17a1 1 0 011-1h12a1 1 0 110 2H4a1 1 0 01-1-1zm3.293-7.707a1 1 0 011.414 0L9 10.586V3a1 1 0 112 0v7.586l1.293-1.293a1 1 0 111.414 1.414l-3 3a1 1 0 01-1.414 0l-3-3a1 1 0 010-1.414z" clipRule="evenodd" />
            </svg>
            {exporting ? "Exporting…" : "Export PDF"}
          </button>
        </div>
      )}

      {/* Messages */}
      <div className="
        flex-1 overflow-y-auto
        px-4 py-4 flex flex-col gap-3
        bg-white dark:bg-slate-900
        transition-colors duration-200
      ">
        {messages.map(msg => (
          <Message key={msg.id} msg={msg} />
        ))}
        <div ref={bottomRef} />
      </div>

      {/* Input */}
      <div className="
        px-4 py-3 flex gap-2 items-end
        bg-white dark:bg-slate-900
        border-t border-slate-200 dark:border-slate-800
        transition-colors duration-200
      ">
        <textarea
          value={input}
          onChange={e => setInput(e.target.value)}
          onKeyDown={onKeyDown}
          placeholder="Ask anything about your documents..."
          rows={1}
          disabled={loading}
          style={{ minHeight: "44px" }}
          className="
            flex-1 resize-none rounded-xl
            px-3 py-2.5 text-sm leading-relaxed
            border border-slate-200 dark:border-slate-700
            bg-slate-50 dark:bg-slate-800
            text-slate-800 dark:text-slate-100
            placeholder:text-slate-400
            dark:placeholder:text-slate-500
            focus:outline-none focus:ring-2
            focus:ring-brand/30
            disabled:opacity-50
            max-h-32 overflow-y-auto
            transition-colors duration-200
          "
        />

        {/* Send / Stop button */}
        {loading ? (
          <button
            onClick={stopStream}
            className="
              w-10 h-10 rounded-xl
              bg-red-500 hover:bg-red-600
              text-white flex items-center
              justify-center shrink-0
              transition-colors
            "
            aria-label="Stop streaming"
            title="Stop"
          >
            &#9632;
          </button>
        ) : (
          <button
            onClick={send}
            disabled={!input.trim()}
            className="
              w-10 h-10 rounded-xl bg-brand
              text-white flex items-center
              justify-center shrink-0
              hover:bg-brand-dark transition-colors
              disabled:opacity-40
              disabled:cursor-not-allowed
            "
            aria-label="Send"
          >
            &#8679;
          </button>
        )}
      </div>
    </div>
  )
}