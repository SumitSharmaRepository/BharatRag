// ============================================
// Chat.jsx — with streaming responses
// ============================================
import { useState, useRef, useEffect, startTransition  } from "react"
import Message from "./Message"
import { useStream } from "../hooks/useStream"

export default function Chat({ language }) {
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
  const [input,    setInput]    = useState("")
  const [loading,  setLoading]  = useState(false)
  const bottomRef  = useRef()
  const stopRef    = useRef()

  const { streamMessage } = useStream()

  useEffect(() => {
    bottomRef.current?.scrollIntoView({
      behavior: "smooth"
    })
  }, [messages])

  async function send() {
    const q = input.trim()
    if (!q || loading) return

    const userId    = "bharatrag_user"
    const msgId     = Date.now()
    const streamId  = msgId + 1

    // Add user message + empty streaming assistant message
    setMessages(prev => [
      ...prev,
      {
        id:       msgId,
        role:     "user",
        content:  q,
        sources:  [],
        streaming: false,
      },
      {
        id:       streamId,
        role:     "assistant",
        content:  "",
        sources:  [],
        agent:    null,
        streaming: true,  // shows typing indicator
        loading:  true,
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
        // Append each chunk to the streaming message
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
        // Update sources + agent when stream ends
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
        setLoading(false)
        stopRef.current = null
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