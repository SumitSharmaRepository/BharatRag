import { useState, useRef, useEffect } from "react"
import Message from "./Message"
import { queryDocuments } from "../api/bharatrag"

export default function Chat({ language }) {
  const [messages, setMessages] = useState([
    {
      id:      "welcome",
      role:    "assistant",
      content: "Namaste! Upload a PDF and ask me anything. " +
               "I support English, Hindi, Hinglish, and Arabic.",
      agent:   null,
      sources: [],
    }
  ])
  const [input,   setInput]   = useState("")
  const [loading, setLoading] = useState(false)
  const bottomRef = useRef()

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" })
  }, [messages])

  async function send() {
    const q = input.trim()
    if (!q || loading) return

    const userId = "bharatrag_user"

    setMessages(prev => [
      ...prev,
      {
        id:      Date.now(),
        role:    "user",
        content: q,
      },
      {
        id:      Date.now() + 1,
        role:    "assistant",
        loading: true,
        content: "",
      },
    ])
    setInput("")
    setLoading(true)

    try {
      const res = await queryDocuments(q, language, userId)

      setMessages(prev => {
        const updated = [...prev]
        updated[updated.length - 1] = {
          id:      Date.now() + 2,
          role:    "assistant",
          content: res.answer,
          agent:   res.agent_used,
          sources: res.sources || [],
        }
        return updated
      })
    } catch (e) {
      setMessages(prev => {
        const updated = [...prev]
        updated[updated.length - 1] = {
          id:      Date.now() + 2,
          role:    "assistant",
          content: `Error: ${e.message}`,
          agent:   null,
          sources: [],
        }
        return updated
      })
    } finally {
      setLoading(false)
    }
  }

  function onKeyDown(e) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault()
      send()
    }
  }

  return (
    <div className="flex-1 flex flex-col min-h-0">

      {/* Messages list */}
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

      {/* Input bar */}
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
            placeholder:text-slate-400 dark:placeholder:text-slate-500
            focus:outline-none focus:ring-2 focus:ring-brand/30
            disabled:opacity-50 max-h-32 overflow-y-auto
            transition-colors duration-200
          "
        />
        <button
          onClick={send}
          disabled={!input.trim() || loading}
          aria-label="Send"
          className="
            w-10 h-10 rounded-xl bg-brand
            text-white flex items-center
            justify-center shrink-0
            hover:bg-brand-dark transition-colors
            disabled:opacity-40 disabled:cursor-not-allowed
          "
        >
          &#8679;
        </button>
      </div>
    </div>
  )
}