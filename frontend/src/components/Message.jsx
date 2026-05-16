import ReactMarkdown from "react-markdown"
import { AgentBadge } from "./Sidebar"

export default function Message({ msg }) {
  const isUser = msg.role === "user"

  if (isUser) {
    return (
      <div className="flex justify-end">
        <div className="
          max-w-[75%] bg-brand text-white
          rounded-2xl rounded-br-sm
          px-4 py-2.5 text-sm leading-relaxed
        ">
          {msg.content}
        </div>
      </div>
    )
  }

  return (
    <div className="flex gap-2 items-start">
      <div className="
        w-7 h-7 rounded-full bg-brand-light
        flex items-center justify-center
        shrink-0 mt-0.5
      ">
        <span className="text-brand text-xs font-bold">B</span>
      </div>

      <div className="flex-1">
        {msg.loading ? (
          <div className="
            bg-slate-100 dark:bg-slate-800
            rounded-2xl rounded-tl-sm
            px-4 py-3 inline-flex gap-1
          ">
            {[0,1,2].map(i => (
              <span
                key={i}
                className="
                  w-1.5 h-1.5 bg-slate-400
                  rounded-full animate-bounce
                "
                style={{ animationDelay: `${i * 0.15}s` }}
              />
            ))}
          </div>
        ) : (
          <>
            <div className="
              bg-slate-100 dark:bg-slate-800
              rounded-2xl rounded-tl-sm
              px-4 py-2.5 text-sm leading-relaxed
              text-slate-800 dark:text-slate-100
              prose prose-sm max-w-none
              prose-headings:text-slate-800
              dark:prose-headings:text-slate-100
              prose-strong:text-slate-800
              dark:prose-strong:text-slate-100
              prose-code:text-brand
              prose-code:bg-white
              dark:prose-code:bg-slate-900
              prose-code:px-1 prose-code:rounded
            ">
              <ReactMarkdown>
                {msg.content}
              </ReactMarkdown>
            </div>

            {(msg.agent || msg.sources?.length > 0) && (
              <div className="
                flex items-center gap-2
                mt-1.5 flex-wrap
              ">
                {msg.agent && (
                  <AgentBadge agent={msg.agent} />
                )}
                {msg.sources?.slice(0, 3).map((s, i) => (
                  <span
                    key={i}
                    className="text-xs text-slate-400"
                  >
                    {s}
                  </span>
                ))}
              </div>
            )}
          </>
        )}
      </div>
    </div>
  )
}