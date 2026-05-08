DAY 1:
→ 4 sentences compared manually
→ One similarity score per pair
→ No storage — data lost after script
→ No metadata — no source info
→ You wrote the similarity math yourself

DAY 2:
→ 7 chunks stored in real database
→ Search ALL chunks with one query
→ Data persists on disk permanently
→ Source citations: filename + page number
→ ChromaDB handles math automatically
→ Scores improved: 0.49 → 0.784

Day 3 with LangChain:
→ rag_chain.invoke(question)
→ One line does everything
→ Same result
→ This is what frameworks are for


The 3-Day Agent Progression

Day 6: retrieve → grade → retry same → fallback
       Basic self-correction

Day 7: retrieve → grade → REWRITE → retry new → fallback
       Intelligent self-correction (Reflexion)

Day 8: retrieve → grade → rewrite → retry
                → HALLUCINATION CHECK → answer
       Verified self-correction