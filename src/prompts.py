# ============================================
# src/prompts.py — Prompt templates
# Single responsibility: define how we talk to Claude
# ============================================
# CONCEPT: Keeping prompts separate from logic
# means you can improve prompts without
# touching the retrieval or chain code
# Java equivalent: keeping SQL in .xml files
# not hardcoded in Java methods
# ============================================

from langchain_core.prompts import ChatPromptTemplate

# ── Main RAG prompt ───────────────────────────────────
# {context}  = retrieved chunks formatted with citations
# {question} = user's question passed through unchanged
RAG_PROMPT = ChatPromptTemplate.from_template("""
You are a helpful document assistant for BharatRAG.

Use ONLY the following context to answer the question.
If the answer is not in the context, say clearly:
"I could not find this information in the document."

Always cite the page number your answer comes from.
Be concise and accurate.

Context:
{context}

Question: {question}

Answer:""")


# ── Hindi/Hinglish prompt ─────────────────────────────
HINDI_PROMPT = ChatPromptTemplate.from_template("""
Aap ek helpful document assistant hain BharatRAG ke liye.

Sirf neeche diye gaye context ke basis par jawab dein.
Agar jawab context mein nahi hai, clearly bolein:
"Yeh jaankari document mein nahi mili."

Page number zaroor batayein.

Context:
{context}

Sawaal: {question}

Jawab:""")


# ── Prompt selector ───────────────────────────────────
def get_prompt(language: str = "English"):
    """
    Return the right prompt based on language choice.

    Args:
        language: "English", "Hindi", or "Hinglish"

    Returns:
        ChatPromptTemplate for that language
    """
    if language in ["Hindi / हिंदी", "Hinglish"]:
        return HINDI_PROMPT
    return RAG_PROMPT