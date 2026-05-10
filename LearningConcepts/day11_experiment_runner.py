# ============================================
# DAY 11: Run Experiment Against Eval Dataset
# ============================================
# This runs your BharatRAG agent against
# all 10 questions in the dataset automatically.
# LangSmith scores each answer and shows results.
# ============================================

import os
from dotenv import load_dotenv
load_dotenv()

from langsmith import Client, evaluate
from langchain_anthropic import ChatAnthropic
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.messages import HumanMessage

# ── Setup ─────────────────────────────────────────────
CHROMA_PATH     = "/home/sumit/bharatrag/chroma_db"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
ANTHROPIC_KEY   = os.getenv("ANTHROPIC_API_KEY")
DATASET_NAME    = "BharatRAG Eval Set"

client = Client()

# ── Load retriever once ───────────────────────────────
print("Loading ChromaDB...")
embeddings  = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL
)
vectorstore = Chroma(
    embedding_function=embeddings,
    persist_directory=CHROMA_PATH,
)
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}
)
print(f"Loaded {vectorstore._collection.count()} chunks")

llm = ChatAnthropic(
    model="claude-sonnet-4-5",
    temperature=0,
    anthropic_api_key=ANTHROPIC_KEY
)

# ============================================
# THE FUNCTION LANGSMITH EVALUATES
# ============================================
# LangSmith calls this function for each
# example in your dataset.
#
# Input:  {"question": "What is session state?"}
# Output: {"answer": "st.session_state is..."}
#
# LangSmith compares output to expected answer
# and scores it automatically.
# ============================================

def bharatrag_pipeline(inputs: dict) -> dict:
    """
    Run BharatRAG for one question.
    Called by LangSmith for each dataset example.

    Args:
        inputs: {"question": "..."}

    Returns:
        {"answer": "..."}
    """
    question = inputs["question"]

    # Step 1: Retrieve relevant chunks
    docs      = retriever.invoke(question)
    doc_texts = []

    for doc in docs:
        page     = doc.metadata.get("page", "?")
        doc_name = doc.metadata.get(
            "doc_name", "unknown"
        )
        doc_texts.append(
            f"[{doc_name}, Page {int(page)+1}]\n"
            f"{doc.page_content}"
        )

    context = "\n\n".join(doc_texts)

    # Step 2: Generate answer
    prompt = f"""You are a helpful document assistant.

Answer using ONLY the provided context.
If not found say exactly:
"I could not find this information in the documents."
Be concise. Cite page numbers when possible.

Context:
{context}

Question: {question}

Answer:"""

    response = llm.invoke([HumanMessage(content=prompt)])

    return {"answer": response.content}


# ============================================
# EVALUATORS
# ============================================
# Evaluators score your answers automatically.
#
# 1. QA Evaluator (built-in LangSmith):
#    Compares agent answer to expected answer
#    Returns: correct / incorrect
#
# 2. Custom LLM-as-judge evaluator:
#    Asks Claude to score answer quality 1-10
#    More nuanced than correct/incorrect
# ============================================

def custom_llm_judge(
    run,        # the actual agent run
    example,    # the dataset example
) -> dict:
    """
    Custom evaluator: Claude scores the answer.
    Returns score 0-1 and reasoning.

    This is LLM-as-judge — Day 8 concept applied
    to systematic evaluation.
    """
    question        = example.inputs["question"]
    expected_answer = example.outputs["answer"]
    actual_answer   = run.outputs.get("answer", "")

    judge_prompt = f"""You are evaluating a RAG system answer.

Question: {question}

Expected answer (reference):
{expected_answer}

Actual answer (to evaluate):
{actual_answer}

Score the actual answer on these criteria:
1. Factual accuracy (does it match expected?)
2. Completeness (covers key points?)
3. Appropriate refusal (says "not found" when expected?)

Return ONLY a JSON object:
{{"score": 0.0 to 1.0, "reasoning": "one sentence"}}

Score guide:
1.0 = perfect match
0.8 = mostly correct, minor gaps
0.5 = partially correct
0.2 = mostly wrong
0.0 = completely wrong or hallucinated"""

    response = llm.invoke(
        [HumanMessage(content=judge_prompt)]
    )

    # Parse JSON response
    import json
    try:
        text   = response.content.strip()
        # Remove markdown code blocks if present
        text   = text.replace("```json", "").replace("```", "")
        result = json.loads(text)
        score  = float(result.get("score", 0.5))
        reason = result.get("reasoning", "No reasoning")
    except Exception as e:
        print(f"Parse error: {e}")
        score  = 0.5
        reason = "Could not parse evaluator response"

    print(f"  Score: {score:.2f} — {reason}")

    return {
        "key":   "llm_judge_score",
        "score": score,
        "comment": reason,
    }


# ============================================
# RUN THE EXPERIMENT
# ============================================

def run_experiment():
    print("\n" + "=" * 55)
    print("Running BharatRAG Experiment")
    print(f"Dataset: {DATASET_NAME}")
    print("=" * 55)

    results = evaluate(
        bharatrag_pipeline,      # your agent function
        data=DATASET_NAME,       # dataset to run against
        evaluators=[
            custom_llm_judge,    # custom LLM scorer
        ],
        experiment_prefix="bharatrag-v1",
        # Name shown in LangSmith dashboard
        metadata={
            "version":    "v1",
            "model":      "claude-sonnet-4-5",
            "chunk_size": 500,
            "k":          3,
        }
    )

    print("\n" + "=" * 55)
    print("Experiment complete!")
    print("View results at:")
    print("smith.langchain.com → Datasets → BharatRAG Eval Set → Experiments")
    print("=" * 55)

    return results


if __name__ == "__main__":
    run_experiment()