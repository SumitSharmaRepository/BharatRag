# ============================================
# DAY 15: RAGAS Evaluation
# ============================================
# RAGAS = Retrieval Augmented Generation
#         Assessment
#
# Measures 4 specific RAG quality metrics:
# 1. Faithfulness      - is answer grounded?
# 2. Answer Relevancy  - does answer address question?
# 3. Context Precision - are retrieved chunks relevant?
# 4. Context Recall    - did retrieval find everything?
#
# LangSmith tells you WHAT happened.
# RAGAS tells you HOW GOOD your RAG is.
# ============================================

import os
from dotenv import load_dotenv
load_dotenv()

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from langchain_anthropic import ChatAnthropic
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.messages import HumanMessage
from pinecone import Pinecone

# ── Setup ─────────────────────────────────────────────
ANTHROPIC_KEY    = os.getenv("ANTHROPIC_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX   = os.getenv("PINECONE_INDEX", "bharatrag")
EMBEDDING_MODEL  = "sentence-transformers/all-MiniLM-L6-v2"

print("Loading components...")

embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL
)

pc          = Pinecone(api_key=PINECONE_API_KEY)
vectorstore = PineconeVectorStore(
    index_name = PINECONE_INDEX,
    embedding  = embeddings,
)
retriever   = vectorstore.as_retriever(
    search_kwargs={"k": 3}
)

llm = ChatAnthropic(
    model             = "claude-sonnet-4-5",
    temperature       = 0,
    anthropic_api_key = ANTHROPIC_KEY,
)

print("Components ready")

# ============================================
# EVALUATION DATASET
# ============================================
# RAGAS needs 4 fields per example:
#
# question:          what you ask
# answer:            what your RAG returned
# contexts:          chunks retrieved (list of strings)
# ground_truth:      the correct answer
#
# We generate answer + contexts by running
# BharatRAG on each question.
# ground_truth = what we expect the correct answer to be.
# ============================================

# Ground truth Q&A pairs
# These are the "golden dataset" their roadmap mentions
EVAL_QUESTIONS = [
    {
        "question":     "What is session state in Streamlit?",
        "ground_truth": "Session state is st.session_state, "
                       "a dictionary that persists data across "
                       "reruns. Normal variables reset every "
                       "rerun but session state survives.",
    },
    {
        "question":     "What is the difference between V1 and V2?",
        "ground_truth": "V1 is a 50-line prototype with no error "
                       "handling or security. V2 adds proper error "
                       "handling, client API key input so clients "
                       "pay their own bills, and file validation.",
    },
    {
        "question":     "What is CRAG?",
        "ground_truth": "CRAG stands for Corrective Retrieval "
                       "Augmented Generation. It is a plug-and-play "
                       "framework that improves RAG performance by "
                       "correcting retrieval errors. It outperforms "
                       "standard RAG on multiple benchmarks.",
    },
    {
        "question":     "What does the grade_node do in BharatRAG?",
        "ground_truth": "grade_node checks if retrieved chunks are "
                       "relevant to the question. It asks Claude to "
                       "evaluate relevance and returns either "
                       "relevant or irrelevant.",
    },
    {
        "question":     "What is pdfplumber used for?",
        "ground_truth": "pdfplumber is a Python library for "
                       "extracting text from PDF files. It replaced "
                       "PyPDF2 in SmartDocs V3 for better and more "
                       "reliable text extraction.",
    },
    {
        "question":     "What is hallucination in RAG?",
        "ground_truth": "Hallucination is when the LLM generates "
                       "information not supported by the retrieved "
                       "context. BharatRAG uses LLM-as-judge "
                       "pattern to detect and prevent hallucinations.",
    },
    {
        "question":     "What is the chunk_size used in BharatRAG?",
        "ground_truth": "BharatRAG uses chunk_size of 500 characters "
                       "with chunk_overlap of 100 characters. This "
                       "was chosen after testing showed 500 gives "
                       "better retrieval precision than 1000.",
    },
    {
        "question":     "What advantages does CRAG have over Self-RAG?",
        "ground_truth": "CRAG does not require instruction tuning "
                       "with special critic tokens unlike Self-RAG. "
                       "CRAG can work with any LLM without "
                       "additional training requirements.",
    },
]


# ============================================
# RUN BHARATRAG ON EACH QUESTION
# ============================================

def run_bharatrag(question: str) -> dict:
    """
    Run the RAG pipeline for one question.
    Returns answer and retrieved contexts.

    RAGAS needs both to score quality.
    """
    # Retrieve chunks
    docs     = retriever.invoke(question)
    contexts = [doc.page_content for doc in docs]

    if not contexts:
        return {
            "answer":   "I could not find relevant information.",
            "contexts": [],
        }

    # Generate answer
    context_text = "\n\n".join(contexts)
    prompt       = f"""Answer using ONLY this context.
If not found say: "I could not find this in the documents."
Be concise and accurate.

Context: {context_text}

Question: {question}

Answer:"""

    response = llm.invoke([HumanMessage(content=prompt)])

    return {
        "answer":   response.content,
        "contexts": contexts,
    }


def build_eval_dataset() -> Dataset:
    """
    Run BharatRAG on all questions.
    Build HuggingFace Dataset for RAGAS.
    """
    print("\nBuilding evaluation dataset...")
    print(f"Running {len(EVAL_QUESTIONS)} questions...")
    print("-" * 50)

    questions     = []
    answers       = []
    contexts_list = []
    ground_truths = []

    for i, item in enumerate(EVAL_QUESTIONS):
        q  = item["question"]
        gt = item["ground_truth"]

        print(f"\n[{i+1}/{len(EVAL_QUESTIONS)}] {q[:50]}...")

        result = run_bharatrag(q)

        questions.append(q)
        answers.append(result["answer"])
        contexts_list.append(result["contexts"])
        ground_truths.append(gt)

        print(f"  Retrieved: {len(result['contexts'])} chunks")
        print(f"  Answer:    {result['answer'][:80]}...")

    # Build HuggingFace Dataset
    # RAGAS expects this specific format
    dataset = Dataset.from_dict({
        "question":     questions,
        "answer":       answers,
        "contexts":     contexts_list,
        "ground_truth": ground_truths,
    })

    print(f"\nDataset built: {len(dataset)} examples")
    return dataset


# ============================================
# RUN RAGAS EVALUATION
# ============================================

def run_ragas_evaluation(dataset: Dataset) -> dict:
    """
    Score the dataset with RAGAS metrics.

    Returns dict of metric scores.
    """
    print("\n" + "=" * 55)
    print("Running RAGAS evaluation...")
    print("This calls Claude for each metric — takes 3-5 mins")
    print("=" * 55)

    # RAGAS needs an LLM for scoring
    # Uses Claude to evaluate quality
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper

    ragas_llm        = LangchainLLMWrapper(llm)
    ragas_embeddings = LangchainEmbeddingsWrapper(embeddings)

    # Run evaluation
    results = evaluate(
        dataset    = dataset,
        metrics    = [
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ],
        llm        = ragas_llm,
        embeddings = ragas_embeddings,
    )

    return results


def print_results(results):
    """Print RAGAS scores in readable format."""
    print("\n" + "=" * 55)
    print("RAGAS Evaluation Results — BharatRAG v1")
    print("=" * 55)

    scores = {
        "Faithfulness":      results["faithfulness"],
        "Answer Relevancy":  results["answer_relevancy"],
        "Context Precision": results["context_precision"],
        "Context Recall":    results["context_recall"],
    }

    for metric, score in scores.items():
        bar    = "█" * int(score * 20)
        spaces = "░" * (20 - int(score * 20))
        grade  = "✅" if score >= 0.8 else \
                 "⚠️" if score >= 0.6 else "❌"
        print(f"{metric:<22} {grade} {score:.3f} "
              f"|{bar}{spaces}|")

    avg = sum(scores.values()) / len(scores)
    print("-" * 55)
    print(f"{'Average':<22}    {avg:.3f}")
    print("=" * 55)

    # Interpretation
    print("\nWhat your scores mean:")
    print()

    if scores["Faithfulness"] < 0.8:
        print("⚠️  Faithfulness < 0.8:")
        print("   Answers contain info not in retrieved chunks.")
        print("   Fix: stricter prompt, lower temperature.")
        print()

    if scores["Answer Relevancy"] < 0.8:
        print("⚠️  Answer Relevancy < 0.8:")
        print("   Answers drift from the question.")
        print("   Fix: add 'Be concise and direct' to prompt.")
        print()

    if scores["Context Precision"] < 0.8:
        print("⚠️  Context Precision < 0.8:")
        print("   Retrieval pulling irrelevant chunks.")
        print("   Fix: reduce k from 3 to 2, or use metadata filter.")
        print()

    if scores["Context Recall"] < 0.8:
        print("⚠️  Context Recall < 0.8:")
        print("   Retrieval missing important content.")
        print("   Fix: increase k from 3 to 5, or reduce chunk_size.")
        print()

    return scores


# ============================================
# MAIN
# ============================================

if __name__ == "__main__":
    print("=" * 55)
    print("BharatRAG — RAGAS Evaluation")
    print("=" * 55)

    # Step 1: Build dataset
    dataset = build_eval_dataset()

    # Step 2: Run RAGAS
    results = run_ragas_evaluation(dataset)

    # Step 3: Print results
    scores = print_results(results)

    # Step 4: Save results to file
    import json
    output = {
        "model":             "claude-sonnet-4-5",
        "chunk_size":        500,
        "k":                 3,
        "num_questions":     len(EVAL_QUESTIONS),
        "faithfulness":      scores["Faithfulness"],
        "answer_relevancy":  scores["Answer Relevancy"],
        "context_precision": scores["Context Precision"],
        "context_recall":    scores["Context Recall"],
    }

    with open("ragas_results_v1.json", "w") as f:
        json.dump(output, f, indent=2)

    print("\nResults saved to ragas_results_v1.json")
    print("Commit this file to track improvement over time.")