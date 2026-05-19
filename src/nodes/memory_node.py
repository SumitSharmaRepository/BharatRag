# ============================================
# src/nodes/memory_node.py
# ============================================
# Saves Q&A to chat_history after successful answer.
# This is Type 2 in-session memory (Day 9).
#
# Runs AFTER hallucination check confirms grounded.
# Appends current Q&A to chat_history in state.
# Next question will have this context available.
# ============================================

from src.agents.state import RAGState


def memory_node(state: RAGState) -> dict:
    """
    Save current Q&A to in-session chat_history.

    Only runs on grounded answers.
    Keeps last 8 messages to avoid context overflow.

    Type 2 memory — survives within session only.
    For cross-session use Mem0 (src/memory/persistent.py)
    """
    question     = state["question"]
    answer       = state["answer"]
    chat_history = state.get("chat_history", [])

    updated_history = chat_history + [
        {"role": "user",      "content": question},
        {"role": "assistant", "content": answer},
    ]

    # Keep only last 8 messages
    # Older history matters less
    # Prevents prompt from growing too large
    updated_history = updated_history[-8:]

    print(
        f"  [memory_node] Saved to history "
        f"(total: {len(updated_history)} messages)"
    )

    return {"chat_history": updated_history}


    """
    curl -s -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is CRAG?", "language": "English"}' > /tmp/r1.json
echo "agent_used: $(python3 -c "import json; d=json.load(open('/tmp/r1.json')); print(d['agent_used'])")"
agent_used: RAGPipeline
(venv) sumit@LAPTOP-SUMITSHARMA:~/bharatrag$ curl -s -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is CRAG?", "language": "English"}' > /tmp/r2.json
echo "agent_used: $(python3 -c "import json; d=json.load(open('/tmp/r2.json')); print(d['agent_used'])")"
agent_used: Cache
(venv) sumit@LAPTOP-SUMITSHARMA:~/bharatrag$ curl -s http://localhost:8000/cache/stats
{"size":1,"hits":1,"misses":1,"hit_rate":"50.0%","top_cached":[{"question":"What is CRAG?","hits":1}]}(venv) sumit@(venv) sumit@LAPTOP-SUMITSHARMA:~/bharatrag$ curl -s -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "Tell me about CRAG", "language": "English"}' > /tmp/r3.json
python3 -c "
import json
d = json.load(open('/tmp/r3.json'))
print('agent_used:', d['agent_used'])
print('answer preview:', d['answer'][:80])
"
agent_used: RAGPipeline
answer preview: Based on the provided documents, here's what I found about CRAG:

**CRAG** is a 
(venv) sumit@LAPTOP-SUMITSHARMA:~/bharatrag$ curl -s http://localhost:8000/cache/stats
{"size":2,"hits":1,"misses":2,"hit_rate":"33.3%","top_cached":[{"question":"What is CRAG?","hits":1},{"question":"Tell me about CRAG","hits":0}]}(venv) sumit@LAPTOP-SUMITSHARMA:~/bharatrag$ 
    """