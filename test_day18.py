# ============================================
# test_day18.py — Day 18 integration test
# ============================================

import os
from dotenv import load_dotenv
load_dotenv()

from src.agents.graph import build_multi_agent

def run(question: str, agent,
        user_id:  str = "sumit_test",
        language: str = "English") -> dict:

    print(f"\nQuestion: '{question}'")
    print(f"Language: {language}")
    print("=" * 55)

    state = {
        "question":     question,
        "answer":       "",
        "agent_used":   "",
        "documents":    [],
        "chat_history": [],
        "user_facts":   "",
        "language":     language,
        "user_id":      user_id,
    }

    result = agent.invoke(state)

    print(f"\nAnswer [{result['agent_used']}]:")
    print(result["answer"][:300])
    print(f"\nRouted to: {result['agent_used']}")
    return result


if __name__ == "__main__":
    print("Building Day 18 multi-agent system...")
    agent = build_multi_agent(with_memory=True)
    print("Ready!\n")

    # Test 1 — should route to TechAgent
    run("What is session state in Streamlit?", agent)

    # Test 2 — should route to ResearchAgent
    run("What is CRAG and how does it improve RAG?", agent)

    # Test 3 — should route to LogisticsAgent
    run("What is the total amount on the invoice?", agent)

    # Test 4 — Hinglish language
    run("What is CRAG?", agent, language="Hinglish")

    # Test 5 — Arabic language
    run("What is session state?", agent,
        language="Arabic / عربي")

    # Test 6 — Memory builds up
    print("\n" + "=" * 55)
    print("Testing memory — tell agent your preference")
    print("=" * 55)

    run("I am a logistics company owner", agent,
        user_id="logistics_user")
    run("What are the charges on the document?", agent,
        user_id="logistics_user")
    # Second question should route to LogisticsAgent
    # because Mem0 remembered "logistics company owner"

    # Test comparison routing
run("Compare V1 and V2 of SmartDocs", agent)

# Test summary routing
run("Summarise the CRAG paper", agent)

# Test summary in Hindi
run("SmartDocs guide ka summary do", agent,
    language="Hinglish")