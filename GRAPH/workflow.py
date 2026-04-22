from langgraph.graph import StateGraph
from llm.generator import generate_answer, get_llm
from utils.confidence import calculate_confidence
from hitl.human import escalate_to_human
from config import CONFIDENCE_THRESHOLD

def build_graph(retriever):
    llm = get_llm()

    def process(state):
        docs = retriever.get_relevant_documents(state["query"])
        answer = generate_answer(llm, state["query"], docs)
        conf = calculate_confidence(docs)
        return {"query": state["query"], "answer": answer, "confidence": conf}

    def route(state):
        return "hitl" if state["confidence"] < CONFIDENCE_THRESHOLD else "output"

    def output(state):
        return {"final_answer": state["answer"], "source": "RAG"}

    def hitl(state):
        ans = escalate_to_human(state["query"])
        return {"final_answer": ans, "source": "HITL"}

    graph = StateGraph(dict)
    graph.add_node("process", process)
    graph.add_node("output", output)
    graph.add_node("hitl", hitl)

    graph.set_entry_point("process")
    graph.add_conditional_edges("process", route, {
        "output": "output",
        "hitl": "hitl"
    })

    return graph.compile()
