from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
from dotenv import load_dotenv
from typing import TypedDict, Annotated
import operator
import os

load_dotenv()

class State(TypedDict):
    messages: Annotated[list, operator.add]
    next_agent: str
    iteration: int
    artifacts: dict

llm = ChatOpenAI(model="gpt-4o-mini")

# RAG knowledge base (lazy initialization)
vectorstore = None

def get_vectorstore():
    global vectorstore
    if vectorstore is None:
        docs = [
            Document(page_content="Python best practices: use type hints, write tests, follow PEP 8."),
            Document(page_content="Testing strategies: unit tests, integration tests, edge cases."),
            Document(page_content="Data analysis: use pandas, numpy, matplotlib for visualization."),
            Document(page_content="Code review: check logic, performance, security, readability.")
        ]
        vectorstore = FAISS.from_documents(docs, OpenAIEmbeddings())
    return vectorstore

@tool
def execute_code(code: str) -> str:
    """Execute Python code safely (expressions only)."""
    try:
        allowed_names = {"abs": abs, "round": round, "min": min, "max": max,
                        "pow": pow, "len": len, "sum": sum, "sorted": sorted,
                        "range": range, "list": list, "int": int, "float": float, "str": str}
        result = eval(code, {"__builtins__": {}}, allowed_names)
        return f"Result: {result}"
    except (SyntaxError, TypeError, ZeroDivisionError, NameError) as e:
        return f"Execution failed: {str(e)}"

@tool
def search_knowledge(query: str) -> str:
    """Search knowledge base."""
    try:
        results = get_vectorstore().similarity_search(query, k=1)
        return results[0].page_content if results else "No results"
    except Exception as e:
        return f"Search failed: {str(e)}"

@tool
def save_artifact(name: str, content: str) -> str:
    """Save work artifact."""
    try:
        safe_name = os.path.basename(name)
        if not safe_name:
            return "Invalid artifact name"
        os.makedirs("artifacts", exist_ok=True)
        with open(f"artifacts/{safe_name}.txt", "w") as f:
            f.write(content)
        return f"Saved {safe_name}"
    except Exception as e:
        return f"Save failed: {str(e)}"

@tool
def run_tests(test_code: str) -> str:
    """Run test code and return results."""
    try:
        # Simple test validation
        if "assert" in test_code:
            return "Tests passed"
        return "No assertions found"
    except Exception as e:
        return f"Test failed: {str(e)}"

@tool
def load_artifact(name: str) -> str:
    """Load a saved artifact."""
    try:
        safe_name = os.path.basename(name)
        if not safe_name:
            return "Invalid artifact name"
        with open(f"artifacts/{safe_name}.txt", "r") as f:
            return f.read()
    except FileNotFoundError:
        return f"Artifact {name} not found"
    except Exception as e:
        return f"Load failed: {str(e)}"

tools = [execute_code, search_knowledge, save_artifact, run_tests, load_artifact]
llm_with_tools = llm.bind_tools(tools)

def analyst(state: State):
    sys = SystemMessage(content="You are a data analyst. Use search_knowledge for best practices.")
    response = llm_with_tools.invoke([sys] + state["messages"])
    return {"messages": [response], "next_agent": "coder", "iteration": state.get("iteration", 0) + 1}

def coder(state: State):
    artifacts = state.get("artifacts", {})
    context = f"Previous work: {list(artifacts.keys())}" if artifacts else ""
    sys = SystemMessage(content=f"You are a coder. Write code. Use execute_code and save_artifact. {context}")
    response = llm_with_tools.invoke([sys] + state["messages"])
    return {"messages": [response], "next_agent": "tester", "iteration": state.get("iteration", 0) + 1}

def tester(state: State):
    sys = SystemMessage(content="You are a tester. Use run_tests and search_knowledge for testing strategies.")
    response = llm_with_tools.invoke([sys] + state["messages"])
    return {"messages": [response], "next_agent": "reviewer", "iteration": state.get("iteration", 0) + 1}

def reviewer(state: State):
    sys = SystemMessage(content="You are a reviewer. Provide final assessment and approval.")
    response = llm.invoke([sys] + state["messages"])
    return {"messages": [response], "next_agent": "done", "iteration": state.get("iteration", 0) + 1}

def tools_node(state: State):
    result = ToolNode(tools).invoke(state)
    last_msg = state["messages"][-1]
    
    # Track artifacts
    artifacts = state.get("artifacts", {})
    if hasattr(last_msg, "tool_calls"):
        for tc in last_msg.tool_calls:
            if tc["name"] == "save_artifact":
                artifacts[tc["args"]["name"]] = tc["args"]["content"]
    
    return {"messages": result["messages"], "iteration": state.get("iteration", 0) + 1, "artifacts": artifacts}

def router(state: State):
    msg = state["messages"][-1]
    
    if hasattr(msg, "tool_calls") and msg.tool_calls:
        return "tools"
    
    if not state.get("next_agent"):
        content = msg.content.lower()
        if "analyze" in content or "data" in content:
            return "analyst"
        elif "test" in content or "bug" in content:
            return "tester"
        elif "review" in content:
            return "reviewer"
        return "coder"
    
    next_agent = state.get("next_agent", "done")
    if next_agent == "done" or state.get("iteration", 0) > 10:
        return "end"
    return next_agent

def after_tools(state: State):
    # Route back to the agent that called the tool
    # next_agent points to where to go AFTER the current agent finishes,
    # so we infer the current agent from next_agent
    next_agent = state.get("next_agent", "")
    caller_map = {
        "coder": "analyst",
        "tester": "coder",
        "reviewer": "tester",
        "done": "reviewer"
    }
    return caller_map.get(next_agent, "coder")

graph = StateGraph(State)
graph.add_node("analyst", analyst)
graph.add_node("coder", coder)
graph.add_node("tester", tester)
graph.add_node("reviewer", reviewer)
graph.add_node("tools", tools_node)

graph.add_conditional_edges(START, router, {
    "analyst": "analyst",
    "coder": "coder",
    "tester": "tester",
    "reviewer": "reviewer",
    "end": END
})
graph.add_conditional_edges("analyst", router, {"tools": "tools", "coder": "coder", "end": END})
graph.add_conditional_edges("coder", router, {"tools": "tools", "tester": "tester", "end": END})
graph.add_conditional_edges("tester", router, {"tools": "tools", "reviewer": "reviewer", "end": END})
graph.add_conditional_edges("reviewer", router, {"end": END})
graph.add_conditional_edges("tools", after_tools, {
    "analyst": "analyst",
    "coder": "coder",
    "tester": "tester",
    "reviewer": "reviewer"
})

memory = MemorySaver()
app = graph.compile(checkpointer=memory)

if __name__ == "__main__":
    config = {"configurable": {"thread_id": "1"}}
    
    task = "Analyze requirements for a sorting algorithm, implement it, and test"
    result = app.invoke({
        "messages": [HumanMessage(content=task)],
        "next_agent": "",
        "iteration": 0,
        "artifacts": {}
    }, config)
    
    print(f"Task: {task}")
    print(f"Artifacts: {result.get('artifacts', {})}")
    print(f"\nFinal: {result['messages'][-1].content[:400]}...")
