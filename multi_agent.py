from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool
from typing import TypedDict, Annotated
import operator
import ast

class AgentState(TypedDict):
    messages: Annotated[list, operator.add]
    agent_history: Annotated[list, operator.add]
    needs_review: bool
    iteration: int

llm = ChatOpenAI(model="gpt-4o-mini")

# RAG knowledge base (lazy initialization)
vectorstore = None

def get_vectorstore():
    global vectorstore
    if vectorstore is None:
        docs = [
            Document(page_content="Python async/await enables concurrent programming."),
            Document(page_content="LangGraph is a framework for building multi-agent systems."),
            Document(page_content="RAG combines retrieval with generation for better answers."),
            Document(page_content="Multi-agent systems: researcher gathers info, writer creates content, reviewer validates.")
        ]
        vectorstore = FAISS.from_documents(docs, OpenAIEmbeddings())
    return vectorstore

@tool
def search_knowledge(query: str) -> str:
    """Search the knowledge base for relevant information."""
    try:
        results = get_vectorstore().similarity_search(query, k=2)
        return "\n".join([doc.page_content for doc in results])
    except Exception as e:
        return f"Search failed: {str(e)}"

@tool
def calculate(expression: str) -> str:
    """Calculate a mathematical expression."""
    try:
        result = ast.literal_eval(expression)
        return f"Result: {result}"
    except (ValueError, SyntaxError) as e:
        return f"Invalid expression: {str(e)}"

tools = [search_knowledge, calculate]
llm_with_tools = llm.bind_tools(tools)

# Specialized agents with tools and RAG
def researcher(state: AgentState):
    prompt = SystemMessage(content="You are a researcher. Use search_knowledge tool for facts.")
    response = llm_with_tools.invoke([prompt] + state["messages"])
    return {
        "messages": [response],
        "agent_history": ["researcher"],
        "needs_review": True,
        "iteration": 1
    }

def writer(state: AgentState):
    prompt = SystemMessage(content="You are a writer. Create clear, engaging content.")
    response = llm.invoke([prompt] + state["messages"])
    return {
        "messages": [response],
        "agent_history": ["writer"],
        "needs_review": True,
        "iteration": 1
    }

def reviewer(state: AgentState):
    prompt = SystemMessage(content="You are a reviewer. Critique and improve. Be concise.")
    response = llm.invoke([prompt] + state["messages"])
    return {
        "messages": [response],
        "agent_history": ["reviewer"],
        "needs_review": False,
        "iteration": 1
    }

def tool_executor(state: AgentState):
    """Execute tools called by agents"""
    tool_msg = ToolNode(tools).invoke(state)["messages"][-1]
    return {"messages": [tool_msg], "agent_history": ["tools"], "iteration": 1}

def supervisor(state: AgentState):
    """Decides next agent or if task is complete"""
    last_msg = state["messages"][-1]
    history = state.get("agent_history", [])
    iteration = state.get("iteration", 0)
    
    # Prevent infinite loops
    if iteration > 4:
        return "end"
    
    # Check if tools need execution
    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
        return "tools"
    
    # Route initial request
    if not history:
        content = last_msg.content.lower()
        if "research" in content or "find" in content:
            return "researcher"
        elif "review" in content or "critique" in content:
            return "reviewer"
        return "writer"
    
    # Auto-review after researcher/writer
    if state.get("needs_review") and "reviewer" not in history:
        return "reviewer"
    
    return "end"

def after_tools(state: AgentState):
    """Route back to researcher after tool execution"""
    return "researcher"

# Build multi-agent graph with tools and handoffs
graph = StateGraph(AgentState)
graph.add_node("researcher", researcher)
graph.add_node("writer", writer)
graph.add_node("reviewer", reviewer)
graph.add_node("tools", tool_executor)

graph.add_conditional_edges(START, supervisor, {
    "researcher": "researcher",
    "writer": "writer",
    "reviewer": "reviewer",
    "end": END
})
graph.add_conditional_edges("researcher", supervisor, {
    "tools": "tools",
    "reviewer": "reviewer",
    "end": END
})
graph.add_conditional_edges("tools", after_tools, {
    "researcher": "researcher",
    "reviewer": "reviewer"
})
graph.add_conditional_edges("writer", supervisor, {
    "reviewer": "reviewer",
    "end": END
})
graph.add_edge("reviewer", END)

memory = MemorySaver()
app = graph.compile(checkpointer=memory)

if __name__ == "__main__":
    config = {"configurable": {"thread_id": "1"}}
    
    tasks = [
        "Research LangGraph features",
        "Calculate 150 * 3 + 50"
    ]
    
    for task in tasks:
        response = app.invoke({
            "messages": [HumanMessage(content=task)],
            "agent_history": [],
            "needs_review": False,
            "iteration": 0
        }, config)
        
        print(f"\nTask: {task}")
        print(f"Agent flow: {' → '.join(response['agent_history'])}")
        print(f"Output: {response['messages'][-1].content[:300]}...")
