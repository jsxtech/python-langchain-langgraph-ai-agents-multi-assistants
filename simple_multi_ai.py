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

load_dotenv()

class State(TypedDict):
    messages: Annotated[list, operator.add]
    agent_chain: Annotated[list, operator.add]
    iteration: int
    feedback: str

llm = ChatOpenAI(model="gpt-4o-mini")

# RAG knowledge base (lazy initialization)
vectorstore = None

def get_vectorstore():
    global vectorstore
    if vectorstore is None:
        docs = [
            Document(page_content="Creative thinking: brainstorming, lateral thinking, SCAMPER technique."),
            Document(page_content="Critical analysis: SWOT, cost-benefit, risk assessment, feasibility study."),
            Document(page_content="Implementation: agile methodology, MVP, iterative development, testing.")
        ]
        vectorstore = FAISS.from_documents(docs, OpenAIEmbeddings())
    return vectorstore

@tool
def search_methods(query: str) -> str:
    """Search for methodologies and best practices."""
    try:
        results = get_vectorstore().similarity_search(query, k=1)
        return results[0].page_content if results else "No results"
    except Exception as e:
        return f"Search failed: {str(e)}"

@tool
def validate_idea(idea: str) -> str:
    """Validate idea feasibility."""
    score = len(idea.split()) % 10
    return f"Feasibility score: {score}/10"

@tool
def create_plan(task: str) -> str:
    """Create implementation plan."""
    return f"Plan: 1. Research 2. Design 3. Develop 4. Test 5. Deploy"

tools = [search_methods, validate_idea, create_plan]
llm_with_tools = llm.bind_tools(tools)

def creative_agent(state: State):
    sys = SystemMessage(content="You are creative. Use search_methods for techniques. Generate innovative ideas.")
    response = llm_with_tools.invoke([sys] + state["messages"])
    return {"messages": [response], "agent_chain": ["creative"], "iteration": state.get("iteration", 0) + 1}

def analytical_agent(state: State):
    sys = SystemMessage(content="You are analytical. Use validate_idea and search_methods. Evaluate critically.")
    response = llm_with_tools.invoke([sys] + state["messages"])
    return {"messages": [response], "agent_chain": ["analytical"], "iteration": state.get("iteration", 0) + 1}

def practical_agent(state: State):
    feedback = state.get("feedback", "")
    sys = SystemMessage(content=f"You are practical. Use create_plan and search_methods. Feedback: {feedback}")
    response = llm_with_tools.invoke([sys] + state["messages"])
    return {"messages": [response], "agent_chain": ["practical"], "iteration": state.get("iteration", 0) + 1}

def tools_node(state: State):
    result = ToolNode(tools).invoke(state)
    return {"messages": result["messages"], "iteration": state.get("iteration", 0) + 1}

def router(state: State):
    msg = state["messages"][-1]
    chain = state.get("agent_chain", [])
    iteration = state.get("iteration", 0)
    
    # Prevent infinite loops
    if iteration > 12:
        return "end"
    
    # Check for tool calls
    if hasattr(msg, "tool_calls") and msg.tool_calls:
        return "tools"
    
    # Sequential workflow: creative → analytical → practical
    if not chain:
        content = msg.content.lower()
        if "analyze" in content or "evaluate" in content:
            return "analytical"
        if "implement" in content or "practical" in content:
            return "practical"
        return "creative"
    
    if "creative" in chain and "analytical" not in chain:
        return "analytical"
    if "analytical" in chain and "practical" not in chain:
        return "practical"
    
    return "end"

def after_tools(state: State):
    chain = state.get("agent_chain", [])
    if "creative" in chain and "analytical" not in chain:
        return "creative"
    if "analytical" in chain and "practical" not in chain:
        return "analytical"
    return "practical"

graph = StateGraph(State)
graph.add_node("creative", creative_agent)
graph.add_node("analytical", analytical_agent)
graph.add_node("practical", practical_agent)
graph.add_node("tools", tools_node)

graph.add_conditional_edges(START, router, {
    "creative": "creative",
    "analytical": "analytical",
    "practical": "practical",
    "end": END
})
graph.add_conditional_edges("creative", router, {"tools": "tools", "analytical": "analytical", "end": END})
graph.add_conditional_edges("analytical", router, {"tools": "tools", "practical": "practical", "end": END})
graph.add_conditional_edges("practical", router, {"tools": "tools", "end": END})
graph.add_conditional_edges("tools", after_tools, {
    "creative": "creative",
    "analytical": "analytical",
    "practical": "practical"
})

memory = MemorySaver()
app = graph.compile(checkpointer=memory)

if __name__ == "__main__":
    config = {"configurable": {"thread_id": "1"}}
    
    task = "Design a new social media feature for content creators"
    result = app.invoke({
        "messages": [HumanMessage(content=task)],
        "agent_chain": [],
        "iteration": 0,
        "feedback": ""
    }, config)
    
    print(f"Task: {task}")
    print(f"Agent workflow: {' → '.join(result['agent_chain'])}")
    print(f"\nFinal: {result['messages'][-1].content[:400]}...")
