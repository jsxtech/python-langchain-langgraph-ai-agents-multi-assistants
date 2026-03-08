from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
from typing import TypedDict, Annotated, Literal
import operator

class State(TypedDict):
    messages: Annotated[list, operator.add]
    team: str
    handoff_to: str
    completed_by: Annotated[list, operator.add]
    iteration: int

llm = ChatOpenAI(model="gpt-4o-mini")

# RAG knowledge base (lazy initialization)
vectorstore = None

def get_vectorstore():
    global vectorstore
    if vectorstore is None:
        docs = [
            Document(page_content="API design: Use REST principles, versioning, proper status codes."),
            Document(page_content="Testing: Unit tests, integration tests, load tests, security tests."),
            Document(page_content="DevOps: CI/CD pipelines, containerization, monitoring, auto-scaling."),
            Document(page_content="Security: OWASP Top 10, encryption, authentication, authorization."),
            Document(page_content="Frontend: Responsive design, accessibility, performance optimization.")
        ]
        vectorstore = FAISS.from_documents(docs, OpenAIEmbeddings())
    return vectorstore

@tool
def create_ticket(title: str, assignee: str) -> str:
    """Create a task ticket."""
    return f"Ticket created: {title} → {assignee}"

@tool
def run_tests(test_type: str) -> str:
    """Run automated tests."""
    return f"Tests passed: {test_type}"

@tool
def deploy_service(environment: str) -> str:
    """Deploy to environment."""
    return f"Deployed to {environment}"

@tool
def search_knowledge(query: str) -> str:
    """Search best practices knowledge base."""
    try:
        results = get_vectorstore().similarity_search(query, k=1)
        return results[0].page_content if results else "No results"
    except Exception as e:
        return f"Search failed: {str(e)}"

@tool
def handoff_task(agent: str, reason: str) -> str:
    """Hand off task to another agent."""
    return f"Handed off to {agent}: {reason}"

tools = [create_ticket, run_tests, deploy_service, search_knowledge, handoff_task]
llm_with_tools = llm.bind_tools(tools)

# Team 1: Development
def backend_dev(state: State):
    sys = SystemMessage(content="You are a backend developer. Use search_knowledge for best practices. Use handoff_task if you need QA or DevOps help.")
    response = llm_with_tools.invoke([sys] + state["messages"])
    return {"messages": [response], "team": "dev", "completed_by": ["backend_dev"], "iteration": 1}

def frontend_dev(state: State):
    sys = SystemMessage(content="You are a frontend developer. Use search_knowledge for design patterns. Use handoff_task if you need backend or QA help.")
    response = llm_with_tools.invoke([sys] + state["messages"])
    return {"messages": [response], "team": "dev", "completed_by": ["frontend_dev"], "iteration": 1}

def devops(state: State):
    sys = SystemMessage(content="You are a DevOps engineer. Use deploy_service and search_knowledge. Use handoff_task if you need QA validation.")
    response = llm_with_tools.invoke([sys] + state["messages"])
    return {"messages": [response], "team": "dev", "completed_by": ["devops"], "iteration": 1}

# Team 2: Research
def ml_researcher(state: State):
    sys = SystemMessage(content="You are an ML researcher. Use search_knowledge for algorithms. Use handoff_task if you need data scientist help.")
    response = llm_with_tools.invoke([sys] + state["messages"])
    return {"messages": [response], "team": "research", "completed_by": ["ml_researcher"], "iteration": 1}

def data_scientist(state: State):
    sys = SystemMessage(content="You are a data scientist. Use search_knowledge for analysis techniques. Use handoff_task if you need ML researcher help.")
    response = llm_with_tools.invoke([sys] + state["messages"])
    return {"messages": [response], "team": "research", "completed_by": ["data_scientist"], "iteration": 1}

# Team 3: Quality
def qa_engineer(state: State):
    sys = SystemMessage(content="You are a QA engineer. Use run_tests and search_knowledge. Use handoff_task if you find security issues.")
    response = llm_with_tools.invoke([sys] + state["messages"])
    return {"messages": [response], "team": "qa", "completed_by": ["qa_engineer"], "iteration": 1}

def security_expert(state: State):
    sys = SystemMessage(content="You are a security expert. Use search_knowledge for OWASP guidelines. Use handoff_task if you need dev team fixes.")
    response = llm_with_tools.invoke([sys] + state["messages"])
    return {"messages": [response], "team": "qa", "completed_by": ["security_expert"], "iteration": 1}

# Team 4: Management
def product_manager(state: State):
    sys = SystemMessage(content="You are a product manager. Use create_ticket and handoff_task to delegate work.")
    response = llm_with_tools.invoke([sys] + state["messages"])
    return {"messages": [response], "team": "management", "completed_by": ["product_manager"], "iteration": 1}

def tech_lead(state: State):
    sys = SystemMessage(content="You are a tech lead. Use search_knowledge for architecture patterns. Use handoff_task to delegate implementation.")
    response = llm_with_tools.invoke([sys] + state["messages"])
    return {"messages": [response], "team": "management", "completed_by": ["tech_lead"], "iteration": 1}

def tools_node(state: State):
    result = ToolNode(tools).invoke(state)
    last_msg = state["messages"][-1]
    
    # Check for handoff requests
    handoff_to = ""
    if hasattr(last_msg, "tool_calls"):
        for tc in last_msg.tool_calls:
            if tc["name"] == "handoff_task":
                handoff_to = tc["args"].get("agent", "")
    
    return {"messages": [result["messages"][-1]], "completed_by": ["tools"], "handoff_to": handoff_to, "iteration": 1}

def supervisor(state: State) -> Literal["backend_dev", "frontend_dev", "devops", "ml_researcher", 
                                         "data_scientist", "qa_engineer", "security_expert", 
                                         "product_manager", "tech_lead", "tools", "end"]:
    msg = state["messages"][-1]
    
    # Prevent infinite loops
    if state.get("iteration", 0) > 3:
        return "end"
    
    # Check for tool calls
    if hasattr(msg, "tool_calls") and msg.tool_calls:
        return "tools"
    
    content = msg.content.lower()
    
    # Management team (check first for high-level tasks)
    if "requirement" in content or "product" in content or "priority" in content or "roadmap" in content:
        return "product_manager"
    if "architecture" in content or "tech lead" in content or "technical decision" in content:
        return "tech_lead"
    
    # Quality team
    if "test" in content or "qa" in content or "quality" in content or "bug" in content:
        return "qa_engineer"
    if "security" in content or "vulnerability" in content or "penetration" in content or "exploit" in content:
        return "security_expert"
    
    # Research team
    if "machine learning" in content or "ml model" in content or "neural" in content or "train" in content:
        return "ml_researcher"
    if "data analysis" in content or "dataset" in content or "visualization" in content or "statistics" in content:
        return "data_scientist"
    
    # Development team
    if "api" in content or "backend" in content or "database" in content or "server" in content:
        return "backend_dev"
    if "ui" in content or "frontend" in content or "interface" in content or "react" in content or "css" in content:
        return "frontend_dev"
    if "deploy" in content or "infrastructure" in content or "devops" in content or "ci/cd" in content or "kubernetes" in content:
        return "devops"
    
    # Default to tech lead for ambiguous tasks
    return "tech_lead"

def after_tools(state: State):
    """Route to handoff target or back to original agent"""
    handoff = state.get("handoff_to", "")
    if handoff:
        agent_map = {
            "backend": "backend_dev",
            "frontend": "frontend_dev",
            "devops": "devops",
            "qa": "qa_engineer",
            "security": "security_expert",
            "ml": "ml_researcher",
            "data": "data_scientist",
            "pm": "product_manager",
            "lead": "tech_lead"
        }
        for key, agent in agent_map.items():
            if key in handoff.lower():
                return agent
    
    # Route back to last agent
    completed = state.get("completed_by", [])
    if "backend_dev" in completed:
        return "backend_dev"
    if "frontend_dev" in completed:
        return "frontend_dev"
    if "devops" in completed:
        return "devops"
    if "qa_engineer" in completed:
        return "qa_engineer"
    if "ml_researcher" in completed:
        return "ml_researcher"
    if "data_scientist" in completed:
        return "data_scientist"
    if "security_expert" in completed:
        return "security_expert"
    if "product_manager" in completed:
        return "product_manager"
    return "tech_lead"

# Build graph
graph = StateGraph(State)

# Add all agents
graph.add_node("backend_dev", backend_dev)
graph.add_node("frontend_dev", frontend_dev)
graph.add_node("devops", devops)
graph.add_node("ml_researcher", ml_researcher)
graph.add_node("data_scientist", data_scientist)
graph.add_node("qa_engineer", qa_engineer)
graph.add_node("security_expert", security_expert)
graph.add_node("product_manager", product_manager)
graph.add_node("tech_lead", tech_lead)
graph.add_node("tools", tools_node)

# Route from START
graph.add_conditional_edges(START, supervisor, {
    "backend_dev": "backend_dev",
    "frontend_dev": "frontend_dev",
    "devops": "devops",
    "ml_researcher": "ml_researcher",
    "data_scientist": "data_scientist",
    "qa_engineer": "qa_engineer",
    "security_expert": "security_expert",
    "product_manager": "product_manager",
    "tech_lead": "tech_lead",
    "end": END
})

# Agents with tools can route to tools node
for agent in ["backend_dev", "frontend_dev", "devops", "qa_engineer", "product_manager", 
              "ml_researcher", "data_scientist", "security_expert", "tech_lead"]:
    graph.add_conditional_edges(agent, supervisor, {"tools": "tools", "end": END})

# Tools route back to agent or handoff target
graph.add_conditional_edges("tools", after_tools, {
    "backend_dev": "backend_dev",
    "frontend_dev": "frontend_dev",
    "devops": "devops",
    "qa_engineer": "qa_engineer",
    "security_expert": "security_expert",
    "ml_researcher": "ml_researcher",
    "data_scientist": "data_scientist",
    "product_manager": "product_manager",
    "tech_lead": "tech_lead"
})

memory = MemorySaver()
app = graph.compile(checkpointer=memory)

if __name__ == "__main__":
    config = {"configurable": {"thread_id": "1"}}
    
    task = "Design a secure REST API with authentication and deploy it"
    result = app.invoke({
        "messages": [HumanMessage(content=task)],
        "team": "",
        "handoff_to": "",
        "completed_by": [],
        "iteration": 0
    }, config)
    
    print(f"Task: {task}")
    print(f"Team: {result['team']}")
    print(f"Workflow: {' → '.join(result['completed_by'])}")
    print(f"\nFinal: {result['messages'][-1].content[:300]}...")
