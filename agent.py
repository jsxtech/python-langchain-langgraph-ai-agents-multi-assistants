from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langgraph.graph import StateGraph, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.tools import tool
import requests
import ast
from datetime import datetime

# RAG knowledge base (lazy initialization)
vectorstore = None

def get_vectorstore():
    global vectorstore
    if vectorstore is None:
        docs = [
            Document(page_content="Python: Use list comprehensions, type hints, and virtual environments."),
            Document(page_content="Web APIs: REST principles, proper HTTP methods, status codes, authentication."),
            Document(page_content="Math: Order of operations (PEMDAS), basic algebra, statistics.")
        ]
        vectorstore = FAISS.from_documents(docs, OpenAIEmbeddings())
    return vectorstore

@tool
def search_knowledge(query: str) -> str:
    """Search knowledge base for information."""
    try:
        results = get_vectorstore().similarity_search(query, k=2)
        return "\n".join([doc.page_content for doc in results])
    except Exception as e:
        return f"Search failed: {str(e)}"

@tool
def calculate(expression: str) -> str:
    """Calculate a mathematical expression safely."""
    try:
        result = ast.literal_eval(expression)
        return f"Result: {result}"
    except (ValueError, SyntaxError) as e:
        return f"Invalid expression: {str(e)}"

@tool
def fetch_url(url: str) -> str:
    """Fetch content from a URL."""
    if not url.startswith(("http://", "https://")):
        return "Invalid URL: must start with http:// or https://"
    try:
        response = requests.get(url, timeout=5, headers={"User-Agent": "Mozilla/5.0"})
        response.raise_for_status()
        return response.text[:500]
    except requests.RequestException as e:
        return f"Failed to fetch: {str(e)}"

@tool
def get_time() -> str:
    """Get current date and time."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

@tool
def save_note(content: str) -> str:
    """Save a note to memory."""
    try:
        with open("notes.txt", "a") as f:
            f.write(f"{datetime.now()}: {content}\n")
        return "Note saved successfully"
    except Exception as e:
        return f"Failed to save: {str(e)}"

# Initialize model with tools and memory
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
tools = [search_knowledge, calculate, fetch_url, get_time, save_note]
llm_with_tools = llm.bind_tools(tools)
memory = MemorySaver()

# Define agent with system prompt
def agent(state: MessagesState):
    messages = state["messages"]
    if not any(isinstance(m, SystemMessage) for m in messages):
        messages = [SystemMessage(content="You are a helpful AI assistant with tools. Use search_knowledge for facts.")] + messages
    return {"messages": [llm_with_tools.invoke(messages)]}

# Build graph with memory
graph = StateGraph(MessagesState)
graph.add_node("agent", agent)
graph.add_node("tools", ToolNode(tools))
graph.set_entry_point("agent")
graph.add_conditional_edges("agent", tools_condition)
graph.add_edge("tools", "agent")

app = graph.compile(checkpointer=memory)

# Run agent with conversation memory
if __name__ == "__main__":
    config = {"configurable": {"thread_id": "1"}}
    
    queries = [
        "What are Python best practices?",
        "Calculate 25 * 4 + 10",
        "What time is it?"
    ]
    
    for query in queries:
        response = app.invoke({"messages": [HumanMessage(content=query)]}, config)
        print(f"\nQuery: {query}")
        print(f"Agent: {response['messages'][-1].content}")
