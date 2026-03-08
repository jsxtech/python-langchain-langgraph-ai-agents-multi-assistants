# Python LangChain LangGraph AI Agents

Multi-agent systems with RAG, tools, memory, and handoffs.

## Setup

```bash
chmod +x setup.sh
./setup.sh
source venv/bin/activate
export OPENAI_API_KEY="your-key-here"
```

## Files Overview

### 1. agent.py (Single Agent)
Simple agent with tools and memory.
- **Tools**: search_knowledge, calculate, fetch_url, get_time, save_note
- **Features**: RAG, memory, error handling
- **Use case**: General purpose assistant

```bash
python agent.py
```

### 2. multi_agent.py (3 Agents)
Researcher → Writer → Reviewer workflow.
- **Agents**: Researcher (with tools), Writer, Reviewer
- **Tools**: search_knowledge, calculate
- **Features**: RAG, auto-handoffs, memory
- **Use case**: Content creation and research

```bash
python multi_agent.py
```

### 3. assistants.py (4 Agents)
Software development workflow.
- **Agents**: Analyst → Coder → Tester → Reviewer
- **Tools**: execute_code, search_knowledge, save_artifact, run_tests, load_artifact
- **Features**: RAG, file-based artifacts, memory
- **Use case**: Code development and testing

```bash
python assistants.py
```

### 4. simple_multi_ai.py (3 Agents)
Sequential thinking pipeline.
- **Agents**: Creative → Analytical → Practical
- **Tools**: search_methods, validate_idea, create_plan
- **Features**: RAG, sequential workflow, memory
- **Use case**: Idea evaluation and planning

```bash
python simple_multi_ai.py
```

### 5. multi_assistants.py (9 Agents)
Enterprise multi-team system.
- **Teams**: Development (3), Research (2), Quality (2), Management (2)
- **Agents**: Backend Dev, Frontend Dev, DevOps, ML Researcher, Data Scientist, QA Engineer, Security Expert, Product Manager, Tech Lead
- **Tools**: create_ticket, run_tests, deploy_service, search_knowledge, handoff_task
- **Features**: RAG, agent handoffs, team routing, memory
- **Use case**: Complex enterprise workflows

```bash
python multi_assistants.py
```

## Key Features

- **RAG (Retrieval-Augmented Generation)**: All agents have access to knowledge bases
- **Tools**: 15+ specialized tools across all implementations
- **Memory**: Conversation persistence via checkpointing
- **Error Handling**: Robust error handling with specific exceptions
- **Lazy Loading**: RAG vectorstores load on first use
- **Iteration Limits**: Prevents infinite loops
- **Agent Handoffs**: Agents can delegate to specialists
- **Workflow Tracking**: See which agents processed each task

## Architecture

All implementations use LangGraph for:
- State management with typed dictionaries
- Conditional routing between agents
- Tool execution nodes
- Memory checkpointing

## Artifacts

- `artifacts/` - Saved work from assistants.py
- `notes.txt` - Notes saved by agent.py

## Requirements

See `requirements.txt` for dependencies.
