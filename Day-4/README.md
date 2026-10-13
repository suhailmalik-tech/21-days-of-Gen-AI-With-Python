📝 What I Learned Today: Building Conversational Memory for AI Chatbots
Today, I explored how Large Language Models (LLMs) are stateless by design and dove deep into the different architectural patterns used to give AI chatbots memory across multiple turns.

1. Client-Side & Local State Management
Streamlit st.session_state: Learned how to persist chat message lists in local browser memory so that the UI can re-render conversation history on every script rerun.

LangChain Built-in Buffers: Explored legacy utilities like ConversationBufferMemory to automatically manage message queues.

2. Industry & Production Methods (Database-Backed Memory)
The Production Standard: Discovered why production systems avoid volatile local RAM and instead use Database-Backed History (such as SQLite or PostgreSQL) mapped to unique session_ids via LangChain's RunnableWithMessageHistory.

Persistence Benefits: Implemented a local SQLite (chat.db) backend, ensuring that user chat histories survive page refreshes, server reboots, and app crashes.

3. Advanced Agentic Architectures
LangGraph: Looked at how the industry is transitioning from linear chains to LangGraph (built by the LangChain team), which treats conversation states as state graphs with built-in checkpointers for advanced AI agents, multi-turn tool use, and fault tolerance.