import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

load_dotenv()

# ✅ Model set to current active version


bot = ChatGroq(model="openai/gpt-oss-20b", temperature=0.7)


st.title("🤖 AI QnA Bot with Memory")
st.markdown("Can remember your old messages because of streamlit session_state functionality!")

# 1. Initialize UI message history for Streamlit rendering
if "messages" not in st.session_state:
    st.session_state.messages = []

# 2. Initialize LangChain history object to pass context to the model
if "langchain_history" not in st.session_state:
    st.session_state.langchain_history = []

# 3. Create a Prompt Template that supports Conversation Memory
prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant. Remember and use the context provided in the conversation history."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}")
])

# Create the runnable chain
chain = prompt_template | bot

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Capture user input
query = st.chat_input("Ask Anything")
if query:
    # Display user message in UI
    with st.chat_message("user"):
        st.markdown(query)
    st.session_state.messages.append({"role": "user", "content": query})

    # Get response from Gemini model using the chain with history
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # Pass the input along with accumulated history tuple list
            res = chain.invoke({
                "input": query,
                "chat_history": st.session_state.langchain_history
            })
            
            # Robust extraction logic for text content
            if isinstance(res.content, list) and len(res.content) > 0:
                final_text = res.content[0].get('text', '')
            elif isinstance(res.content, dict):
                final_text = res.content.get('text', '')
            else:
                final_text = res.content
            
            st.markdown(final_text)

    # ✅ Update LangChain memory with clean roles (human vs assistant)
    st.session_state.langchain_history.append(("human", query))
    st.session_state.langchain_history.append(("assistant", final_text))

    # Store in UI history
    st.session_state.messages.append({"role": "assistant", "content": final_text})
    
    st.rerun()
