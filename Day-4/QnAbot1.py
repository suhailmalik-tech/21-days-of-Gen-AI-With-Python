from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
import streamlit as st

load_dotenv()

# Page config / UI
st.set_page_config(page_title="AI QnA Bot with Memory", page_icon="🤖")
st.title("🤖 AI QnA Bot with Memory")
st.markdown(
    "Remembering conversation history using Streamlit Session State & LCEL."
)

# Initialize Groq LLM Model
bot = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.7)

# 1. Initializing single unified session state for history
if "chat_history" not in st.session_state:
  st.session_state.chat_history = []  # Stores LangChain message objects

# 2. Creating Prompt Template
prompt_template = ChatPromptTemplate.from_messages([
    (
        "system",
        (
            "You are a helpful AI assistant. Accurately remember and use the"
            " context provided in the conversation history."
        ),
    ),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
])

# Create the LCEL runnable chain
chain = prompt_template | bot

# 3. Display prior messages from session state on every rerun
for message in st.session_state.chat_history:
  if isinstance(message, HumanMessage):
    with st.chat_message("user"):
      st.markdown(message.content)
  elif isinstance(message, AIMessage):
    with st.chat_message("assistant"):
      st.markdown(message.content)

# 4. Capture user input
query = st.chat_input("Ask Anything")
if query:
  # Display user message immediately in UI
  with st.chat_message("user"):
    st.markdown(query)

  # Get response from model using the chain
  with st.chat_message("assistant"):
    with st.spinner("Thinking..."):
      # Pass input and the full history list to the chain
      res = chain.invoke(
          {"input": query, "chat_history": st.session_state.chat_history}
      )

      final_text = res.content
      st.markdown(final_text)

  # 5. Append messages to session state as LangChain message classes
  st.session_state.chat_history.append(HumanMessage(content=query))
  st.session_state.chat_history.append(AIMessage(content=final_text))

 