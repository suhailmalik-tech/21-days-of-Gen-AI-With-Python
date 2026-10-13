from dotenv import load_dotenv
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_google_genai import ChatGoogleGenerativeAI
import streamlit as st


load_dotenv()

st.set_page_config(page_title="Production AI Chatbot", page_icon="🤖")
st.title("🤖 Enterprise QnA Bot (SQLite Persistence)")
st.markdown(
    "This chatbot saves messages directly to a local SQLite database (`chat.db`)."
    " Even if you refresh the browser, your history stays!"
)

# 2. Manage a Unique Session ID per User Browser Tab using Streamlit Session State
if "session_id" not in st.session_state:
  import uuid

  st.session_state.session_id = str(uuid.uuid4())

# 3. Initialize Google Gemini Model
llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash", temperature=0.7)

# 4. Create Prompt Template with MessagesPlaceholder
prompt = ChatPromptTemplate.from_messages([
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

# Core chain
chain = prompt | llm


# 5. Define SQLite Session History Factory
def get_session_history(session_id: str):
  # Creates a local 'chat.db' file in your directory automatically
  return SQLChatMessageHistory(
      session_id=session_id, connection="sqlite:///chat.db"
  )


# 6. Wrap with RunnableWithMessageHistory
conversational_chatbot = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
)

# 7. Render existing chat history from SQLite on every rerun
# Because SQLite stores the history, we pull directly from the database to render UI bubbles!
current_history = get_session_history(st.session_state.session_id).messages

for message in current_history:
  # LangChain message types use .type ('human' or 'ai')
  role = "user" if message.type == "human" else "assistant"
  with st.chat_message(role):
    st.markdown(message.content)

# 8. Handle New User Input
query = st.chat_input("Ask anything...")
if query:
  # Display user message immediately in UI
  with st.chat_message("user"):
    st.markdown(query)

  # Generate AI response using the conversational chain
  with st.chat_message("assistant"):
    with st.spinner("Thinking..."):
      # Pass input and the session config dictionary
      response = conversational_chatbot.invoke(
          {"input": query},
          config={"configurable": {"session_id": st.session_state.session_id}},
      )
      final_text = response.content
      st.markdown(final_text)

  