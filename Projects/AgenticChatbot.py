# Load and import Env for Api keys #
from dotenv import load_dotenv
load_dotenv()

#Import Every Required framework / Library #
from langchain_groq import ChatGroq
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver
import streamlit as st

#select model and enable Streaming for better strcutured Answers#
llm = ChatGroq(model="openai/gpt-oss-20b", streaming=True)
# make search a object model around GoogleSerperAPI #
search = GoogleSerperAPIWrapper()
#create tool for web surfing#
tools = [search.run]


# manage local session memory #
if "memory" not in st.session_state:
    st.session_state.memory = MemorySaver()
    st.session_state.history = []

# create agent specify the prompt #
agent = create_agent(
    model=llm,
    tools=tools,
    checkpointer=st.session_state.memory,
    system_prompt="You are an amazing ai agent and can search on google as well."

)

print(st.session_state.memory)


#User Interface using Streamlit#

st.subheader("⚡ Flash - Answers at Speed of Thought")

for message in st.session_state.history:
    role = message["role"]
    content = message["content"]
    st.chat_message(role).markdown(content)



query = st.chat_input("Ask Anything ?")
if query:
    st.chat_message("user").markdown(query)
    st.session_state.history.append({"role":"user", "content":query})


    response = agent.stream(
        {"messages":[{"role":"user", "content":query}]},
        {"configurable": {"thread_id": "1"}},
        stream_mode="messages"


    )



    ai_container = st.chat_message("ai")
    with ai_container:
        space = st.empty()

        message = ""

        for chunk in response:
            message = message + chunk[0].content
            space.write(message)
        
        st.session_state.history.append({"role":"ai", "content":message})