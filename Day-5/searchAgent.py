from dotenv import load_dotenv
load_dotenv()

from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_groq import ChatGroq 
from langchain.agents import create_agent
from langchain_core.tools import tool

model  = ChatGroq(model="openai/gpt-oss-20b")
search = GoogleSerperAPIWrapper()

agent = create_agent(
    model=model,
    tools=[search.run],
    system_prompt="You are a agent and can search for any question , information on google."
)


while True:
    query = input("User: ")
    if query.lower() =="quit":
        print("Good bye")
        break

    response = agent.invoke({"messages":[{"role":"user", "content":query}]})

    print(response["messages"][-1].content)