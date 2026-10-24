from dotenv import load_dotenv
load_dotenv()

from langgraph.graph  import START,END, StateGraph
from langchain_groq import ChatGroq
from pydantic import BaseModel
from langgraph.types import interrupt
from langgraph.checkpoint.memory import InMemorySaver


class MailState(BaseModel):
    query:str = ""
    draft:str =""
    human_feedback: str = ""
    final_response:str = ""


llm = ChatGroq(model="openai/gpt-oss-20b")

def DraftEmail(state:MailState) -> MailState:
    draft = llm.invoke(state.query).content
    state.draft = draft
    return state


def HumanFeedBack(state:MailState) -> MailState:
    feedback = interrupt({
        "draft_email": state.draft,
        "question": "Do you want to continue or re-write the mail"



    })

    fb = (feedback or "").strip().lower()
    if fb in ("approved", "ok", "yes"):
        state.human_feedback = ""

        return state
    else:
        state.human_feedback = feedback
        return state


def Finalize_node(state:MailState) -> MailState:
    if state.human_feedback:
        state.final_response = state.human_feedback
        return state

    else:
        print("Email sent Successfully ...")
        state.final_response = "Email sent successfully"
        return state
    

graph_builder = StateGraph(MailState)

graph_builder.add_node("draft", DraftEmail)
graph_builder.add_node("human_fb", HumanFeedBack)
graph_builder.add_node("final", Finalize_node)


graph_builder.add_edge(START, "draft")
graph_builder.add_edge("draft", "human_fb")
graph_builder.add_edge("human_fb", "final")
graph_builder.add_edge("final", END)


graph = graph_builder.compile(checkpointer=InMemorySaver)

from IPython.display import Image
Image(graph.get_graph().draw_mermaid_png())