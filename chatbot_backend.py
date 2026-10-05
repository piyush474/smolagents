from langgraph.graph import StateGraph, END, START
from typing import TypedDict, Annotated, Literal, Optional
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini")

system_message = SystemMessage(content="You are an assistant that answers questions or replies within 20 words or less.")


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def chat_node(state: ChatState) -> ChatState:
    # Prepend for the model only. Do not return it, so it is never stored or shown.
    messages = [system_message] + state["messages"]
    response = llm.invoke(messages)
    return {"messages": [response]}


checkpoint = MemorySaver()

graph = StateGraph(ChatState)
graph.add_node("chat", chat_node)
graph.add_edge(START, "chat")
graph.add_edge("chat", END)

chatbot = graph.compile(checkpointer=checkpoint)


if __name__ == "__main__":
    # unique identifier for each thread/conversation
    thread_id = "1"

    while True:
        user_input = input("You: ")
        print("user input: ", user_input)
        if user_input.lower() in ["exit", "quit"]:
            break
        config = {"configurable": {"thread_id": thread_id}}
        result = chatbot.invoke({"messages": [HumanMessage(content=user_input),]}, config=config)
        print("bot: ", result["messages"][-1].content)
