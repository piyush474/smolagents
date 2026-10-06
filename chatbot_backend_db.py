from langgraph.graph import StateGraph, END, START
from typing import TypedDict, Annotated, Literal, Optional
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages
import sqlite3

from langgraph.prebuilt import ToolNode, tools_condition
from langchain_tavily import TavilySearch
from langchain_core.tools import tool
from typing import Any
import requests
import math
import os

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini")

SYSTEM_PROMPT = "You are an assistant that answers questions or replies within 20 words or less.But you can use tools to get more information."

search_tool = TavilySearch(max_results=5, topic="general", search_depth="advanced")

@tool
def calculator(expression: str) -> str:
    """Use this to calculate the result of a math expression."""
    try:
        allowed = {
            "math": math,
            "abs": abs,
            "round": round,
            "ceil": math.ceil,
            "floor": math.floor,
            "sqrt": math.sqrt,
            "sin": math.sin,
            "cos": math.cos,
            "min": min,
            "max": max,
            "sum": sum,
            "prod": math.prod,
            "factorial": math.factorial,
            "log": math.log,
            "exp": math.exp,
            "pow": math.pow,
            "sqrt": math.sqrt,
        }
        result = eval(expression,{"__builtins__": {}}, allowed)
        return f"The result of {expression} is {result}"
    except Exception as e:
        return f"Calculator Error: {e}"
    

@tool
def get_stock_price(ticker: str) -> str:
    """Use this to get the current stock price of a company."""
    try:
        response = requests.get(f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={ticker}&apikey=HTM6MLWOXVJCK77J")
        response.raise_for_status()
        data = response.json()
        return data
    except Exception as e:
        return f"Stock Price Error: {e}"

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    system_prompt: str
    thread_id: str
    thread_title: str


def format_thread_title(question: str) -> str:
    title = " ".join(question.split())
    if len(title) > 40:
        title = title[:37].rstrip() + "..."
    return title


tools = [search_tool, calculator, get_stock_price]
llm_with_tools = llm.bind_tools(tools)
tool_node = ToolNode(tools)

def chat_node(state: ChatState) -> ChatState:
    system_prompt = state.get("system_prompt") or SYSTEM_PROMPT
    response = llm_with_tools.invoke([SystemMessage(content=system_prompt)] + state["messages"])
    return {
        "messages": [response],
        "system_prompt": system_prompt,
        "thread_id": state.get("thread_id") or "",
        "thread_title": state.get("thread_title") or "",
    }

conn = sqlite3.connect(database="chatbot_backend_db.db", check_same_thread=False)
checkpoint = SqliteSaver(conn)



graph = StateGraph(ChatState)
graph.add_node("chat", chat_node)
graph.add_node("tools", tool_node)

graph.add_edge(START, "chat")
graph.add_conditional_edges("chat", tools_condition)
graph.add_edge("tools", "chat")
graph.add_edge("chat", END)

chatbot = graph.compile(checkpointer=checkpoint)


def get_thread_state(thread_id: str) -> dict:
    state = chatbot.get_state(config={"configurable": {"thread_id": thread_id}})
    values = state.values or {}
    messages = values.get("messages", [])
    title = values.get("thread_title") or ""
    if not title:
        for msg in messages:
            if isinstance(msg, HumanMessage):
                title = format_thread_title(msg.content)
                break
    return {
        "thread_id": values.get("thread_id") or thread_id,
        "thread_title": title or "Untitled",
        "messages": messages,
    }


def get_all_threads():
    """Return saved threads as {thread_id, thread_title, messages}, oldest first."""
    thread_ids = []
    seen = set()
    for ckpt in checkpoint.list(None):
        thread_id = ckpt.config["configurable"]["thread_id"]
        if thread_id in seen:
            continue
        seen.add(thread_id)
        thread_ids.append(thread_id)

    return [get_thread_state(thread_id) for thread_id in reversed(thread_ids)]




# if __name__ == "__main__":
#     # unique identifier for each thread/conversation
#     thread_id = "1"

#     while True:
#         user_input = input("You: ")
#         print("user input: ", user_input)
#         if user_input.lower() in ["exit", "quit"]:
#             break
#         config = {"configurable": {"thread_id": thread_id}}
#         result = chatbot.invoke({"messages": [HumanMessage(content=user_input),]}, config=config)
#         print("bot: ", result["messages"][-1].content)
