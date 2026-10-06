from chatbot_backend_db import chatbot, get_all_threads, get_thread_state, format_thread_title
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
import uuid


def get_thread_id():
    return str(uuid.uuid4())

def add_thread(thread_id):
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)

def thread_title(thread_id):
    return st.session_state["thread_titles"].get(thread_id, "New Conversation")

def set_thread_title(thread_id, question):
    st.session_state["thread_titles"][thread_id] = format_thread_title(question)

def reset_chat():
    st.session_state["thread_id"] = get_thread_id()
    st.session_state["message_history"] = []

def _tool_attr(tool_call, key, default=None):
    if isinstance(tool_call, dict):
        return tool_call.get(key, default)
    return getattr(tool_call, key, default)

def _tool_from_call(tool_call):
    return {
        "name": _tool_attr(tool_call, "name") or "tool",
        "args": _tool_attr(tool_call, "args") or {},
    }

def messages_to_history(messages):
    history = []
    pending_tools = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            pending_tools = []
            history.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage):
            if getattr(msg, "tool_calls", None):
                pending_tools.extend(
                    _tool_from_call(tc)
                    for tc in msg.tool_calls
                    if _tool_attr(tc, "name")
                )
            content = msg.content if isinstance(msg.content, str) else ""
            if content:
                history.append({
                    "role": "assistant",
                    "content": content,
                    "tools": pending_tools,
                })
                pending_tools = []
        elif isinstance(msg, ToolMessage):
            continue
    return history

def format_tool_args(args):
    if not args:
        return ""
    parts = []
    for key, value in args.items():
        text = " ".join(str(value).split())
        if len(text) > 80:
            text = text[:77].rstrip() + "..."
        parts.append(f"{key}: {text}")
    return " · ".join(parts)

def _tool_line(tool):
    args = format_tool_args(tool.get("args"))
    line = f":material/build: `{tool['name']}`"
    if args:
        line = f"{line} — {args}"
    return line

def render_tools(tools):
    if not tools:
        return
    names = ", ".join(f"`{tool['name']}`" for tool in tools)
    with st.status(f"Tools used: {names}", state="complete", expanded=False):
        for tool in tools:
            st.write(_tool_line(tool))

def render_message(msg):
    with st.chat_message(msg["role"]):
        if msg.get("tools"):
            render_tools(msg["tools"])
        if msg.get("content"):
            st.write(msg["content"])

def stream_assistant(payload, config, tool_slot):
    status_holder = {"box": None}
    used_tools = []
    pending_by_id = {}
    pending_by_name = {}
    seen_ids = set()

    def show_tool(tool):
        names = ", ".join(f"`{item['name']}`" for item in used_tools)
        label = f"Calling {names}"
        if status_holder["box"] is None:
            with tool_slot:
                status_holder["box"] = st.status(label, expanded=True)
        else:
            status_holder["box"].update(label=label, state="running", expanded=True)
        status_holder["box"].write(_tool_line(tool))

    def generate():
        for chunk, metadata in chatbot.stream(
            payload, config=config, stream_mode="messages"
        ):
            for tool_call in getattr(chunk, "tool_calls", None) or []:
                name = _tool_attr(tool_call, "name")
                if not name:
                    continue
                entry = _tool_from_call(tool_call)
                tool_id = _tool_attr(tool_call, "id")
                if tool_id:
                    pending_by_id[tool_id] = entry
                pending_by_name[name] = entry

            if isinstance(chunk, ToolMessage):
                tool_id = getattr(chunk, "tool_call_id", None)
                if tool_id in seen_ids:
                    continue
                if tool_id:
                    seen_ids.add(tool_id)
                tool = (
                    pending_by_id.get(tool_id)
                    or pending_by_name.get(chunk.name)
                    or {"name": chunk.name or "tool", "args": {}}
                )
                used_tools.append(tool)
                show_tool(tool)
                status_holder["box"].write(
                    f":material/check_circle: `{tool['name']}` finished"
                )

            content = getattr(chunk, "content", None)
            if (
                metadata.get("langgraph_node") == "chat"
                and isinstance(content, str)
                and content
            ):
                yield content

        if status_holder["box"] is not None:
            names = ", ".join(f"`{tool['name']}`" for tool in used_tools)
            status_holder["box"].update(
                label=f"Tools used: {names}",
                state="complete",
                expanded=False,
            )

    result = st.write_stream(generate())
    return result or "", used_tools

def open_thread(thread_id):
    thread = get_thread_state(thread_id)
    st.session_state["thread_id"] = thread["thread_id"]
    st.session_state["thread_titles"][thread["thread_id"]] = thread["thread_title"]
    st.session_state["message_history"] = messages_to_history(thread["messages"])

st.title("Agentic Chatbot with langGraph")

if "chat_threads" not in st.session_state:
    saved_threads = get_all_threads()
    st.session_state["chat_threads"] = [thread["thread_id"] for thread in saved_threads]
    st.session_state["thread_titles"] = {
        thread["thread_id"]: thread["thread_title"]
        for thread in saved_threads
        if thread["thread_title"]
    }
    if saved_threads:
        latest = saved_threads[-1]
        st.session_state["thread_id"] = latest["thread_id"]
        st.session_state["message_history"] = messages_to_history(latest["messages"])

if "thread_titles" not in st.session_state:
    st.session_state["thread_titles"] = {}

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = get_thread_id()

if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

# ======================= sidebar threading section =======================

st.sidebar.title("My Conversations")

if st.sidebar.button("New Conversation"):
    reset_chat()
    st.rerun()

for thread_id in st.session_state["chat_threads"][::-1]:
    if thread_id not in st.session_state["thread_titles"]:
        continue
    if st.sidebar.button(thread_title(thread_id), key=thread_id):
        open_thread(thread_id)
        st.rerun()

# ======================= sidebar threading section ends here =======================

for msg in st.session_state.message_history:
    render_message(msg)


user_input = st.chat_input("Enter your message")

if user_input:
    add_thread(st.session_state["thread_id"])
    if st.session_state["thread_id"] not in st.session_state["thread_titles"]:
        set_thread_title(st.session_state["thread_id"], user_input)

    st.session_state.message_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    CONFIG = {"configurable": {"thread_id": st.session_state["thread_id"]}}
    payload = {
        "messages": [HumanMessage(content=user_input)],
        "thread_id": st.session_state["thread_id"],
        "thread_title": thread_title(st.session_state["thread_id"]),
    }

    with st.chat_message("assistant"):
        tool_slot = st.container()
        result, used_tools = stream_assistant(payload, CONFIG, tool_slot)

    st.session_state.message_history.append({
        "role": "assistant",
        "content": result,
        "tools": used_tools,
    })
    st.rerun()
