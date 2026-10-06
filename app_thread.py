from chatbot_backend import chatbot
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
import uuid

def get_thread_id():
    return str(uuid.uuid4())

def add_thread(thread_id):
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)

def thread_title(thread_id):
    return st.session_state["thread_titles"].get(thread_id, "New Conversation")

def set_thread_title(thread_id, question):
    title = " ".join(question.split())
    if len(title) > 40:
        title = title[:37].rstrip() + "..."
    st.session_state["thread_titles"][thread_id] = title

def reset_chat():
    st.session_state["thread_id"] = get_thread_id()
    st.session_state["message_history"] = []

def load_conversation(thread_id):
    state = chatbot.get_state(config={"configurable": {"thread_id": thread_id}})
    return state.values.get("messages", [])

st.title("Agentic Chatbot with langGraph")

# create message history when the app runs for the first time
if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = get_thread_id()

if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = []

if "thread_titles" not in st.session_state:
    st.session_state["thread_titles"] = {}

# ======================= sidebar threading section =======================

st.sidebar.title("My Conversations")

if st.sidebar.button("New Conversation"):
    reset_chat()
    st.rerun()

for thread_id in st.session_state["chat_threads"][::-1]:
    if thread_id not in st.session_state["thread_titles"]:
        continue
    if st.sidebar.button(thread_title(thread_id), key=thread_id):
        st.session_state["thread_id"] = thread_id
        messages = load_conversation(thread_id)
        temp_messages = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                role = "user"
                if thread_id not in st.session_state["thread_titles"]:
                    set_thread_title(thread_id, msg.content)
            elif isinstance(msg, AIMessage):
                role = "assistant"
            else:
                continue
            temp_messages.append({"role": role, "content": msg.content})
        st.session_state["message_history"] = temp_messages
        st.rerun()

# ======================= sidebar threading section ends here =======================



for msg in st.session_state.message_history:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])


user_input = st.chat_input("Enter your message")

if user_input:
    add_thread(st.session_state["thread_id"])
    if st.session_state["thread_id"] not in st.session_state["thread_titles"]:
        set_thread_title(st.session_state["thread_id"], user_input)

    st.session_state.message_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)
        
    CONFIG = {"configurable": {"thread_id": st.session_state["thread_id"]}}

    with st.chat_message("assistant"):
        result = st.write_stream(
            chunk.content for chunk,metadata in chatbot.stream({"messages": [HumanMessage(content=user_input),]}, config=CONFIG, stream_mode="messages")
            if isinstance(chunk, AIMessage)
        )
    st.session_state.message_history.append({"role": "assistant", "content": result})
    st.rerun()
    

