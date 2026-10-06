from chatbot_backend import chatbot
import streamlit as st
from langchain_core.messages import HumanMessage


if __name__ == "__main__":
    st.title("SmolAgents")

    thread_id = "thread-1"
    config = {"configurable": {"thread_id": thread_id}}

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    user_input = st.chat_input("Enter your message")
    if user_input:
        with st.chat_message("user"):
            st.write(user_input)
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        # with st.chat_message("assistant"):
        #     result = st.write_stream(
        #         chunk.content for chunk,metadata in chatbot.stream({"messages": [HumanMessage(content=user_input),]}, config=config, stream_mode="messages")
        #     )
        # result = chatbot.invoke({"messages": [HumanMessage(content=user_input),]}, config=config)
        with st.chat_message("assistant"):
            result = st.write_stream(
                chunk.content for chunk,metadata in chatbot.stream({"messages": [HumanMessage(content=user_input),]}, config=config, stream_mode="messages")
            )
        st.session_state.chat_history.append({"role": "assistant", "content": result})
        print(chatbot.get_state(config))
