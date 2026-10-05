from chatbot_backend import chatbot
from langchain_core.messages import HumanMessage

if __name__ == "__main__":
    chatbot.invoke({"messages": [HumanMessage(content="Hello, how are you?"),]})