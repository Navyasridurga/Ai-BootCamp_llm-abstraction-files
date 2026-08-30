
from llm import client

from langchain_core.messages import SystemMessage,HumanMessage,AIMessage
system_msg = SystemMessage("You are a helpful coding assistant.")

messages = [
 SystemMessage("you are an expert in general knowledge"),
 HumanMessage("what is the capital city of india"),
 AIMessage("capital:delhi"),
 HumanMessage("What is the capital city of maharastra"),
 AIMessage("capital:mumbai"),
 HumanMessage("what is the capital of gujarat")

]
response =client.invoke(messages)
print(response.content)