from llm import client

lc_messages=[
    {
        "role":"user",
        "content":"hello tell me about operating system as gate level question exact in 2026",
    },

]
response=client.invoke(lc_messages)
print(response.content)