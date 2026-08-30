from llm import client
lc_messages=[
    {
        "role":"user",
        "content":"hello tell me about you",
    },

]
response=client.invoke(lc_messages)
print(response.content)



