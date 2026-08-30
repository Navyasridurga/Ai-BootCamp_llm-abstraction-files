from llm import client
from langchain_core.prompts import ChatPromptTemplate
import os
print(os.getenv("NVIDIA_API_KEY"))

prompt="you are tourist guide,your name is{name}, and your tone is {tone}"
prom_tem=ChatPromptTemplate([prompt])
#print(prom_tem)
first_tem=prom_tem.invoke({
    "name":"navya",
    "tone":"historical"})
#print(first_tem)
second_tem=prom_tem.invoke({
    "name":"navya",
    "tone":"comedy"})
#print(second_tem)

#res=client.invoke(first_tem)
#print(res)
res=client.invoke(second_tem)
print(res)

