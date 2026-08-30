from langchain_community.document_loaders import TextLoader,PyPDFLoader,WebBaseLoader

from langchain_text_splitters import  RecursiveCharacterTextSplitter

from bs4 import BeautifulSoup

from langchain_google_genai import GoogleGenerativeAIEmbeddings

from langchain_community.vectorstores import FAISS

from langchain_core.runnables import RunnablePassthrough

from langchain_core.output_parsers import StrOutputParser

from llm import client



from langchain_core.prompts import ChatPromptTemplate


from langchain_google_genai import ChatGoogleGenerativeAI


import llm



# 1st pipe line
#text loader
FILE_PATH = r"C:\Users\nagas\OneDrive\Desktop\amazon.txt"
data=PyPDFLoader(r"C:\Users\nagas\Downloads\Oop in java final (1).pdf")
doc_thanjavur=WebBaseLoader("https://en.wikipedia.org/wiki/Thanjavur")

loader = TextLoader(FILE_PATH)   # ✅ Correct
documents = loader.load()
#print(documents[0].metadata)
#print(documents[0].page_content)


#print(documents)
document=data.load()
#print(document)
#print(document[0].page_content)
#print(document[0].metadata)
#for i in document:
    #print(i)
    #print(i.metadata)
    #print(i.page_content)
doc_web=doc_thanjavur.load()
#print(doc_web)
doc_web
for i in doc_web:
    print(i)
    #print(i.page_content)

#text splitter
splitter=RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
chunks=splitter.split_text(documents[0].page_content)

#print(chunks)

# embedding
embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-2-preview",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)
vector=embeddings.embed_query("navya")
vector[:5]
# vector db
# vector_db=FAISS.load_local(
#     "faiss_index",
#     allow_dangerous_deserialization=True
# )

vector_db = FAISS.from_texts(
    texts=chunks,
    embedding=embeddings
)


retrivel=vector_db.as_retriever(search_kwargs={"k":5})
vector_db.save_local("faiss_index")
print("local database created")



#----------------------- 2nd pipeline------------------------------------------

import os

from dotenv import load_dotenv



print("Current directory:", os.getcwd())
print("API Key:", os.getenv("NVIDIA_API_KEY"))


load_dotenv()
from langchain_nvidia_ai_endpoints import ChatNVIDIA
client=ChatNVIDIA(
    model="nvidia/nemotron-3-super-120b-a12b",
    api_key=os.getenv("NVIDIA_API_KEY"),
    temperature=0,
    top_p=1,
    max_completion_tokens=512
)


# prompt="you are tourist guide,your name is{name}, and your tone is {tone}"
# prom_tem=ChatPromptTemplate([prompt])
# #print(prom_tem)
# first_tem=prom_tem.invoke({
#     "name":"navya",
#     "tone":"historical"})
# #print(first_tem)
# second_tem=prom_tem.invoke({
#     "name":"navya",
#     "tone":"comedy"})
# #print(second_tem)

# #res=client.invoke(first_tem)
# #print(res)
# # res=client.invoke(second_tem)
# # print(res)
# qus=input("enter the question")
# res=chain.invoke()
# print(res)



prompt="""You are a helpful assistant ,answer the questions the ,dont explain everything just give me in 2 to 3 lines summary
context"{context}
question:{question}"""
prom_tem=ChatPromptTemplate.from_messages(["human",prompt])

client = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)
chain=(
    {
    "context":retrivel,
    "question":RunnablePassthrough()
}
|prom_tem |client | StrOutputParser()
)

qus=input("enter the question")
response = chain.invoke(qus)
print(response) 
#print(type(prompt))


