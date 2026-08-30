from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import  RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate

import os

from langchain_nvidia_ai_endpoints import ChatNVIDIA
client=ChatNVIDIA(
    model="nvidia/nemotron-3-super-120b-a12b",
    api_key=os.getenv("NVIDIA_API_KEY"),
    temperature=0,
    top_p=1,
    max_completion_tokens=512
)


from langchain_core.runnables import RunnablePassthrough
embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-2-preview",
    api_key=os.getenv("GOOGLE_API_KEY")
)
vector=embeddings.embed_query("navya")
vector[:5]
splitter=RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
FILE_PATH = r"C:\Users\nagas\OneDrive\Desktop\amazon.txt"
loader = TextLoader(FILE_PATH) 
documents = loader.load()
chunks=splitter.split_text(documents[0].page_content)
   
print(chunks)
print(vector)


vector_db=FAISS.from_documents(
    documents=chunks,
    embedding=embeddings
)


retrivel=vector_db.as_retriever(search_kwargs={"k":5})

chain=({
    "context":retrivel,
    "question":RunnablePassthrough()
})
# prompt |client
# qus=input("enter the question")
# res=chain.invoke(qus)
# print(res)