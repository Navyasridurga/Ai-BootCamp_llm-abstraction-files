from langchain_community.document_loaders import TextLoader,PyPDFLoader,WebBaseLoader
from bs4 import BeautifulSoup
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
    #print(i)
    print(i.page_content)
