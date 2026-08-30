from langchain_google_genai import GoogleGenerativeAI,ChatGoogleGenerativeAI
client=ChatGoogleGenerativeAI(model="gemini=3.6-flash", google_api_key=os.getenv("GOOGLE_API_KEY"))
import requests as req
from langchain.tools import tool
from langchain.agents import create_agent
from langchain_core.messages import SystemMessage,HumanMessage,AIMessage
import os
from dotenv import load_dotenv
from langchain_tavily import TavilySearch

load_dotenv()
from langchain_nvidia_ai_endpoints import ChatNVIDIA
client = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
     GOOGLE_API_KEY = os.getenv(" google_api_key")
)

@tool
def get_products() ->  str:
    """
    Fetches the products for a fake store api and retirn it
    """

    url = (
        "https://fakestoreapi.com/products"
    )
    

    res = req.get(url)

    if res.status_code == 200:
        data = res.json()
        return data
    else:
        return f"Could not fetch products. Status code: {res.status_code}"
search_agent=create_agent(
  model =client,
  tools=[get_products],
  system_prompt=SystemMessage(
      """
you are a products assisstant in ecommerece of fakestore api 
for any question given
for any question to current events or events after jan 2025
you must use the search tool and answer,dont rely on internal sources

"""
  )

)
res_agent=search_agent.invoke({
    "messages":[HumanMessage(
       # "who is the cm of tamil nadu in 2026 march "
       #content="what is the weather of delhi on 31st july 2026 "
       content=" give me products which are having rating below 2.0"
    )]
}
)
#print(res_agent)
print(res_agent["messages"][-1].content[0]["text"])
