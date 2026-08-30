from llm import client
from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_tavily import TavilySearch


import os
from dotenv import load_dotenv
from langchain_tavily import TavilySearch
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()
from langchain_nvidia_ai_endpoints import ChatNVIDIA
client = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    GOOGLE_API_KEY = os.getenv(" google_api_key")
)

tavily_search = TavilySearch(
    tavily_api_key=os.getenv("TAVILY_API_KEY"),
    max_results=5,
    topic="general",
)



    # include_answer=False,
    # include_raw_content=False,
    # include_images=False,
    # include_image_descriptions=False,
    # search_depth="basic",
    # time_range="day",
    # start_date=None,
    # end_date=None,
    # include_domains=None,
    # exclude_domains=None,
    # include_usage= False

#manually
# res_tavily=tavily_search.invoke({
#     "query":"who is the cm of tamil nadu"

# })
# print(res_tavily)

from langchain.agents import create_agent

from  langchain_core.messages import SystemMessage,HumanMessage

search_agent=create_agent(
  model =client,
  tools=[tavily_search],
  system_prompt=SystemMessage(
      """
you are a search assisstant
for any question to current evemts after jan 2025
you must use the search tool and answer,dont rely on internal sources

"""
  )

)
res_agent=search_agent.invoke({
    "messages":[HumanMessage(
       # "who is the cm of tamil nadu in 2026 march "
       #content="what is the weather of delhi on 31st july 2026 "
       #content="what is the weather of  tanuku"
       content="hi my name is navya ,what is the weather of attili"
    )]
}
)
#print(res_agent)
#print(res_agent["messages"][-1])
#print(res_agent["messages"][-1].content[0]["text"])
#custom tool
import http
import requests as req
from langchain.tools import tool
weather_api="467998f562e29b665acb175c25e27d80"
@tool
def get_weather(city: str) -> dict | str:
    """
    Fetches the current weather forecast for a given city.
    """

    url = (
        f"https://api.openweathermap.org/data/2.5/forecast"
        f"?q={city}&appid={weather_api}"
    )

    res = req.get(url)

    if res.status_code == 200:
        data = res.json()
        return data
    else:
        return f"Could not fetch weather data for {city}. Status code: {res.status_code}"

weather_agent=create_agent(
    model=client,
     tools=[get_weather],
     checkpointer=InMemorySaver(),

      system_prompt=SystemMessage(
          """
   you are a helpful weather assistant,get the suggestions for clothing and medicine for city
based on weather data 
    
    """
      )


)



result=weather_agent.invoke(
    {"messages":[
     {"role":"user",
      "content":"what is the weather in san fransico"}]},
      config={
          "configurable":{
              "thread_id":"1"
              }
              }
)
#print(res_agent["messages"][-1].content[0]["text"])
print(result["messages"][-1].content[0]["text"])