from langchain_ollama import ChatOllama
from langchain_core.tools import tool
import requests
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub

import os
from dotenv import load_dotenv

os.environ['LANGCHAIN_PROJECT'] = '4_agent.py.py'
load_dotenv()
search_tool = DuckDuckGoSearchRun()
API_KEY = os.getenv("WEATHERSTACK_API_KEY")


@tool
def get_weather_data(city: str) -> str:
    """
    This function fetches the current weather data for a give city
    """
    url = f"https://api.weatherstack.com/current?access_key={API_KEY}&query={city}"
    response = requests.get(url)
    return response.json()

llm = ChatOllama(model="llama3.2", temperature=0)

# Step 2: Pull the ReAct prompt from Langchain Hub
prompt = hub.pull("hwchase17/react") # pulls the standard ReAct agent prompt

# Step 3: Create the ReAct agent manually with the pulled prompt
agent = create_react_agent(
    llm = llm,
    tools = [search_tool, get_weather_data],
    prompt = prompt
)

# Step 4: Wrap it with AgentExecutor
agent_executor = AgentExecutor(
    agent = agent,
    tools = [search_tool, get_weather_data],
    verbos = True,
    max_iterations = 5
)

# What is the release date of Agenvers End Game?
# What is the current temp of Kathmandu?
# Identify the birthplace city of Pasang Lhamu Sherpa (search) and give its currenct temperature.

# Step 5: Invoke
response = agent_executor.invoke({"input": "What is the current temp of Kathmandu"})
print(response)
print(response['output'])