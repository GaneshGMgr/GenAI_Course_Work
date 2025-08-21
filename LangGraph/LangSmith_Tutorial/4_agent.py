# -------------------- setup --------------------
from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub
import requests
import os
from dotenv import load_dotenv

# Load .env variables
load_dotenv()
API_KEY = os.getenv("WEATHERSTACK_API_KEY")  # your weatherstack key

# -------------------- tools --------------------
@tool
def get_weather_data(city: str) -> str:
    """Fetch current weather data for a city using Weatherstack API."""
    url = f"https://api.weatherstack.com/current?access_key={API_KEY}&query={city}"
    response = requests.get(url)
    if response.status_code != 200:
        return f"Error fetching data: {response.status_code}"
    data = response.json()
    if "current" not in data:
        return f"No weather data found for {city}."
    temp = data["current"]["temperature"]
    desc = data["current"]["weather_descriptions"][0]
    return f"The current temperature in {city} is {temp}°C with {desc}."

# Use DDGS for web search
from duckduckgo_search import DDGS

@tool
def web_search(query: str) -> str:
    """Search the web using DuckDuckGo and return top 3 results."""
    with DDGS() as ddgs:
        results = list(ddgs.text(query, max_results=3))
    if not results:
        return "No results found."
    output = []
    for r in results:
        title = r.get("title", "")
        url = r.get("href", "")
        snippet = r.get("body", "")
        output.append(f"{title} - {url}\n{snippet}")
    return "\n\n".join(output)

# -------------------- LLM and agent --------------------
llm = ChatOllama(model="llama3.2", temperature=0)

# Pull standard ReAct agent prompt
prompt = hub.pull("hwchase17/react")  

# Create ReAct agent
agent = create_react_agent(
    llm=llm,
    tools=[web_search, get_weather_data],
    prompt=prompt
)

# Wrap with AgentExecutor
agent_executor = AgentExecutor(
    agent=agent,
    tools=[web_search, get_weather_data],
    verbose=True,
    max_iterations=5
)

# -------------------- run --------------------
if __name__ == "__main__":
    question = input("Ask your question: ").strip()
    response = agent_executor.invoke({"input": question})
    print("\n--- AGENT OUTPUT ---")
    print(response['output'])


# What is the release date of Agenvers End Game?
# What is the current temp of Kathmandu?
# Identify the birthplace city of Pasang Lhamu Sherpa (search) and give its currenct temperature.