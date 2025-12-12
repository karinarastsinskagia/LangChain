import requests

from langchain_ollama import ChatOllama
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

SYSTEM_PROMPT = """You are a weather forecast assistant.
Your job is to take a location and find its current weather conditions.
Use the 'search' tool to resolve the location and the 'get_weather' tool to get actual weather data.
Do not guess the weather yourself.
Return a concise summary in this style:

"The weather in {location} is currently {temp} degrees Celsius with a wind speed of {wind} kilometers per hour.Advice: {advice}"

Here, {advice} should be a short recommendation like 'take an umbrella' or 'wear warm clothes'.
Do not add anything else besides this single sentence.
"""

location = input("Which country or city are your interested for the weather conditions ?:")
if not location:
    location = "New York"

model = ChatOllama(
    model="mistral",
    temperature=0.1,
    max_tokens=10000,
    timeout=30
)

@tool
def search(location: str) -> dict:
    """Return latitude and longitude for a given location name using OSM."""
    url = "https://nominatim.openstreetmap.org/search"
    params = {"q": location, "format": "json", "limit": 1}

    res = requests.get(url, params=params, headers={"User-Agent": "LangChainWeatherBot"})
    data = res.json()

    if not data:
        return {"error": "Location not found"}


    return {"lat": float(data[0]["lat"]), "lon": float(data[0]["lon"]), "location": location}

@tool
def get_weather(lat: float, lon: float) -> str:

    """Get current weather using Open-Meteo (free)."""

    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "current_weather": True,
    }

    res = requests.get(url, params=params)
    current = res.json().get("current_weather", {})

    temp = current.get("temperature")
    wind = current.get("windspeed")

    return f"Weather is: {temp}°C, wind {wind} km/h"

agent = create_agent(
    model,
    tools=[search,get_weather],
    system_prompt=SYSTEM_PROMPT,
)
input_messages = [
    {"role": "user", "content": location}
]
response = agent.invoke({"messages": input_messages })
messages = response.get("messages", [])

# Find the last assistant message
for msg in messages:
    # print(msg)
    if isinstance(msg, AIMessage):
        print(msg.content)

#todo Create chain are you interested for other location ? which one?

