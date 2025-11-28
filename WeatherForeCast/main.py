from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from langchain.agents import create_agent
from langchain.tools import tool

SYSTEM_PROMPT = """You are a weather forecast assistant.
Your job is to take a location and find its current weather conditions.
Use the 'search' tool to resolve the location and the 'get_weather' tool to get actual weather data.
Do not guess the weather yourself.
Return a concise summary."""

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
    """
    Convert a location name (city, country) to a standardized location or coordinates.
    """
    return {"lat": 40.7128, "lon": -74.0060, "name": location}

@tool
def get_weather(lat: float, lon: float) -> str:
    """
    Fetch weather data for the given coordinates from a real API.
    """
    return f"Current weather at coordinates ({lat}, {lon}): 22°C, clear sky"


agent = create_agent(
    model,
    tools=[search,get_weather],
    system_prompt=SYSTEM_PROMPT,
)

response = agent.invoke({"messages": [{"role": "user", "content": location}]})

messages = response.get("messages", [])
final_text= ""
# Find the last assistant message
for msg in messages:
    print(msg)
    # if msg.get("role") == "assistant":
    #     final_text = msg.get("content")
    #     break

print(final_text)

#todo Create chain are you interested for other location ? which one?
#todo use memory

