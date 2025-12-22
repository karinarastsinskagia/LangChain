### Installation

The following commands are required in order to execute projects: 

`pip install langchain`

`pip install -U langchain-ollama`

`ollama pull gpt-oss:20b` | `ollama pull mistral` | `ollama pull gemma`

`pip install langchain community`

**Note**: I used ChatOllama model because it is totally free and not required extra configuration (OPEN AI Key etc.).

### Description

1. `DocProccessor` folder, use llm for analyzing the given csv of pros & cons and grouping them into topics/categories. 
2. `WeatherForeCast` folder, includes an AI agent for fetch weather information for the requested cities / countries. 
3. `Chatbot` folder, includes a local conversational chatbot assistant. 
