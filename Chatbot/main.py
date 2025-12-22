import requests
from langchain_classic.agents import AgentExecutor,create_react_agent
from langchain_classic.memory import ConversationBufferMemory
from langchain_community.tools import wikipedia, WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama

prompt = PromptTemplate.from_template("""
You are a helpful AI assistant.
Use tools when necessary to answer factual questions.

Conversation history:
{chat_history}

You have access to the following tools:
{tools}

Tool names:
{tool_names}

User question:
{input}

{agent_scratchpad}

Always end with 'Final Answer:' followed by your complete response after using tools or gathering information. Do not repeat or loop indefinitely.
""")

model = ChatOllama(
    model="mistral",
    temperature=0.3,
    max_tokens=10000,
    timeout=30
)

wiki = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper(top_k_results=3, doc_content_chars_max=2000))
memory = ConversationBufferMemory(memory_key="chat_history",return_messages=True)

agent = create_react_agent(
    model,
    tools=[wiki],
    prompt=prompt,
)

agent_executor = AgentExecutor(
            agent=agent,
            tools=[wiki],
            memory=memory,
            max_iterations = 2,
            verbose= True,  # Enable for debugging
            handle_parsing_errors=True  # Gracefully handle agent errors
        )


while True:
    user_query = input("What you would like to ask/know? ('q' to quit): ")
    if user_query.strip().lower() in {"q", "quit", "exit"}:
        print("Goodbye!")
        break

    if not user_query.strip():
        print("Nothing you want to ask? Ok bye!")
        break


    # Invoke the agent with the running history
    response = agent_executor.invoke({"input": user_query})
    response_messages = response.get("messages", [])

    # Find and print the assistant's message
    for msg in response_messages:
        print(msg)

    # Ask whether the user wants another location (outside of the LLM to keep outputs clean)
    again = input("Would you like to ask anything else? (y/n): ").strip().lower()
    if again not in {"y", "yes"}:
        print("Goodbye!")
        break

