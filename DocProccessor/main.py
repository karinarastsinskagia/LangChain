from langchain_classic.chains.question_answering.map_reduce_prompt import messages
from langchain_ollama import ChatOllama
from langchain.agents import create_agent
from langchain.messages import SystemMessage, HumanMessage, AIMessage
from langchain_community.document_loaders.csv_loader import CSVLoader

SYSTEM_PROMPT = """You are an expert text analyser for pros and cons based on reviews.
Your job is to take a text with pros and cons and group them into topics 
where each topic should have a sentence with the format of Users said, Users mentioned .."""

pros_themes = [
    "Ease of Use & Navigation",
    "Customer Support & Service",
    "Customization & Flexibility",
    "Feature Set & Functionality",
    "Performance & Reliability",
    "Implementation & Onboarding",
    "User Experience & Engagement",
    "Value for Money / Pricing & ROI"
]

cons_themes = [
    "Limited Features / Missing Tools",
    "Customization Limitations",
    "Reporting & Analytics Issues",
    "Complexity & Learning Curve",
    "Cost & Add-ons",
    "Integration & Technical Challenges",
    "Performance or Reliability Issues"
]


def separate_pros_cons_by_read_file(file_path: str):
    loader = CSVLoader(file_path=file_path)

    data = loader.load()

    pros_list = []
    cons_list = []

    for review in data:
        text = review.page_content

        parts = text.split("cons:")
        pros = parts[0].replace("pros:", "").strip()
        cons = parts[1].strip()

        pros_list.append(pros)
        cons_list.append(cons)

    return pros_list, cons_list


pros, cons = separate_pros_cons_by_read_file("pros-cons.csv")
# print(pros)
# print(cons)

model = ChatOllama(
    model="gemma",
    temperature=0.1,
    max_tokens=10000,
    timeout=30
)

predefined_messages = [
    SystemMessage("You are a helpful assistant that summarize the given pros and cons in topics."),
    HumanMessage(
        content=f"Here are the pros:\n{pros}\n\nHere are the cons:\n{cons}\n\n"
                f"Please group them into into these categories for pros:{pros_themes} and these for cons:{cons_themes}."
                f"Write a short, unique description summarizing what users express, using varied sentence structures and tones (e.g., Users find…, Users appreciate…, Users value…, Users desire…, etc.)."
                f"Indicate how many reviews mention this theme (note that a single review may mention multiple themes)."
                f"I want the Output format to be: Theme: [Theme Name]  Description: [Short, varied summary of user sentiment]  Number of Mentions: [Count]"),
    AIMessage("The main pros and cons based on the given input are...")
]

response: AIMessage = model.invoke(predefined_messages)
print(response.content)

# agent = create_agent(
#     model,
#     # tools=[analyse_pros_cons],
#     system_prompt=SYSTEM_PROMPT,
# )
# response = agent.invoke({"messages": predefined_messages})
# print(type(response['messages']))
# # Get AI Message
# ai_message: AIMessage = response['messages'][2]
# print(ai_message.content)
