from langchain_ollama import ChatOllama
from langchain.messages import SystemMessage, HumanMessage, AIMessage
from langchain_community.document_loaders.csv_loader import CSVLoader

SYSTEM_PROMPT = """You are an expert text analyser for pros and cons based on reviews.
Your job is to take a text with pros and cons and group them into topics 
where each topic should have a sentence with the format of Users said, Users mentioned .."""

def separate_pros_cons_by_read_file(file: str):
    loader = CSVLoader(file_path=file)

    data = loader.load()

    pros_list = []
    cons_list = []

    for review in data:
        text = review.page_content

        parts = text.split("cons:")
        row_pros = parts[0].replace("pros:", "").strip()
        row_con = parts[1].strip()

        pros_list.append(row_pros)
        cons_list.append(row_con)

    return pros_list, cons_list

def separate_pros_cons_topics_by_read_file(file: str):
    loader = CSVLoader(file_path=file)

    data = loader.load()

    pros_themes_list = []
    cons_themes_list = []

    for topic in data:
        text = topic.page_content

        parts = text.split("cons_themes:")
        row_pros = parts[0].replace("pros_themes:", "").strip()
        row_con = parts[1].strip()

        pros_themes_list.append(row_pros)
        cons_themes_list.append(row_con)

    return pros_themes_list, cons_themes_list

def ask_llm_to_extract_pros_and_cons_topics(pros_list: list, cons_list:list, pros_themes_list: list, cons_themes_list:list)-> str:

    model = ChatOllama(
        model="gemma",
        temperature=0.1,
        max_tokens=10000,
        timeout=30
    )

    predefined_messages = [
        SystemMessage("You are a helpful assistant that summarize the given pros and cons in topics."),
        HumanMessage(
            content=f"Here are the pros:\n{pros_list}\n\nHere are the cons:\n{cons_list}\n\n"
                    f"Please group them into into these categories for pros:{pros_themes_list} and these for cons:{cons_themes_list}."
                    f"Write a short, unique description summarizing what users express, using varied sentence structures and tones (e.g., Users find…, Users appreciate…, Users value…, Users desire…, etc.)."
                    f"Indicate how many reviews mention this theme (note that a single review may mention multiple themes)."
                    f"I want the Output format to be: Theme: [Theme Name]  Description: [Short, varied summary of user sentiment]  Number of Mentions: [Count]"),
        AIMessage("The main pros and cons based on the given input are...")
    ]

    response: AIMessage = model.invoke(predefined_messages)
    return response.content

file_reviews_path = input("Do you want to insert other file input? If yes, type your file path otherwise press enter:")
if not file_reviews_path:
    file_reviews_path = "pros-cons.csv"


file_topics_path = input("Do you want to insert other file input for topics? If yes, type your file path otherwise press enter:")
if not file_topics_path:
    file_topics_path = "pros-cons-topics.csv"

pros, cons = separate_pros_cons_by_read_file(file_reviews_path)
pros_themes, cons_themes = separate_pros_cons_topics_by_read_file(file_topics_path)

result = ask_llm_to_extract_pros_and_cons_topics(pros,cons,pros_themes,cons_themes)
print(result)