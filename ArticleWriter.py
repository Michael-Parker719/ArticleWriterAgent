from agents import Agent, Runner, FunctionTool, function_tool
from agents.items import MessageOutputItem, ItemHelpers
import asyncio
from newsapi import NewsApiClient
import os
from dotenv import load_dotenv
load_dotenv()

NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

newsapi = NewsApiClient(api_key=NEWSAPI_KEY)

@function_tool  
async def fetch_trending_topics():
    headlines = newsapi.get_top_headlines(language='en', page_size=3)
    articles = headlines.get('articles', [])

    formatted = []
    for article in articles:
        title = article.get('title')
        description = article.get('description')
        url = article.get('url')
        if title and description and url:
            formatted.append(f"Title: {title}\nDescription: {description}\nURL: {url}")

    result = "\n\n".join(formatted)
    return MessageOutputItem(output=result)

news_scout_agent = Agent(
    name="news_scout_agent",
    instructions="You are a news scout. Provide 3 relevant, recent trending news topics including their titles, short descriptions, and URLs.",
    tools=[fetch_trending_topics]
)

writer_agent = Agent(
    name="writer_agent",
    instructions="You are a professional news writer. Based on the given topic, generate a headline and a complete article (100-150 words). Your writing should be clear, engaging, and factual. Avoid bias or unsupported claims.",
)


manager_agent = Agent(
    name="manager_agent",
    instructions=
    "You are the manager agent of a multi-agent content creation system called 'Daily Danny'. Your goal is to publish one original news-style article for a specific user."
    "You have access to the following agents:"
    "- News Scout Agent: Fetches trending topics or news headlines."
    "- Writer Agent: Writes a news article and headline about a given topic."
    # "- Image Finder Agent: Finds public domain or relevant web images related to a given topic."
    # "- DALL·E Agent: Generates a cover image using a given image description."
    # "- Database Agent: Posts the finalized article (title, content, images) to a MySQL database."

    "Your job is to:"
    "1. Ask the News Scout Agent to find 3 recent trending topics."
    "2. Select the most engaging topic (based on novelty, importance, and relevance to the user)."
    "3. Ask the Writer Agent to create a compelling article and headline."
    # "4. Use the Image Finder Agent to get 2-3 related images."
    # "5. Ask the DALL·E Agent to create a high-quality cover image."
    # "6. Pass the title, article body, image URLs, and metadata to the Database Agent for saving."

    "Be sure to log all steps for transparency. Prioritize journalistic integrity, clarity, and user relevance.",
    tools=[
        news_scout_agent.as_tool(
            tool_name="news_scout_tool",
            tool_description="Use this tool to find trending news topics from external sources (RSS, NewsAPI, Bing News).",
        ),
        writer_agent.as_tool(
            tool_name="writer_tool",
            tool_description="Use this tool to write a news article and headline based on the given topic.",
        ),
        # image_finder_agent,
        # dalle_agent,
        # database_agent,
    ],
)






async def main():
    prompt = input("Enter a topic you would like to write an article about:\n> ")
    print("Starting Article Writer...")
    result = await Runner.run(manager_agent, prompt)

    for item in result.new_items:
        if isinstance(item, MessageOutputItem):
            # item.agent.name is the agent's name
            text = ItemHelpers.extract_last_content(item.raw_item)
            print(f"{item.agent.name}: {text}")
        else:
            # for tool calls, handoffs, etc., just show the type and raw data
            print(f"{item.type}: {item.raw_item}")
    

if __name__ == "__main__":
    asyncio.run(main())