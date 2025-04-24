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

## Fetch trending topics using NewsAPI
## This function fetches the top 3 trending news articles and formats them for the agent.
@function_tool  
async def fetch_trending_topics(topic: str):
    print(f"🔍 Fetching news for: {topic}")
    try:
        headlines = newsapi.get_top_headlines(q=topic, language='en', page_size=3)
        print("API Status:", headlines.get('status'))
        print("Total Results:", headlines.get('totalResults', 'N/A'))
        articles = headlines.get('articles', [])

        if len(articles) < 3:
            print("🔁 Trying fallback get_everything...")
            headlines = newsapi.get_everything(q=topic, language='en', page_size=5, sort_by='relevancy')
            print("API Status:", headlines.get('status'))
            print("Total Results:", headlines.get('totalResults', 'N/A'))
            articles = headlines.get('articles', [])

        if not articles:
            return "🚫 No articles found for this topic. Try a broader or different topic."

        formatted = []
        for article in articles:
            title = article.get('title', '[No Title]')
            description = article.get('description', '[No Description]')
            url = article.get('url', '[No URL]')
            formatted.append(f"Title: {title}\nDescription: {description}\nURL: {url}")

        result = "\n\n".join(formatted)
        return result

    except Exception as e:
        return f"❌ Error fetching news: {str(e)}"




## News Scout Agent
## This agent fetches trending topics and formats them for the writer agent.
news_scout_agent = Agent(
    name="news_scout_agent",
    instructions="You are a news scout. Provide 3 relevant, recent trending news topics including their titles, short descriptions, and URLs.",
    tools=[fetch_trending_topics]
)

## Writer Agent
## This agent generates a news article and headline based on the given topic.
writer_agent = Agent(
    name="writer_agent",
    instructions="You are a professional news writer. Based on the given topic, generate a headline and a complete article (100-150 words). Your writing should be clear, engaging, and factual. Avoid bias or unsupported claims.",
)

## Manager Agent
## This agent coordinates the workflow between the news scout and writer agents.
manager_agent = Agent(
    name="manager_agent",
    instructions=
        """
        You are the manager agent of a multi-agent content creation system called 'Daily Danny'. Your goal is to publish one original news-style article for a specific user.

        You have access to the following agents:
        - News Scout Agent: Fetches trending topics or news headlines.
        - Writer Agent: Writes a news article and headline about a given topic.

        Your job is to:
        1. Use the user-provided topic to ask the News Scout Agent for trending news in that category.
        2. After fetching trending topics, present 3 of them to the user labeled with numbers and ask which one they would like to write about. Wait for their response before continuing.
        3. Ask the Writer Agent to write an article about the chosen result.
        4. Ask the user if they are satisified with the article. If not, ask them what they would like to change and update the article accordingly.
        5. If the user is satisfied, end the conversation and provide the final article.

        Be sure to log all steps for transparency. Prioritize journalistic integrity, clarity, and user relevance.
        """,
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

def extract_all_text(run_result):
    texts = []
    for item in run_result.new_items:
        # 1) Agent-generated messages
        if isinstance(item, MessageOutputItem):
            texts.append(ItemHelpers.extract_last_content(item.raw_item))
        # 2) Tool‐call outputs (if you care about those too)
        else:
            texts.append(str(item.raw_item))
    return "\n\n".join(texts)






async def main():
    prompt = input("Enter a topic you would like to write an article about:\n> ")
    print("Starting Article Writer...")
    result = await Runner.run(manager_agent, prompt)

    while True:
        # Print out any new agent messages
        for item in result.new_items:
            if isinstance(item, MessageOutputItem):
                text = ItemHelpers.extract_last_content(item.raw_item)
                print(f"{item.agent.name}: {text}")

        # Check if the latest agent message is asking the user a question
        last_msg = ItemHelpers.extract_last_content(result.new_items[-1].raw_item).lower()
        if any(w in last_msg for w in ("which one", "would you like", "yes", "no")):
            user_input = input("Your response:\n> ").strip()
            if not user_input:
                print("No input—exiting.")
                break

            all_text = extract_all_text(result)
            all_text += f"\n\nUser: {user_input}"
            result = await Runner.run(
                manager_agent,
                all_text,
            )
            continue

        # If the agent isn’t asking anything, we assume it’s done
        break

    print("✅ Done.")
    

if __name__ == "__main__":
    asyncio.run(main())