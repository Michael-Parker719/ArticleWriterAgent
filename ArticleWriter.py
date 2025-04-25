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
    print(f"Fetching news for: {topic}")
    try:
        page_size = 6
        headlines = newsapi.get_top_headlines(q=topic, language='en', page_size=page_size)
        print("API Status:", headlines.get('status'))
        print("Total Results:", headlines.get('totalResults', 'N/A'))
        articles = headlines.get('articles', []) or []

        if len(articles) < page_size:
            print("Not enough articles found, trying get_everything...")
            more = newsapi.get_everything(q=topic, language='en', page_size=page_size, sort_by='relevancy')
            print("Fallback Status:", more.get('status'))
            print("Fallback Total:", more.get('totalResults', 'N/A'))

            extra = more.get('articles', []) or []
            urls = {a['url'] for a in articles if a.get('url')}
            for art in extra:
                if art.get('url') not in urls:
                    articles.append(art)
                    urls.add(art['url'])
                if len(articles) > page_size:
                    break


        if not articles:
            return "No articles found for this topic. Try a broader or different topic."

        formatted = []
        i = 0
        for art in articles[:page_size]:
            title = art.get('title', '[No Title]')
            description = art.get('description', '[No Description]')
            url = art.get('url', '[No URL]')
            formatted.append(f"{i+1}) Title: {title}\nDescription: {description}\nURL: {url}")
            i+=1
        result = "\n\n".join(formatted)
        return result

    except Exception as e:
        return f"Error fetching news: {str(e)}"




## News Scout Agent
## This agent fetches trending topics and formats them for the writer agent.
news_scout_agent = Agent(
    name="news_scout_agent",
    instructions="You are a news scout. Provide **up to 5** relevant, recent trending news topics including their titles, short descriptions, and URLs."
    "This will be returned to you in the format: {i}) Title: {title}\nDescription: {description}\nURL: {url}"
    "It is very important to include the Title, description, and URL in the same format you recieve it, as this will be used to find sources later."
    "If no articles are found, return 'No articles found for this topic. Try a broader or different topic.'",
    tools=[fetch_trending_topics]
)

## Writer Agent
## This agent generates a news article and headline based on the given topic.
writer_agent = Agent(
    name="writer_agent",
    instructions="You are a professional news writer. Based on the given topic, generate a headline and a complete article (100-150 words)." 
    "Your writing should be clear, engaging, and factual. Avoid bias or unsupported claims."
    "Include a headline, article body, and a list of sources in markdown format. "
    "If you can't find any sources, return 'No sources found.'"
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

        You are the manager agent of Daily Danny. Your job is:
        1. Take the user’s topic and find headlines from recent news. → call news_scout_tool(topic).
        2. Present at most 5 headlines to the user. If 5 headlines are available, present 5 → ask “Which one?”
        3. On choice → call writer_tool with the chosen headline to draft the article. Be sure to inclue the headline, description, and URL in the prompt to the writer agent.
        4. Present the article to the user
        6) Ask the user if they’re satisfied or want edits.
        """,
    tools=[
        news_scout_agent.as_tool(
            tool_name="news_scout_tool",
            tool_description="Use this tool to fetch trending news headlines and topics. Pass a topic to get relevant articles.",
        ),
        writer_agent.as_tool(
            tool_name="writer_tool",
            tool_description="Use this tool to write a news article and headline based on the given topic.",
        ),
        # research_agent (use OpenAI WebSearchTool),
        # image_finder_agent (use OpenAI WebSearchTool),
        # dalle_agent (Use Dalle to make a cover image),
        # database_agent (Upload the article to an SQL database),
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