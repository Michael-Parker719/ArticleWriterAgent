from newsapi import NewsApiClient
import os
from dotenv import load_dotenv
load_dotenv()

NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")
newsapi = NewsApiClient(api_key=NEWSAPI_KEY)

def main():
    print("Fetching trending topics...")
    # Fetch trending topics using NewsAPI
    headlines = newsapi.get_everything(q="AI", language='en', page_size=3)
    articles = headlines.get('articles', [])

    if not articles:
        print("No articles found.")
        return

    formatted = []
    for article in articles:
        title = article.get('title')
        description = article.get('description')
        url = article.get('url')
        if title and description and url:
            formatted.append(f"Title: {title}\nDescription: {description}\nURL: {url}")

    result = "\n\n".join(formatted)
    print(result)

if __name__ == "__main__":
    main()