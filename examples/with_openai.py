"""
Example showing how to use the DeepResearcher with OpenAI models.
"""

import asyncio
import os

from langchain_openai import ChatOpenAI
from langchain_deepresearch import DeepResearcher


async def main():
    # Set up your API key
    os.environ["OPENAI_API_KEY"] = "your-openai-api-key"  # Or set in your environment

    # Create an OpenAI LangChain model
    llm = ChatOpenAI(
        model="gpt-3.5-turbo",  # You can also use "gpt-4" for more complex research
        temperature=0.2,        # Lower temperature for more factual responses
    )

    # Create the researcher
    researcher = DeepResearcher(
        llm=llm,
        google_api_key="your-google-api-key",  # Or use environment variable
        google_cx="your-google-cx-id",         # Or use environment variable
        max_time_seconds=1800,  # 30 minutes
        verbose=True
    )

    # Run the research
    result = await researcher.research(
        query="Latest advancements in fusion energy technology",
        breadth=3,  # Number of parallel search paths
        depth=2,    # How deep to explore recursively
    )

    # Process results
    if result["success"]:
        # Save the report to a file
        with open("fusion_energy_report.md", "w", encoding="utf-8") as f:
            f.write(result["report"])

        print(f"Research completed with {len(result['learnings'])} insights")
        print(f"Report saved to fusion_energy_report.md")
        print("\nSample learnings:")
        for i, learning in enumerate(result["learnings"][:5], 1):
            print(f"{i}. {learning}")
    else:
        print(f"Research failed: {result.get('error')}")


if __name__ == "__main__":
    asyncio.run(main())
