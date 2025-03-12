"""
Basic usage examples for langchain-deepresearch.

This script demonstrates how to use the DeepResearcher with
different LangChain models.
"""

import asyncio
import os
import argparse
from typing import Optional

from langchain_deepresearch import DeepResearcher


async def run_research(
        query: str,
        model_type: str = "openai",
        model_name: Optional[str] = None,
        breadth: int = 3,
        depth: int = 2,
        time_limit: Optional[int] = None,
        output_file: Optional[str] = None
):
    """
    Run research with the specified model type and parameters.

    Args:
        query: Research query
        model_type: Model provider ('openai', 'anthropic', 'google', etc.)
        model_name: Specific model name (if None, uses provider's default)
        breadth: Search breadth
        depth: Research depth
        time_limit: Maximum time in seconds
        output_file: Optional file to save the report
    """
    # Initialize the appropriate LLM based on model_type
    llm = None

    if model_type == "openai":
        from langchain_openai import ChatOpenAI

        # Default to gpt-3.5-turbo if no model specified
        model = model_name or "gpt-3.5-turbo"
        llm = ChatOpenAI(model=model)

    elif model_type == "anthropic":
        from langchain_anthropic import ChatAnthropic

        # Default to claude-3-haiku if no model specified
        model = model_name or "claude-3-haiku-20240307"
        llm = ChatAnthropic(model=model)

    elif model_type == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI

        # Default to gemini-pro if no model specified
        model = model_name or "gemini-pro"
        llm = ChatGoogleGenerativeAI(model=model)

    elif model_type == "huggingface":
        from langchain_huggingface import HuggingFaceEndpoint

        # Default to Mistral if no model specified
        repo_id = model_name or "mistralai/Mistral-7B-Instruct-v0.2"
        llm = HuggingFaceEndpoint(repo_id=repo_id)

    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    print(f"Initialized {model_type} model: {llm}")

    # Create the researcher with the LLM
    researcher = DeepResearcher(
        llm=llm,
        # API keys can be provided directly or through environment variables
        google_api_key=os.environ.get("GOOGLE_API_KEY"),
        google_cx=os.environ.get("GOOGLE_CX"),
        verbose=True
    )

    print(f"Starting research: '{query}'")
    print(f"Parameters: breadth={breadth}, depth={depth}, time_limit={time_limit or 'default'}")

    # Run the research
    start_time = asyncio.get_event_loop().time()
    result = await researcher.research(
        query=query,
        breadth=breadth,
        depth=depth,
        time_limit=time_limit
    )
    elapsed = asyncio.get_event_loop().time() - start_time

    # Process the results
    if result["success"]:
        print(f"\n✅ Research completed in {elapsed:.1f} seconds")

        if result.get("early_completion"):
            print("⏰ Research completed early with relevant findings")

        # Print statistics
        print(f"🔎 Consulted {len(result.get('visited_urls', []))} sources")
        print(f"📚 Gathered {len(result.get('learnings', []))} insights")

        # Save or display report
        if output_file:
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(result["report"])
            print(f"📄 Report saved to {output_file}")
        else:
            print("\n--- RESEARCH REPORT ---\n")
            print(result["report"])
    else:
        print(f"\n❌ Research failed: {result.get('error')}")
        if result.get("message"):
            print(result["message"])


async def main():
    parser = argparse.ArgumentParser(description="Run research with langchain-deepresearch")
    parser.add_argument("query", help="Research query")
    parser.add_argument("--model-type", choices=["openai", "anthropic", "google", "huggingface"],
                        default="openai", help="Model provider")
    parser.add_argument("--model-name", help="Specific model name")
    parser.add_argument("--breadth", type=int, default=3, help="Search breadth")
    parser.add_argument("--depth", type=int, default=2, help="Research depth")
    parser.add_argument("--time-limit", type=int, help="Time limit in seconds")
    parser.add_argument("--output", help="Output file for report")
    args = parser.parse_args()

    await run_research(
        query=args.query,
        model_type=args.model_type,
        model_name=args.model_name,
        breadth=args.breadth,
        depth=args.depth,
        time_limit=args.time_limit,
        output_file=args.output
    )


if __name__ == "__main__":
    asyncio.run(main())
