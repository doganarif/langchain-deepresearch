"""
Example showing how to use the DeepResearcher with a custom LangChain model.
"""

import asyncio
from typing import List, Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage

from langchain_deepresearch import DeepResearcher


# Example of creating a custom LangChain model wrapper
class CustomLLMWrapper(BaseChatModel):
    """Example wrapper for a custom LLM API that implements the LangChain interface."""

    def __init__(self, api_key: str):
        """Initialize with your custom API key."""
        super().__init__()
        self.api_key = api_key

    def _generate(self, messages: List[BaseMessage], stop: List[str] = None, **kwargs) -> Any:
        """
        This is implemented for compatibility, but we'll use the async version.
        You would implement your API call here for synchronous usage.
        """
        raise NotImplementedError("Use async version instead")

    async def _agenerate(self, messages: List[BaseMessage], stop: List[str] = None, **kwargs) -> Any:
        """Implement the async call to your custom LLM API."""
        # This is where you would implement the actual API call
        # For example purposes, we'll just mock a response

        # Extract the prompt from messages
        prompt = ""
        for message in messages:
            if isinstance(message, SystemMessage):
                prompt += f"System: {message.content}\n\n"
            elif isinstance(message, HumanMessage):
                prompt += f"Human: {message.content}\n\n"
            else:
                prompt += f"{message.type}: {message.content}\n\n"

        # In a real implementation, you would call your API here
        # For example:
        # import aiohttp
        # async with aiohttp.ClientSession() as session:
        #     async with session.post(
        #         "https://your-llm-api.com/generate",
        #         json={"prompt": prompt, "api_key": self.api_key},
        #     ) as response:
        #         result = await response.json()
        #         response_text = result["generated_text"]

        # Mock a response for this example
        # In a real implementation, replace this with actual API call results
        from langchain_core.messages import AIMessage
        from langchain_core.outputs import ChatGeneration, ChatResult

        # Generate a mock response based on the input
        if "search queries" in prompt.lower():
            # Generate mock search queries
            response_text = """{"queries": [
                {"query": "recent breakthroughs in quantum computing 2024", "research_goal": "Find the latest advances"},
                {"query": "quantum computing impact on cryptography", "research_goal": "Understand security implications"}
            ]}"""
        elif "extract key learnings" in prompt.lower():
            # Generate mock search result analysis
            response_text = """{"learnings": [
                "Quantum computers are approaching quantum advantage for specific problem domains",
                "Several companies have announced quantum processors exceeding 100 qubits"
            ], "followUpQuestions": [
                "What are the practical applications of current quantum computers?"
            ]}"""
        else:
            # Default response for report generation
            response_text = "# Research Report\n\nThis is a mock research report that would be generated by your custom LLM."

        # Create an AIMessage with the response
        ai_message = AIMessage(content=response_text)

        # Wrap in ChatGeneration and ChatResult for LangChain compatibility
        chat_generation = ChatGeneration(message=ai_message)
        chat_result = ChatResult(generations=[chat_generation])

        return chat_result


async def main():
    # Create a custom LLM wrapper
    custom_llm = CustomLLMWrapper(api_key="your-custom-api-key")

    # Create the researcher with the custom LLM
    researcher = DeepResearcher(
        llm=custom_llm,
        google_api_key="your-google-api-key",  # Or use environment variable
        google_cx="your-google-cx-id",  # Or use environment variable
        max_time_seconds=1800,  # 30 minutes
        verbose=True
    )

    # Run the research with reduced parameters for testing
    result = await researcher.research(
        query="Advancements in AI for scientific discovery",
        breadth=2,  # Reduced for testing
        depth=1,  # Reduced for testing
        time_limit=600  # 10 minutes for testing
    )

    # Process results
    if result["success"]:
        print("Research completed successfully")
        print(f"Report length: {len(result['report'])} characters")
        print("\nReport preview:")
        print(result["report"][:500] + "...")
    else:
        print(f"Research failed: {result.get('error')}")


if __name__ == "__main__":
    asyncio.run(main())
