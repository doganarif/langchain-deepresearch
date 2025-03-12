"""
ResearchSearchService for handling web search operations with LangChain models.
"""

import asyncio
import logging
import json
import re
from typing import Dict, Any, List, Optional, Union
from urllib.parse import quote_plus

import aiohttp

# LangChain imports
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.language_models.llms import BaseLLM
from langchain_core.messages import SystemMessage, HumanMessage

logger = logging.getLogger(__name__)


class ResearchSearchService:
    """
    Service for handling web search operations for research with LangChain models.
    """

    # Default API settings - should be overridden with real keys
    GOOGLE_API_KEY = ""
    GOOGLE_CX = ""

    # Firecrawl configuration - use localhost for testing, update in production
    FIRECRAWL_URL = "http://127.0.0.1:3002/v1/scrape"
    FIRECRAWL_API_KEY = ""  # Add your key if needed

    def __init__(self):
        """Initialize the ResearchSearchService."""
        pass

    async def generate_serp_queries(
            self,
            query: str,
            client: Union[BaseChatModel, BaseLLM],
            num_queries: int,
            model: Optional[str] = None,
            learnings: List[str] = None,
            elapsed_seconds: float = 0,
            max_time_seconds: float = 1700,
            system_prompt: Optional[str] = None,
    ) -> List[Dict[str, str]]:
        """
        Generate search queries based on the main research question.

        Args:
            query: The main research question/topic
            client: LangChain model to use
            num_queries: Number of search queries to generate
            model: Optional model name (not used with LangChain)
            learnings: Optional list of previous learnings to avoid redundancy
            elapsed_seconds: Time elapsed since research started
            max_time_seconds: Maximum allowed research time
            system_prompt: Custom system prompt for query generation

        Returns:
            List of query objects with search queries and research goals
        """
        # Check time constraints - reduce number of queries if we're running short on time
        time_ratio = elapsed_seconds / max_time_seconds

        if time_ratio > 0.7:  # If we've used 70% of our time
            # Reduce number of queries to speed things up
            num_queries = max(1, min(num_queries, 2))

        # If we've used 90% of our time, just return a basic query
        if time_ratio > 0.9:
            return [{"query": query, "research_goal": "Quick information gathering"}]

        learnings_text = ""
        if learnings and len(learnings) > 0:
            # Format previous learnings to inform query generation
            learnings_sample = learnings[: min(3, len(learnings))]
            learnings_text = "\n".join(
                [f"- {learning}" for learning in learnings_sample]
            )
            learnings_text = f"\n\nPrevious learnings:\n{learnings_text}\n\nGenerate queries to find new information beyond these learnings."

        default_system_prompt = """You are an expert researcher who knows how to break down complex research topics into specific search queries.
        For each query, provide both the exact search query string and the specific research goal that query aims to address."""

        # Use custom system prompt if provided
        system_prompt = system_prompt or default_system_prompt

        user_prompt = f"""Research topic: {query}

        Generate {num_queries} distinct search queries to research this topic effectively.{learnings_text}

        Return your response in this JSON format:
        {{
            "queries": [
                {{
                    "query": "specific search query string",
                    "research_goal": "what this query aims to discover"
                }}
            ]
        }}

        Make the search queries specific, detailed, and varied to cover different aspects of the topic."""

        try:
            # Set a timeout for query generation (shorter as we approach the overall time limit)
            remaining_time_ratio = 1.0 - time_ratio
            query_timeout = max(5, int(15 * remaining_time_ratio))

            # Generate queries with timeout
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
            ]

            try:
                async with asyncio.timeout(query_timeout):
                    response = client.invoke(messages)
                    content = response.content
            except asyncio.TimeoutError:
                logger.warning(
                    f"Query generation timed out after {query_timeout}s, using default query"
                )
                return [
                    {"query": query, "research_goal": "General information gathering"}
                ]

            logger.info(f"Raw response from generate_serp_queries: {content}")

            # Parse JSON using regex and fallback mechanisms
            # Extract JSON with regex
            json_match = re.search(r"\{[\s\S]*\}", content)
            if not json_match:
                logger.warning("No JSON found in response from generate_serp_queries")
                # Fallback to simple queries if JSON parsing fails
                return [
                    {"query": query, "research_goal": "General information gathering"}
                ]

            json_str = json_match.group(0)
            logger.info(f"Extracted JSON string: {json_str}")

            try:
                data = json.loads(json_str)
                queries = data.get("queries", [])
                logger.info(f"Parsed queries: {queries}")

                if not isinstance(queries, list) or not queries:
                    raise ValueError("Parsed 'queries' is not a list or is empty")
                return queries

            except (json.JSONDecodeError, ValueError) as e:
                logger.error(f"Error parsing JSON or invalid format: {e}")
                return [
                    {"query": query, "research_goal": "General information gathering"}
                ]

        except Exception as e:
            logger.error(f"Error generating search queries: {str(e)}", exc_info=True)
            # Fallback to simple query
            return [{"query": query, "research_goal": "General information gathering"}]

    async def search_web(
            self,
            query: str,
            timeout: int = 10000,
            limit: int = 5,
            elapsed_seconds: float = 0,
            max_time_seconds: float = 1700,
    ) -> Dict[str, Any]:
        """
        Search the web using Google Programmable Search API and then enrich results with Firecrawl.

        Args:
            query: The search query
            timeout: Timeout in milliseconds
            limit: Maximum number of results to return
            elapsed_seconds: Time elapsed since research started
            max_time_seconds: Maximum allowed research time

        Returns:
            Dictionary with search results
        """
        # Check time constraints
        if elapsed_seconds > max_time_seconds:
            return {"data": [], "error": "Time limit reached"}

        logger.info(f"Searching web for: {query}")

        try:
            # Adjust timeout based on time remaining (shorter as we approach the time limit)
            remaining_time_ratio = 1.0 - (elapsed_seconds / max_time_seconds)
            adjusted_timeout = min(
                timeout, max(5000, int(timeout * remaining_time_ratio))
            )  # At least 5 seconds

            # Set timeout for the request (converting from ms to seconds)
            timeout_seconds = adjusted_timeout / 1000

            # 1. First, get search results from Google
            encoded_query = quote_plus(query)
            google_url = f"https://www.googleapis.com/customsearch/v1?key={self.GOOGLE_API_KEY}&cx={self.GOOGLE_CX}&q={encoded_query}&num={limit}"
            logger.info(f"Google Search URL: {google_url}")

            formatted_results = {"data": []}

            conn = aiohttp.TCPConnector(ssl=False)
            async with aiohttp.ClientSession(connector=conn) as session:
                try:
                    # Get Google search results with timeout
                    async with asyncio.timeout(timeout_seconds):
                        async with session.get(google_url) as response:
                            logger.info(
                                f"Got Google API response status: {response.status}"
                            )
                            if response.status == 200:
                                google_results = await response.json()
                                logger.info(f"Google API response: {google_results}")

                                if "items" not in google_results:
                                    logger.warning(
                                        f"No search results found for query: {query}"
                                    )
                                    return {
                                        "data": [],
                                        "error": "No search results found",
                                    }

                                # Decide whether to use Firecrawl based on time
                                # Use Firecrawl only if we have plenty of time left (first 60% of time budget)
                                use_firecrawl = (
                                        remaining_time_ratio > 0.4 and elapsed_seconds < 900
                                )  # Less than 15 minutes elapsed

                                if use_firecrawl:
                                    logger.info("Using Firecrawl enrichment")
                                else:
                                    logger.info("Skipping Firecrawl to save time")

                                # Process each result
                                for item in google_results["items"]:
                                    result_url = item.get("link", "")
                                    if not result_url or result_url.endswith(
                                            (".pdf", ".doc", ".xls")
                                    ):
                                        logger.info(
                                            f"Skipping non-HTML content: {result_url}"
                                        )
                                        continue

                                    # Basic result from Google
                                    result_item = {
                                        "title": item.get("title", "No title"),
                                        "url": result_url,
                                        "snippet": item.get("snippet", "No snippet"),
                                    }

                                    # Add pagemap data if available
                                    if "pagemap" in item:
                                        pagemap = item["pagemap"]
                                        if (
                                                "metatags" in pagemap
                                                and pagemap["metatags"]
                                        ):
                                            meta = pagemap["metatags"][0]
                                            if "og:description" in meta:
                                                result_item["snippet"] += (
                                                        " " + meta["og:description"]
                                                )

                                    # Try to enrich with Firecrawl if we have time
                                    if use_firecrawl:
                                        try:
                                            # More aggressive timeout for Firecrawl
                                            fc_timeout = max(
                                                5, int(timeout_seconds * 0.5)
                                            )

                                            # Make a request to Firecrawl to get the full content
                                            firecrawl_headers = {}
                                            if self.FIRECRAWL_API_KEY:
                                                firecrawl_headers["Authorization"] = f"Bearer {self.FIRECRAWL_API_KEY}"
                                            firecrawl_headers["Content-Type"] = "application/json"

                                            firecrawl_payload = {
                                                "url": result_url,
                                                "formats": ["markdown"],
                                                "timeout": int(
                                                    fc_timeout * 1000
                                                ),  # Provide timeout in milliseconds
                                            }

                                            async with asyncio.timeout(fc_timeout):
                                                async with session.post(
                                                        self.FIRECRAWL_URL,
                                                        json=firecrawl_payload,
                                                        headers=firecrawl_headers,
                                                ) as fc_response:
                                                    if fc_response.status == 200:
                                                        fc_data = (
                                                            await fc_response.json()
                                                        )
                                                        logger.info(
                                                            f"Firecrawl response: {fc_data}"
                                                        )
                                                        if fc_data.get(
                                                                "success"
                                                        ) and "markdown" in fc_data.get(
                                                            "data", {}
                                                        ):
                                                            markdown_content = fc_data[
                                                                "data"
                                                            ]["markdown"]
                                                            if not markdown_content.startswith(
                                                                    "Something went wrong"
                                                            ):
                                                                result_item[
                                                                    "content"
                                                                ] = markdown_content
                                                                logger.info(
                                                                    f"Firecrawl enrichment successful for {result_url}"
                                                                )
                                                            else:
                                                                logger.warning(
                                                                    f"Firecrawl returned error in markdown: {markdown_content}"
                                                                )
                                        except (
                                                asyncio.TimeoutError,
                                                aiohttp.ClientError,
                                        ) as fc_error:
                                            logger.warning(
                                                f"Firecrawl error for {result_url}: {str(fc_error)}"
                                            )
                                        except Exception as fc_error:
                                            logger.warning(
                                                f"Error enriching with Firecrawl: {str(fc_error)}"
                                            )

                                    formatted_results["data"].append(result_item)

                                logger.info(
                                    f"Returning {len(formatted_results['data'])} results"
                                )
                                return formatted_results
                            else:
                                error_text = await response.text()
                                logger.error(
                                    f"Google Search API error: {response.status}, {error_text}"
                                )
                                return {
                                    "data": [],
                                    "error": f"API error: {response.status}",
                                }

                except asyncio.TimeoutError:
                    logger.error(
                        f"Google Search API request timed out after {timeout_seconds} seconds"
                    )
                    return {"data": [], "error": "Request timed out"}
                except Exception as e:
                    logger.error(f"Error in Google Search: {e}")
                    return {"data": [], "error": str(e)}

        except Exception as e:
            logger.error(f"Error in search_web: {str(e)}", exc_info=True)
            return {"data": [], "error": str(e)}

    async def process_serp_result(
            self,
            query: str,
            search_result: Dict[str, Any],
            client: Union[BaseChatModel, BaseLLM],
            model: Optional[str] = None,
            elapsed_seconds: float = 0,
            max_time_seconds: float = 1700,
            system_prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Process search results to extract learnings and follow-up questions.

        Args:
            query: The search query
            search_result: Search results from search_web
            client: LangChain model to use
            model: Optional model name (not used with LangChain)
            elapsed_seconds: Time elapsed since research started
            max_time_seconds: Maximum allowed research time
            system_prompt: Custom system prompt for result analysis

        Returns:
            Dictionary with learnings and follow-up questions
        """
        # Check time constraints
        if elapsed_seconds > max_time_seconds:
            return {"learnings": [], "followUpQuestions": []}

        # Extract and format search results for the AI to process
        results_text = ""
        for i, item in enumerate(search_result.get("data", [])):
            title = item.get("title", "No title")
            snippet = item.get("snippet", "No snippet")
            url = item.get("url", "No URL")
            content = item.get("content", "")  # From Firecrawl enrichment

            results_text += f"Result {i + 1}:\nTitle: {title}\nURL: {url}\nSnippet: {snippet}\n"

            # Add content if available, but limit its length
            if content:
                # Truncate content to prevent tokens overload
                max_content_length = 5000  # Adjust based on your model's context window
                truncated_content = content[:max_content_length]
                if len(content) > max_content_length:
                    truncated_content += "... [content truncated]"
                results_text += f"Content:\n{truncated_content}\n"

            results_text += "\n"

        if not results_text:
            results_text = "No search results found."

        default_system_prompt = """You are a research assistant analyzing search results.
        Extract key learnings from the provided search results and suggest follow-up questions for deeper research."""

        # Use custom system prompt if provided
        system_prompt = system_prompt or default_system_prompt

        user_prompt = f"""Research query: {query}

        Search results:
        {results_text}

        Based on these search results:
        1. Extract 2-3 key learnings or facts related to the research query
        2. Suggest 1-2 follow-up questions for deeper research

        Return your response in this JSON format:
        {{
            "learnings": [
                "specific fact or learning from search results"
            ],
            "followUpQuestions": [
                "specific follow-up question for deeper research"
            ]
        }}"""

        try:
            # Calculate appropriate timeout based on time left
            remaining_time_ratio = 1.0 - (elapsed_seconds / max_time_seconds)
            processing_timeout = max(
                10, int(30 * remaining_time_ratio)
            )  # Between 10-30 seconds

            # Generate analysis with timeout
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
            ]

            try:
                async with asyncio.timeout(processing_timeout):
                    response = client.invoke(messages)
                    content = response.content
            except asyncio.TimeoutError:
                logger.warning(f"SERP processing timed out after {processing_timeout}s")
                # Extract basic learnings from snippets if timeout
                basic_learnings = []
                for item in search_result.get("data", []):
                    snippet = item.get("snippet", "")
                    if snippet and len(snippet) > 50:  # Only use substantial snippets
                        basic_learnings.append(
                            snippet[:150] + "..."
                        )  # Truncate long snippets

                if basic_learnings:
                    return {"learnings": basic_learnings[:3], "followUpQuestions": []}
                return {"learnings": [], "followUpQuestions": []}

            logger.info(f"Raw response from process_serp_result: {content}")

            # Parse JSON from response
            # Extract JSON with regex
            json_match = re.search(r"\{[\s\S]*\}", content)
            if not json_match:
                logger.warning("No JSON found in response from process_serp_result")
                # Return empty results if JSON parsing fails
                return {"learnings": [], "followUpQuestions": []}

            json_str = json_match.group(0)
            logger.info(f"Extracted JSON string: {json_str}")
            try:
                data = json.loads(json_str)
                learnings = data.get("learnings", [])
                follow_up_questions = data.get("followUpQuestions", [])

                if not isinstance(learnings, list) or not isinstance(
                        follow_up_questions, list
                ):
                    raise ValueError(
                        "Parsed 'learnings' or 'followUpQuestions' are not lists"
                    )

                logger.info(
                    f"Parsed learnings: {learnings}, follow-up questions: {follow_up_questions}"
                )

                return {
                    "learnings": learnings,
                    "followUpQuestions": follow_up_questions,
                }

            except (json.JSONDecodeError, ValueError) as e:
                logger.error(f"Error parsing JSON or invalid format: {e}")
                # Extract basic learnings from snippets if JSON parsing fails
                basic_learnings = []
                for item in search_result.get("data", []):
                    snippet = item.get("snippet", "")
                    if snippet:
                        basic_learnings.append(snippet[:150])  # Truncate long snippets

                if basic_learnings:
                    return {"learnings": basic_learnings[:2], "followUpQuestions": []}
            return {"learnings": [], "followUpQuestions": []}

        except Exception as e:
            logger.error(f"Error processing search results: {str(e)}", exc_info=True)
            return {"learnings": [], "followUpQuestions": []}
