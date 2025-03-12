"""
DeepResearchService coordinates the research process for LangChain models.
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Union

# LangChain imports
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.language_models.llms import BaseLLM

logger = logging.getLogger(__name__)


class DeepResearchService:
    """Coordinator service for performing deep research on user queries with LangChain models."""

    def __init__(
            self,
            client: Union[BaseChatModel, BaseLLM],
            search_service=None,
            report_service=None,
            system_prompts: Optional[Dict[str, str]] = None
    ):
        """
        Initialize the DeepResearchService.

        Args:
            client: LangChain model to use for research
            search_service: Search service to use (created if not provided)
            report_service: Report service to use (created if not provided)
            system_prompts: Optional dictionary of custom system prompts
        """
        # Import here to avoid circular imports
        from langchain_deepresearch.services.search_service import ResearchSearchService
        from langchain_deepresearch.services.report_service import ResearchReportService

        self.search_service = search_service or ResearchSearchService()
        self.report_service = report_service or ResearchReportService()
        self.ai_client = client

        # Configuration - Enhanced for deeper research
        self.total_searches = 0
        self.max_searches = 200  # Increased from 100
        self.start_time = None
        self.max_time_seconds = 2400  # Increased from 1700 to 40 minutes
        self.min_research_time_seconds = 180  # Increased from 120 to 3 minutes
        self.min_learnings_required = 8  # Increased from 5

        # System prompts for different research stages
        self.system_prompts = {
            "query_generation": """You are an expert researcher who knows how to break down complex research topics into specific search queries.
                For each query, provide both the exact search query string and the specific research goal that query aims to address.""",
            "result_analysis": """You are a research assistant analyzing search results.
                Extract key learnings from the provided search results and suggest follow-up questions for deeper research.""",
            "report_generation": """You are an expert research report writer with extensive experience in analysis.
                Create a comprehensive, well-structured research report that synthesizes all the provided information
                with proper citations. Your report should be balanced and objectively assess the topic."""
        }

        # Update with custom prompts if provided
        if system_prompts:
            self.system_prompts.update(system_prompts)

    async def deep_research(
            self,
            query: str,
            breadth: int,
            depth: int,
            client=None,
            model: str = None,
    ) -> Dict[str, Any]:
        """
        Main research function that recursively explores a topic.

        Args:
            query: Research query/topic
            breadth: Number of parallel searches to perform
            depth: How many levels deep to research
            client: LangChain model (defaults to instance client)
            model: Model name (for report generation, optional)

        Returns:
            Dictionary with research results
        """
        # Initialize time tracking
        self.start_time = time.time()

        # Use provided client or class client
        client = client or self.ai_client

        # Initialize results
        learnings = []
        visited_urls = []
        self.total_searches = 0  # Reset counter for each research session

        try:
            # Start research with time awareness
            research_result = await self._time_aware_research(
                query=query,
                breadth=breadth,
                depth=depth,
                client=client,
                model=model,
                learnings=learnings,
                visited_urls=visited_urls,
            )

            # If we have enough results, generate the report
            if research_result and (
                    len(research_result["learnings"]) >= 3
                    or time.time() - self.start_time > self.min_research_time_seconds
            ):
                # Generate final report, with early completion note if needed
                early_completion_note = ""
                if research_result.get("early_completion"):
                    early_completion_note = (
                        "\n\n> **Note**: This research was completed early with the most relevant findings "
                        "to stay within the time limit. For more comprehensive research, try a more focused query "
                        "or adjust the depth/breadth settings."
                    )

                report = await self.report_service.write_final_report(
                    prompt=query,
                    learnings=research_result["learnings"],
                    visited_urls=research_result["visited_urls"],
                    client=client,
                    model=model,
                    early_completion=research_result.get("early_completion", False),
                    max_time_seconds=self.max_time_seconds,
                    start_time=self.start_time,
                    system_prompt=self.system_prompts.get("report_generation")
                )

                # Add early completion note if necessary
                if early_completion_note and research_result.get("early_completion"):
                    report += early_completion_note

                # Return successful result with report
                return {
                    "success": True,
                    "report": report,
                    "learnings": research_result["learnings"],
                    "visited_urls": research_result["visited_urls"],
                    "total_time": int(time.time() - self.start_time),
                    "early_completion": research_result.get("early_completion", False)
                }
            else:
                # Not enough results
                error_message = "Research couldn't find enough relevant information. Please try a more specific query."
                return {
                    "success": False,
                    "error": "Not enough results",
                    "message": error_message
                }

        except asyncio.TimeoutError:
            logger.error(f"Research timed out")
            return await self._handle_timeout(
                query, learnings, visited_urls, client, model
            )
        except Exception as e:
            logger.error(
                f"Error in deep research: {str(e)}",
                exc_info=True,
            )
            return {"success": False, "error": str(e)}

    def _should_continue_research(
            self, learnings: Optional[List[str]], elapsed_seconds: float
    ) -> bool:
        """Determine if research should continue based on time and data collected."""
        # Safely handle None by converting to empty list
        safe_learnings = learnings or []

        # Always research for at least the minimum time
        if elapsed_seconds < self.min_research_time_seconds:
            return True

        # Check if we're approaching the hard time limit
        if elapsed_seconds > self.max_time_seconds:
            return False

        # If we have enough learnings and have spent some time, we can finish
        if (
                len(safe_learnings) >= self.min_learnings_required
                and elapsed_seconds > self.min_research_time_seconds
        ):
            # The more learnings we have, the more willing we are to stop early
            if (
                    len(safe_learnings) > 20 and elapsed_seconds > 600
            ):  # 10+ minutes with 20+ learnings
                return False
            if (
                    len(safe_learnings) > 10 and elapsed_seconds > 900
            ):  # 15+ minutes with 10+ learnings
                return False

        # Continue research if no stopping condition met
        return True

    async def _time_aware_research(
            self,
            query: str,
            breadth: int,
            depth: int,
            client,
            model: str,
            learnings: List[str] = None,
            visited_urls: List[str] = None,
            current_depth: int = 0,
    ) -> Dict[str, Any]:
        """
        Time-aware recursive function to explore a topic in depth.
        Monitors elapsed time and can return early with partial results.
        """
        learnings = learnings or []
        visited_urls = visited_urls or []
        max_depth_reached = current_depth

        # Check elapsed time at the start of each recursive call
        elapsed_seconds = time.time() - self.start_time
        if elapsed_seconds > self.max_time_seconds:
            logger.info(
                f"Time limit approaching ({elapsed_seconds:.1f}s), finishing research early"
            )
            return {
                "learnings": learnings,
                "visited_urls": visited_urls,
                "early_completion": True,
                "max_depth_reached": max_depth_reached,
            }

        # Log progress
        progress = f"Researching at depth {current_depth + 1}/{depth}, breadth {breadth}, elapsed: {elapsed_seconds:.1f}s"
        logger.info(progress)

        # Check if we should continue research based on time and data collected
        if not self._should_continue_research(learnings, elapsed_seconds):
            logger.info(
                f"Sufficient data collected ({len(learnings)} learnings) in {elapsed_seconds:.1f}s, finishing early"
            )
            return {
                "learnings": learnings,
                "visited_urls": visited_urls,
                "early_completion": True,
                "max_depth_reached": max_depth_reached,
            }

        # Adjust breadth based on time constraints
        adaptive_breadth = breadth
        if elapsed_seconds > (
                self.max_time_seconds * 0.5
        ):  # If past halfway point in time
            adaptive_breadth = max(1, breadth // 2)  # Reduce breadth
            logger.info(
                f"Reducing breadth to {adaptive_breadth} due to time constraints"
            )

        try:
            # Generate search queries using the search service
            serp_queries = await self.search_service.generate_serp_queries(
                query=query,
                model=model,
                client=client,
                num_queries=adaptive_breadth,
                learnings=learnings,
                elapsed_seconds=elapsed_seconds,
                max_time_seconds=self.max_time_seconds,
                system_prompt=self.system_prompts.get("query_generation")
            )

            # Process each query
            all_new_learnings = []
            all_new_urls = []

            # Process each query
            for query_obj in serp_queries:
                # Check time again before each search
                if time.time() - self.start_time > self.max_time_seconds:
                    break

                # Search for content
                serp_query = query_obj["query"]
                search_result = await self.search_service.search_web(
                    serp_query,
                    timeout=30000,
                    limit=3,
                    elapsed_seconds=time.time() - self.start_time,
                    max_time_seconds=self.max_time_seconds,
                )

                # Collect new URLs
                new_urls = [
                    item.get("url")
                    for item in search_result.get("data", [])
                    if item.get("url")
                ]
                all_new_urls.extend(new_urls)

                # Process search results
                result = await self.search_service.process_serp_result(
                    query=serp_query,
                    search_result=search_result,
                    model=model,
                    client=client,
                    elapsed_seconds=time.time() - self.start_time,
                    max_time_seconds=self.max_time_seconds,
                    system_prompt=self.system_prompts.get("result_analysis")
                )

                all_new_learnings.extend(result.get("learnings", []))

                # Check if we have enough data already
                if len(learnings or []) + len(all_new_learnings) >= 100:
                    logger.info(
                        f"Collected 100+ learnings, may finish early if time constraints"
                    )

            # If we need to go deeper and have time and follow-up questions
            deeper_result = None
            if (
                    current_depth < depth - 1
                    and time.time() - self.start_time
                    < self.max_time_seconds * 0.75  # Only go deeper if < 75% of time used
                    and len(all_new_learnings) > 0
            ):

                # Create a simpler follow-up query
                next_query = f"""
                    Previous research: {query}
                    Follow-up: Learn more about {", ".join(all_new_learnings[:2])}
                    """.strip()

                # Calculate new breadth for next level - decrease as we go deeper
                new_breadth = max(1, adaptive_breadth // 2)

                # Recursive call with a time check
                try:
                    deeper_result = await self._time_aware_research(
                        query=next_query,
                        breadth=new_breadth,
                        depth=depth,
                        client=client,
                        model=model,
                        learnings=(learnings or []) + all_new_learnings,
                        visited_urls=(visited_urls or []) + all_new_urls,
                        current_depth=current_depth + 1,
                    )
                except Exception as deep_err:
                    logger.error(f"Error in deeper research: {str(deep_err)}")
                    deeper_result = {
                        "learnings": [],
                        "visited_urls": [],
                        "early_completion": True,
                        "max_depth_reached": current_depth + 1,
                    }

                if deeper_result:
                    max_depth_reached = max(
                        max_depth_reached, deeper_result.get("max_depth_reached", 0)
                    )
                    all_new_learnings = list(
                        set(all_new_learnings + deeper_result.get("learnings", []))
                    )
                    all_new_urls = list(
                        set(all_new_urls + deeper_result.get("visited_urls", []))
                    )

            # Final results
            combined_learnings = list(set((learnings or []) + all_new_learnings))
            combined_urls = list(set((visited_urls or []) + all_new_urls))

            early_completion = False
            if deeper_result and deeper_result.get("early_completion", False):
                early_completion = True
            elif time.time() - self.start_time > self.max_time_seconds * 0.75:
                early_completion = True

            return {
                "learnings": combined_learnings,
                "visited_urls": combined_urls,
                "early_completion": early_completion,
                "max_depth_reached": max(current_depth + 1, max_depth_reached),
            }
        except Exception as e:
            logger.error(f"Error in _time_aware_research: {str(e)}", exc_info=True)
            # Return what we have so far instead of losing all progress
            return {
                "learnings": learnings or [],  # Ensure we return an empty list if None
                "visited_urls": visited_urls
                                or [],  # Ensure we return an empty list if None
                "early_completion": True,
                "error": str(e),
                "max_depth_reached": max_depth_reached,
            }

    async def _handle_timeout(
            self,
            query: str,
            learnings: Optional[List[str]],
            visited_urls: Optional[List[str]],
            client,
            model: str,
    ) -> Dict[str, Any]:
        """Handle timeout by creating a report with collected data."""
        # Even if we timed out, we might have collected some useful information
        if learnings and len(learnings) > 0:
            # Generate a simple report with what we have
            report = await self.report_service.write_final_report(
                prompt=query,
                learnings=learnings,
                visited_urls=visited_urls,
                client=client,
                model=model,
                early_completion=True,
                max_time_seconds=self.max_time_seconds,
                start_time=self.start_time,
                system_prompt=self.system_prompts.get("report_generation")
            )

            # Add timeout note
            timeout_note = (
                "\n\n> **Note**: This research reached the maximum time limit. "
                "The findings above represent what was discovered in the available time. "
                "For more comprehensive research, try a more specific query or reduce the depth/breadth settings."
            )
            report += timeout_note

            # Return what we have with timeout info
            return {
                "success": True,
                "report": report,
                "learnings": learnings or [],
                "visited_urls": visited_urls or [],
                "reason": "timeout",
                "total_time": int(time.time() - self.start_time)
            }
        else:
            # No learnings collected, send error message
            error_message = (
                "Research timed out after 30 minutes without finding useful information. "
                "Please try a more specific query or reduce the depth/breadth settings."
            )
            return {
                "success": False,
                "error": "Research timed out",
                "message": error_message
            }
