"""
Main DeepResearcher class that provides the primary interface for LangChain integration.
"""

import os
import logging
from typing import Dict, List, Optional, Any, Union

# LangChain imports
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.language_models.llms import BaseLLM

# Local imports
from langchain_deepresearch.services.deep_research_service import DeepResearchService
from langchain_deepresearch.services.search_service import ResearchSearchService
from langchain_deepresearch.services.report_service import ResearchReportService

logger = logging.getLogger(__name__)


class DeepResearcher:
    """
    DeepResearcher provides autonomous research capabilities to any LangChain model.

    This class coordinates the research process using web searches, content analysis,
    and recursive exploration to generate comprehensive research reports.
    """

    def __init__(
            self,
            llm: Union[BaseChatModel, BaseLLM],
            google_api_key: Optional[str] = None,
            google_cx: Optional[str] = None,
            firecrawl_api_key: Optional[str] = None,
            firecrawl_url: Optional[str] = None,
            max_time_seconds: int = 2400,
            min_research_time_seconds: int = 180,
            min_learnings_required: int = 8,
            max_searches: int = 200,
            verbose: bool = False,
            system_prompts: Optional[Dict[str, str]] = None
    ):
        """
        Initialize the DeepResearcher with a LangChain model.

        Args:
            llm: Any LangChain-compatible chat model or language model
            google_api_key: Google Search API key (also can use GOOGLE_API_KEY env var)
            google_cx: Google Custom Search CX ID (also can use GOOGLE_CX env var)
            firecrawl_api_key: Optional Firecrawl API key for content extraction
            firecrawl_url: Optional custom Firecrawl URL
            max_time_seconds: Maximum research time in seconds (default: 2400 seconds/40 minutes)
            min_research_time_seconds: Minimum research time in seconds (default: 180 seconds/3 minutes)
            min_learnings_required: Minimum number of learnings required for a successful research
            max_searches: Maximum number of searches to perform
            verbose: Whether to output verbose logging
            system_prompts: Optional dictionary of custom system prompts for different research stages
        """
        # Set up logging
        logging_level = logging.DEBUG if verbose else logging.INFO
        logging.basicConfig(
            level=logging_level,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )

        # Initialize LLM client
        self.llm = llm

        # Initialize system prompts with defaults or provided prompts
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

        # Initialize services
        self.search_service = ResearchSearchService()
        self.report_service = ResearchReportService()
        self.research_service = DeepResearchService(
            client=llm,
            search_service=self.search_service,
            report_service=self.report_service,
            system_prompts=self.system_prompts
        )

        # Configure API keys
        self._configure_api_keys(
            google_api_key=google_api_key,
            google_cx=google_cx,
            firecrawl_api_key=firecrawl_api_key,
            firecrawl_url=firecrawl_url
        )

        # Configure research parameters
        self.research_service.max_time_seconds = max_time_seconds
        self.research_service.min_research_time_seconds = min_research_time_seconds
        self.research_service.min_learnings_required = min_learnings_required
        self.research_service.max_searches = max_searches

    def _configure_api_keys(
            self,
            google_api_key: Optional[str],
            google_cx: Optional[str],
            firecrawl_api_key: Optional[str],
            firecrawl_url: Optional[str]
    ) -> None:
        """
        Configure API keys for services, using parameters or environment variables.

        Args:
            google_api_key: Google Search API key
            google_cx: Google Custom Search CX ID
            firecrawl_api_key: Firecrawl API key
            firecrawl_url: Firecrawl URL

        Raises:
            ValueError: If required API keys are not provided
        """
        # For Google Search API - try parameters first, then environment variables
        gkey = google_api_key or os.environ.get("GOOGLE_API_KEY")
        gcx = google_cx or os.environ.get("GOOGLE_CX")

        if not gkey or not gcx:
            raise ValueError(
                "Google Search API key and CX ID are required. "
                "Provide them as parameters or set GOOGLE_API_KEY and GOOGLE_CX environment variables."
            )

        # Set API keys on search service
        self.search_service.GOOGLE_API_KEY = gkey
        self.search_service.GOOGLE_CX = gcx

        # Set Firecrawl settings if provided
        if firecrawl_api_key:
            self.search_service.FIRECRAWL_API_KEY = firecrawl_api_key
        if firecrawl_url:
            self.search_service.FIRECRAWL_URL = firecrawl_url

    async def research(
            self,
            query: str,
            breadth: int = 3,
            depth: int = 2,
            time_limit: Optional[int] = None,
            report_model: Optional[str] = None,
            min_learnings_required: Optional[int] = None,
            max_searches: Optional[int] = None,
            system_prompts: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Perform deep research on a topic.

        Args:
            query: The research query/topic
            breadth: Number of parallel searches (default: 3)
            depth: How deep to recursively explore the topic (default: 2)
            time_limit: Maximum research time in seconds (overrides instance setting)
            report_model: Optional model name to use for report generation
            min_learnings_required: Minimum learnings required (overrides instance setting)
            max_searches: Maximum searches to perform (overrides instance setting)
            system_prompts: Optional dictionary of custom system prompts for this research only

        Returns:
            Dictionary with research results:
            {
                "success": bool,      # Whether research was successful
                "report": str,        # Markdown research report
                "learnings": list,    # List of learnings discovered
                "visited_urls": list, # List of URLs consulted
                "total_time": int,    # Time spent researching in seconds
                "early_completion": bool  # Whether completed early
            }

            Or on failure:
            {
                "success": False,
                "error": str,         # Error message
                "message": str        # User-friendly error message
            }
        """
        # Apply overrides for this specific research
        original_settings = {}

        try:
            # Handle time limit override
            if time_limit is not None:
                original_settings["max_time_seconds"] = self.research_service.max_time_seconds
                self.research_service.max_time_seconds = time_limit

            # Handle min learnings override
            if min_learnings_required is not None:
                original_settings["min_learnings_required"] = self.research_service.min_learnings_required
                self.research_service.min_learnings_required = min_learnings_required

            # Handle max searches override
            if max_searches is not None:
                original_settings["max_searches"] = self.research_service.max_searches
                self.research_service.max_searches = max_searches

            # Apply per-research system prompts if provided
            original_prompts = None
            if system_prompts:
                original_prompts = self.research_service.system_prompts.copy()
                self.research_service.system_prompts.update(system_prompts)

            # Run the research
            result = await self.research_service.deep_research(
                query=query,
                breadth=breadth,
                depth=depth,
                model=report_model
            )

            # Restore original prompts if they were modified
            if original_prompts:
                self.research_service.system_prompts = original_prompts

            return result

        finally:
            # Restore original settings
            for key, value in original_settings.items():
                setattr(self.research_service, key, value)

    async def quick_research(
            self,
            query: str,
            time_limit: int = 300,
            system_prompts: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Perform a quick research on a topic with reduced parameters.

        Args:
            query: The research query/topic
            time_limit: Maximum research time in seconds (default: 300 seconds/5 minutes)
            system_prompts: Optional custom system prompts for this quick research

        Returns:
            Same return format as research()
        """
        return await self.research(
            query=query,
            breadth=2,
            depth=1,
            time_limit=time_limit,
            min_learnings_required=3,
            system_prompts=system_prompts
        )
