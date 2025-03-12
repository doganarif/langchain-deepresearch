"""
ResearchReportService for generating research reports with any LangChain model.
"""

import asyncio
import json
import logging
from typing import List, Dict, Optional, Any, Union

# LangChain imports
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.language_models.llms import BaseLLM
from langchain_core.messages import SystemMessage, HumanMessage

logger = logging.getLogger(__name__)


class ResearchReportService:
    """Service for generating research reports using LangChain models."""

    def __init__(self):
        """Initialize the ResearchReportService."""
        pass

    async def write_final_report(
            self,
            prompt: str,
            learnings: List[str],
            visited_urls: List[str],
            client: Union[BaseChatModel, BaseLLM],
            model: Optional[str] = None,
            early_completion: bool = False,
            conversation_history: Optional[List[Dict]] = None,
            max_time_seconds: float = 2400,
            start_time: float = 0,
            system_prompt: Optional[str] = None,
    ) -> str:
        """
        Generate final report based on research learnings with citations.

        Args:
            prompt: Original research query
            learnings: List of insights and facts discovered during research
            visited_urls: List of URLs consulted
            client: LangChain model to use for report generation
            model: Optional model name override (not used with LangChain)
            early_completion: Whether research completed early
            conversation_history: Optional conversation context
            max_time_seconds: Maximum time allowed for research
            start_time: When research started
            system_prompt: Custom system prompt for report generation

        Returns:
            Markdown formatted research report
        """
        # Check if we have any learnings, if not return a basic message
        if not learnings:
            return "I wasn't able to gather sufficient information on this topic. Please try a more specific query."

        # Limit the learnings if we have too many
        if len(learnings) > 40:
            # Sort learnings by length (assuming longer ones have more detail)
            sorted_learnings = sorted(learnings, key=len, reverse=True)
            learnings = sorted_learnings[:40]

        # Match learnings to source URLs when possible
        # This is a simplified approach - for a real implementation, you'd want to track which URL each learning came from
        learning_with_sources = []
        for i, learning in enumerate(learnings):
            source_index = i % len(visited_urls) if visited_urls else None
            source_url = visited_urls[source_index] if source_index is not None and source_index < len(visited_urls) else None
            learning_with_sources.append({
                "learning": learning,
                "source": source_url
            })

        learnings_string = "\n".join([
            f'<learning source="{item["source"]}">\n{item["learning"]}\n</learning>'
            if item["source"] else
            f'<learning>\n{item["learning"]}\n</learning>'
            for item in learning_with_sources
        ])

        # Modify the prompt if this was an early completion
        completion_note = ""
        if early_completion:
            completion_note = (
                "Note: The research had to be completed early due to time constraints. "
                "The report should acknowledge this and focus on summarizing the most important "
                "information gathered so far."
            )

        # Prepare conversation context if available
        context_info = ""
        if conversation_history and len(conversation_history) > 0:
            # Get up to last 5 messages for context
            recent_messages = (
                conversation_history[-5:]
                if len(conversation_history) > 5
                else conversation_history
            )
            history_text = "\n".join(
                [
                    (
                        f"{'User' if msg['role'] == 'user' else 'Assistant'}: {msg['content'][:200]}..."
                        if len(msg["content"]) > 200
                        else f"{'User' if msg['role'] == 'user' else 'Assistant'}: {msg['content']}"
                    )
                    for msg in recent_messages
                ]
            )
            context_info = (
                f"\n\n<conversation_context>\nThis research is part of an ongoing conversation. "
                f"Consider this context when writing the report:\n{history_text}\n</conversation_context>\n\n"
            )

        user_prompt = (
            f"Given the following prompt from the user, write a final research report on the topic using "
            f"the learnings from research. Return a detailed markdown report with citations. "
            f"Each learning may include a source URL in the markup.\n\n<prompt>{prompt}</prompt>\n\n"
            f"Here are all the learnings from research with their sources when available:\n\n<learnings>\n{learnings_string}\n</learnings>\n\n"
            f"Sources consulted:\n{json.dumps(visited_urls)}\n\n"
            f"{context_info}"
            f"{completion_note}"
        )

        default_system_prompt = """You are an expert research report writer with extensive experience in analysis.
Create a comprehensive, well-structured research report that synthesizes all the provided information
with proper citations. Your report should be balanced and objectively assess the topic.

Your report should include:
- An executive summary that presents a balanced assessment of the topic
- A detailed main section organized by topics or themes
- Visual elements like tables, bullet points, and hierarchical headers
- Proper citations for facts and claims using [1], [2], etc. superscript notation
- A reference list or bibliography at the end of the report
- A conclusion that summarizes key insights

When drafting the report:
- Present both supporting and conflicting information with equal attention
- Distinguish between verified facts, claims, and your analysis
- Highlight gaps in the available information
- Assess limitations and reliability of sources objectively
- Focus on information most relevant to the research query
- Present a balanced assessment of the topic
- Use precise, specific language rather than superlatives or generalizations
- Maintain a professional, analytical tone throughout

When source URLs are provided for specific facts, cite them appropriately. When a fact doesn't 
have a specific source, cite it as "Based on multiple sources" or use the most relevant source.

Remember that this report will be used to make informed decisions - be thorough, balanced, and 
evidence-based in your assessment.
"""

        # Use custom system prompt if provided
        system_prompt_text = system_prompt or default_system_prompt

        try:
            # Set a reasonable timeout for report generation (60 seconds)
            current_time = asyncio.get_event_loop().time() if start_time > 0 else 0
            elapsed_seconds = current_time - start_time if start_time > 0 else 0

            timeout_seconds = min(90, max_time_seconds - elapsed_seconds - 30)
            timeout_seconds = max(30, timeout_seconds)  # At least 30 seconds

            # Set a task for report generation with timeout
            try:
                async with asyncio.timeout(timeout_seconds):
                    # Create messages for LangChain model
                    messages = [
                        SystemMessage(content=system_prompt_text),
                        HumanMessage(content=user_prompt),
                    ]

                    # Generate response
                    response = client.invoke(messages)
                    report = response.content
            except asyncio.TimeoutError:
                logger.warning(
                    f"Report generation timed out after {timeout_seconds}s, using simplified report"
                )
                # If timeout, generate a simpler report with just the learnings
                report = f"# Research Report: {prompt}\n\n"

                # Add context acknowledgment if we have conversation history
                if conversation_history and len(conversation_history) > 0:
                    report += "This report continues our earlier conversation about this topic.\n\n"

                report += "## Key Findings\n\n"
                for i, item in enumerate(learning_with_sources[:15], 1):
                    source_ref = f" [Source: {item['source']}]" if item["source"] else ""
                    report += f"{i}. {item['learning']}{source_ref}\n\n"

                if len(learning_with_sources) > 15:
                    report += "## Additional Information\n\n"
                    for i, item in enumerate(learning_with_sources[15:], 16):
                        source_ref = f" [Source: {item['source']}]" if item["source"] else ""
                        report += f"{i}. {item['learning']}{source_ref}\n\n"

            if not report:
                # Fallback if report generation failed
                report = f"# Research Report: {prompt}\n\n"
                report += "## Key Findings\n\n"
                for i, item in enumerate(learning_with_sources, 1):
                    source_ref = f" [Source: {item['source']}]" if item["source"] else ""
                    report += f"{i}. {item['learning']}{source_ref}\n\n"

            # Append sources in a bibliography section
            if visited_urls:
                source_count = min(30, len(visited_urls))
                urls_section = "\n\n## References\n\n"
                for i, url in enumerate(visited_urls[:source_count], 1):
                    urls_section += f"{i}. {url}\n"
                if len(visited_urls) > source_count:
                    urls_section += f"\nPlus {len(visited_urls) - source_count} additional sources consulted."
                report += urls_section

            return report

        except Exception as e:
            logger.error(f"Error generating final report: {str(e)}", exc_info=True)
            # Create a basic report with the findings if the full report generation fails
            basic_report = f"# Research Findings on {prompt}\n\n"
            basic_report += (
                "Due to technical issues, I've compiled a simple list of findings with sources where available:\n\n"
            )
            for i, item in enumerate(learning_with_sources, 1):
                source_ref = f" [Source: {item['source']}]" if item["source"] else ""
                basic_report += f"{i}. {item['learning']}{source_ref}\n\n"

            if visited_urls:
                basic_report += "\n\n## Sources\n\n"
                for i, url in enumerate(visited_urls[:20], 1):
                    basic_report += f"{i}. {url}\n"

            return basic_report

    def _extract_json(self, text: str) -> str:
        """Extract JSON object from text."""
        try:
            # Find JSON-like pattern
            start_idx = text.find("{")
            if start_idx == -1:
                # Try with array
                start_idx = text.find("[")
                if start_idx == -1:
                    return ""

            # Find matching closing bracket
            bracket_count = 0
            in_string = False
            escape_next = False

            for i in range(start_idx, len(text)):
                char = text[i]

                if escape_next:
                    escape_next = False
                    continue

                if char == "\\":
                    escape_next = True
                elif char == '"' and not escape_next:
                    in_string = not in_string
                elif not in_string:
                    if char in "{[":
                        bracket_count += 1
                    elif char in "}]":
                        bracket_count -= 1
                        if bracket_count == 0:
                            # Found complete JSON
                            return text[start_idx : i + 1]

            # If we get here, JSON might be malformed
            return text[start_idx:]

        except Exception as e:
            logger.error(f"Error extracting JSON: {str(e)}", exc_info=True)
            return ""
