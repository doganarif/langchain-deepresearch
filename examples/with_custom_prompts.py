"""
Example showing how to use the DeepResearcher with custom system prompts.
"""

import asyncio
import os

from langchain_openai import ChatOpenAI
from langchain_deepresearch import DeepResearcher


async def main():
    """Example showing how to use custom system prompts for all research."""

    # Set up your API key
    os.environ["OPENAI_API_KEY"] = "your-openai-api-key"  # Or set in your environment

    # Custom system prompts to guide the research process
    custom_prompts = {
        # Prompt for generating search queries
        "query_generation": """You are a venture capital analyst researching a market opportunity.
        Create specific search queries to gather competitive intelligence, market size data, 
        growth trends, and regulatory concerns. For each query, provide both the exact search 
        string and what specific data that query aims to discover.""",

        # Prompt for analyzing search results
        "result_analysis": """You are a venture capital analyst evaluating a potential investment.
        Extract key financial data, competitive advantages, market positioning, and risk factors 
        from these search results. Be particularly attentive to:
        1. Market size and growth projections
        2. Competitive landscape and barriers to entry
        3. Revenue models and unit economics
        4. Regulatory considerations and potential obstacles""",

        # Prompt for generating the final report
        "report_generation": """You are a senior investment analyst at a top venture capital firm.
        Create a comprehensive investment analysis report synthesizing all research findings.
        
        Your report must include:
        1. Executive Summary with clear investment recommendation
        2. Market Analysis with size, growth rate, and trends
        3. Competitive Landscape assessment
        4. Risk Analysis covering market, execution, and regulatory risks
        5. Financial Projections and potential returns
        6. Final Recommendation with confidence level and key metrics to track
        
        Use a professional, data-driven tone appropriate for investment committee review.
        Support all assertions with specific data points and proper citations."""
    }

    # Create an OpenAI LangChain model
    llm = ChatOpenAI(model="gpt-4")

    # Create the researcher with custom system prompts
    researcher = DeepResearcher(
        llm=llm,
        google_api_key=os.environ.get("GOOGLE_API_KEY"),
        google_cx=os.environ.get("GOOGLE_CX"),
        system_prompts=custom_prompts,
        verbose=True
    )

    # Run the research with a venture capital focused query
    result = await researcher.research(
        query="Market opportunity for carbon capture technology startups in 2025",
        breadth=3,
        depth=2,
        time_limit=1800  # 30 minutes
    )

    # Process results
    if result["success"]:
        # Save the report to a file
        with open("carbon_capture_vc_analysis.md", "w", encoding="utf-8") as f:
            f.write(result["report"])

        print(f"Investment analysis completed with {len(result['learnings'])} insights")
        print(f"Report saved to carbon_capture_vc_analysis.md")
    else:
        print(f"Research failed: {result.get('error')}")


async def research_with_per_query_prompts():
    """Example showing how to use different prompts for different research queries."""

    # Create an OpenAI LangChain model
    llm = ChatOpenAI(model="gpt-4")

    # Create the researcher with default prompts
    researcher = DeepResearcher(
        llm=llm,
        google_api_key=os.environ.get("GOOGLE_API_KEY"),
        google_cx=os.environ.get("GOOGLE_CX")
    )

    # Custom prompt just for this specific research query
    academic_research_prompts = {
        "query_generation": """You are a scientific researcher preparing a literature review.
        Generate precise academic search queries that will find peer-reviewed papers,
        research studies, meta-analyses, and scientific journals. Focus on finding
        quantitative studies, methodologies, and critical evaluations of the topic.""",

        "report_generation": """You are writing a scientific literature review for an academic journal.
        Organize findings into clear themes, highlighting:
        - Methodological approaches used in the field
        - Major findings and consensus views
        - Contradictory results and open questions
        - Gaps in current research
        
        Use formal academic language and maintain a balanced, critical perspective.
        Structure with proper sections and comprehensive citations."""
    }

    # Run research with custom prompts for just this query
    result = await researcher.research(
        query="Recent advances in quantum error correction techniques",
        breadth=3,
        depth=2,
        system_prompts=academic_research_prompts  # Apply only to this research query
    )

    # Process results
    if result["success"]:
        # Save the report to a file
        with open("quantum_error_correction_review.md", "w", encoding="utf-8") as f:
            f.write(result["report"])

        print(f"Academic review completed with {len(result['learnings'])} insights")
        print(f"Report saved to quantum_error_correction_review.md")
    else:
        print(f"Research failed: {result.get('error')}")


async def legal_research_example():
    """Example showing how to customize system prompts for legal research."""

    # Create an OpenAI LangChain model
    llm = ChatOpenAI(model="gpt-4")

    # Create the researcher with default settings
    researcher = DeepResearcher(
        llm=llm,
        google_api_key=os.environ.get("GOOGLE_API_KEY"),
        google_cx=os.environ.get("GOOGLE_CX")
    )

    # Legal research system prompts
    legal_prompts = {
        "query_generation": """You are a legal researcher at a top law firm.
        Generate precise legal search queries designed to find:
        1. Relevant case law and precedents
        2. Statutes and regulations
        3. Legal commentary and analysis
        4. Recent developments and changes in legislation
        
        For each query, specify the exact search string and what legal resources
        that query aims to discover. Prioritize authoritative legal sources.""",

        "result_analysis": """You are a legal associate extracting key information from legal research.
        Focus on:
        1. Precedents and their applications
        2. Statutory language and interpretation
        3. Jurisdictional distinctions
        4. Legal principles and doctrines
        5. Potentially conflicting interpretations
        
        Be precise in your legal analysis, noting dates, specific courts, and
        distinguishing between majority opinions, dissents, and dicta where mentioned.""",

        "report_generation": """You are a senior legal associate preparing a legal memorandum.
        Create a formal legal research memorandum that:
        1. Begins with a clear statement of the legal question
        2. Provides a brief answer/executive summary
        3. Analyzes relevant law including statutes and cases
        4. Discusses competing interpretations where applicable
        5. Concludes with reasoned legal analysis and recommendations
        
        Follow proper legal citation format (Bluebook style where possible).
        Maintain formal legal writing style and use precise legal terminology.
        Structure with proper sections including Facts, Issues, Analysis, and Conclusion."""
    }

    # Run legal research with custom prompts
    result = await researcher.research(
        query="Legal implications of using AI-generated content in commercial products",
        breadth=3,
        depth=2,
        system_prompts=legal_prompts
    )

    # Process results
    if result["success"]:
        # Save the report to a file
        with open("ai_content_legal_memo.md", "w", encoding="utf-8") as f:
            f.write(result["report"])

        print(f"Legal memorandum completed with {len(result['learnings'])} legal findings")
        print(f"Report saved to ai_content_legal_memo.md")
    else:
        print(f"Research failed: {result.get('error')}")


async def technical_research_example():
    """Example showing how to customize system prompts for technical documentation."""

    # Create an OpenAI LangChain model
    llm = ChatOpenAI(model="gpt-4")

    # Create the researcher
    researcher = DeepResearcher(
        llm=llm,
        google_api_key=os.environ.get("GOOGLE_API_KEY"),
        google_cx=os.environ.get("GOOGLE_CX")
    )

    # Technical system prompts
    technical_prompts = {
        "query_generation": """You are a senior software engineer researching a technical topic.
        Generate specific search queries to find:
        1. Official documentation and technical specifications
        2. Implementation examples and tutorials
        3. Known issues and limitations
        4. Best practices and optimization techniques
        
        Focus on finding authoritative technical sources like GitHub repositories,
        official documentation, technical blogs by recognized experts, and academic papers.""",

        "result_analysis": """You are a technical lead extracting key technical information.
        Focus on practical details like:
        1. Implementation requirements and dependencies
        2. Configuration parameters and options
        3. Performance characteristics and trade-offs
        4. Specific code patterns and architectures
        5. Error handling and debugging approaches
        
        Prioritize actionable technical details that would help a development team
        implement the technology effectively.""",

        "report_generation": """You are writing technical documentation for a development team.
        Create comprehensive technical documentation that:
        1. Starts with a high-level technical overview
        2. Explains core concepts and architecture
        3. Details implementation approaches with concrete examples
        4. Discusses trade-offs and performance considerations
        5. Provides troubleshooting guidance
        
        Include code examples where relevant, formatted in proper markdown code blocks.
        Use technical language appropriate for experienced developers.
        Structure with clear headings and subheadings for easy reference."""
    }

    # Run technical research with custom prompts
    result = await researcher.research(
        query="Implementing distributed tracing in microservices architecture",
        breadth=3,
        depth=2,
        system_prompts=technical_prompts
    )

    # Process results
    if result["success"]:
        # Save the report to a file
        with open("distributed_tracing_technical_guide.md", "w", encoding="utf-8") as f:
            f.write(result["report"])

        print(f"Technical documentation completed with {len(result['learnings'])} insights")
        print(f"Report saved to distributed_tracing_technical_guide.md")
    else:
        print(f"Research failed: {result.get('error')}")


if __name__ == "__main__":
    asyncio.run(main())
    # Uncomment to run the other examples
    # asyncio.run(research_with_per_query_prompts())
    # asyncio.run(legal_research_example())
    # asyncio.run(technical_research_example())
