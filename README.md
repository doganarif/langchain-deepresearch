<a href="https://syntaks.ai" target="_blank"><img src="https://img.shields.io/badge/syntaks.ai-AI%20MVPs%20in%204%20Weeks-f39c12?style=for-the-badge&logo=data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjQiIGhlaWdodD0iMjQiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyI+PHBhdGggZD0iTTEyIDJsMyAxMC0xMCA5eiIgZmlsbD0iIzAwZmY5ZCIvPjwvc3ZnPg==" alt="syntaks.ai - AI MVPs in 4 Weeks"></a>

# LangChain DeepResearch

[![PyPI version](https://badge.fury.io/py/langchain-deepresearch.svg)](https://badge.fury.io/py/langchain-deepresearch)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/downloads/)

A LangChain integration that provides autonomous, recursive research capabilities powered by any LangChain-compatible LLM.

## 🔍 Features

- **Model Agnostic**: Works with any LangChain-compatible LLM (OpenAI, Anthropic, Llama, etc.)
- **Time-Aware Research**: Automatically manages research time constraints
- **Recursive Exploration**: Follows leads discovered during research to explore topics in depth
- **Adaptive Strategies**: Adjusts research breadth and depth based on time and findings
- **Comprehensive Reporting**: Generates structured research reports with citations
- **Customizable System Prompts**: Control the personality, focus, and output style at each stage of research

## 📦 Installation

```bash
pip install langchain-deepresearch
```

Or install from source:

```bash
git clone https://github.com/doganarif/langchain-deepresearch.git
cd langchain-deepresearch
pip install -e .
```

## 🚀 Quick Start

```python
import asyncio
from langchain_openai import ChatOpenAI
from langchain_deepresearch import DeepResearcher

async def main():
    # Initialize any LangChain model
    llm = ChatOpenAI(model="gpt-3.5-turbo")
    
    # Create the researcher with your LangChain model
    researcher = DeepResearcher(
        llm=llm,
        google_api_key="your-google-api-key",  # Or use environment variable
        google_cx="your-google-cx-id"          # Or use environment variable
    )
    
    # Run the research
    result = await researcher.research(
        query="Latest advancements in fusion energy", 
        breadth=3,    # Number of parallel searches
        depth=2       # Depth of recursive exploration
    )
    
    # Access results
    if result["success"]:
        print(result["report"])  # Markdown report
        print(f"Sources consulted: {len(result['visited_urls'])}")
        print(f"Insights gathered: {len(result['learnings'])}")

if __name__ == "__main__":
    asyncio.run(main())
```

## 🔄 Works with Any LangChain LLM

Use it with any LangChain-compatible model:

```python
# With OpenAI
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-4")
researcher = DeepResearcher(llm=llm)

# With Anthropic
from langchain_anthropic import ChatAnthropic
llm = ChatAnthropic(model="claude-3-opus-20240229")
researcher = DeepResearcher(llm=llm)

# With Hugging Face models
from langchain_huggingface import HuggingFaceEndpoint
llm = HuggingFaceEndpoint(repo_id="mistralai/Mistral-7B-Instruct-v0.2")
researcher = DeepResearcher(llm=llm)
```

## ⚙️ Configuration

You'll need to configure search API keys:

```python
# Configure with explicit parameters
researcher = DeepResearcher(
    llm=llm,
    google_api_key="your-google-api-key",
    google_cx="your-google-cx-id",
    max_time_seconds=2400,  # 40 minutes (default)
    min_learnings_required=8  # Minimum insights before completion
)

# Or use environment variables
# GOOGLE_API_KEY
# GOOGLE_CX
```

## 🎭 Customizing System Prompts

You can customize the system prompts used at different stages of the research process:

### Global Customization (applied to all research)

```python
# Define custom system prompts
custom_prompts = {
    # Prompt for generating search queries
    "query_generation": """You are a venture capital analyst researching a market opportunity.
    Create specific search queries to gather competitive intelligence, market size data, 
    growth trends, and regulatory concerns...""",
    
    # Prompt for analyzing search results
    "result_analysis": """You are a venture capital analyst evaluating a potential investment.
    Extract key financial data, competitive advantages, and market positioning...""",
    
    # Prompt for generating the final report
    "report_generation": """You are a senior investment analyst at a top venture capital firm.
    Create a comprehensive investment analysis report with clear investment recommendation..."""
}

# Initialize with custom prompts
researcher = DeepResearcher(
    llm=llm,
    google_api_key="your-key",
    google_cx="your-cx-id",
    system_prompts=custom_prompts  # Apply to all research
)
```

### Per-Query Customization

```python
# These prompts apply only to this specific research query
academic_prompts = {
    "query_generation": """You are a scientific researcher preparing a literature review...""",
    "report_generation": """You are writing a scientific literature review for an academic journal..."""
}

result = await researcher.research(
    query="Recent advances in quantum computing",
    system_prompts=academic_prompts  # Override just for this query
)
```

## 🔍 How It Works

1. **Query Analysis**: The agent breaks down the research topic into specific search queries
2. **Multiple Search Paths**: Executes multiple parallel search paths to gather diverse information
3. **Content Extraction**: Analyzes search results to extract key learnings and insights
4. **Recursive Exploration**: Uses discovered information to generate follow-up searches for deeper exploration
5. **Report Generation**: Synthesizes all findings into a cohesive research report

## 📋 Advanced Usage

### Quick Research

For faster, less in-depth research:

```python
result = await researcher.quick_research(
    query="Carbon capture technologies",  
    time_limit=300  # 5 minutes (default)
)
```

### Advanced Research Parameters

```python
result = await researcher.research(
    query="Advances in protein folding algorithms",
    breadth=5,                    # More parallel searches
    depth=3,                      # Deeper exploration
    time_limit=3600,              # Longer time limit (1 hour)
    min_learnings_required=12,    # Require more learnings
    max_searches=250,             # Allow more searches
    system_prompts=custom_prompts # Custom system prompts
)
```

### Using with LangChain Chains and Agents

```python
from langchain.agents import initialize_agent, Tool
from langchain_deepresearch import DeepResearcher

# Initialize a DeepResearcher instance
researcher = DeepResearcher(llm=llm)

# Create a Tool for agents
research_tool = Tool(
    name="DeepResearch",
    description="Thoroughly researches a topic and generates a comprehensive report",
    func=lambda query: researcher.research(query, breadth=3, depth=2)
)

# Use it in an agent
agent = initialize_agent([research_tool, ...], llm, agent="zero-shot-react-description")
```

## 📚 Example Use Cases

### Venture Capital Analysis

```python
vc_prompts = {
    "report_generation": """You are a senior investment analyst at a top venture capital firm.
    Create a comprehensive investment analysis report with clear investment recommendation..."""
}

result = await researcher.research(
    query="Market opportunity for carbon capture startups",
    system_prompts=vc_prompts
)
```

### Academic Literature Review

```python
academic_prompts = {
    "query_generation": """You are a scientific researcher preparing a literature review.
    Generate precise academic search queries that will find peer-reviewed papers...""",
    "report_generation": """You are writing a scientific literature review for an academic journal..."""
}

result = await researcher.research(
    query="Recent advances in quantum error correction",
    system_prompts=academic_prompts
)
```

### Legal Research

```python
legal_prompts = {
    "query_generation": """You are a legal researcher at a top law firm.
    Generate precise legal search queries designed to find relevant case law and statutes...""",
    "report_generation": """You are a senior legal associate preparing a legal memorandum..."""
}

result = await researcher.research(
    query="Legal implications of AI-generated content",
    system_prompts=legal_prompts
)
```

### Technical Documentation

```python
technical_prompts = {
    "query_generation": """You are a senior software engineer researching a technical topic...""",
    "report_generation": """You are writing technical documentation for a development team..."""
}

result = await researcher.research(
    query="Implementing distributed tracing in microservices",
    system_prompts=technical_prompts
)
```

## 📝 API Reference

### DeepResearcher Class

```python
DeepResearcher(
    llm,                           # LangChain model (required)
    google_api_key=None,           # Google Search API key
    google_cx=None,                # Google Custom Search CX ID
    firecrawl_api_key=None,        # Optional Firecrawl API key
    firecrawl_url=None,            # Optional Firecrawl URL
    max_time_seconds=2400,         # Max research time (40 min)
    min_research_time_seconds=180, # Min research time (3 min)
    min_learnings_required=8,      # Min insights needed
    max_searches=200,              # Max searches to perform
    verbose=False,                 # Verbose logging
    system_prompts=None            # Custom system prompts
)
```

### Research Method

```python
await researcher.research(
    query,                         # Research query (required)
    breadth=3,                     # Parallel searches
    depth=2,                       # Recursive depth
    time_limit=None,               # Specific time limit
    report_model=None,             # Alternative model for report
    min_learnings_required=None,   # Override min learnings
    max_searches=None,             # Override max searches
    system_prompts=None            # Query-specific prompts
)
```

### Quick Research Method

```python
await researcher.quick_research(
    query,                         # Research query (required)
    time_limit=300,                # 5 minutes (default)
    system_prompts=None            # Custom system prompts
)
```

## 🙋 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- This project uses [LangChain](https://www.langchain.com/) for LLM integration
- Powered by Google Programmable Search Engine for web searches
