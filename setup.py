from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="langchain-deepresearch",
    version="0.1.0",
    author="Arif Dogan",
    author_email="me@arif.sh",
    description="Autonomous research capabilities for LangChain models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/doganarif/langchain-deepresearch",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.9",
    install_requires=[
        "langchain-core>=0.1.0",
        "aiohttp>=3.8.0",
        "asyncio>=3.4.3",
        "PyYAML>=6.0",
        "requests>=2.0.0",
    ],
    extras_require={
        "openai": ["langchain-openai>=0.0.1"],
        "anthropic": ["langchain-anthropic>=0.0.1"],
        "google": ["langchain-google-genai>=0.0.1"],
        "huggingface": ["langchain-huggingface>=0.0.1"],
        "all": [
            "langchain-openai>=0.0.1",
            "langchain-anthropic>=0.0.1",
            "langchain-google-genai>=0.0.1",
            "langchain-huggingface>=0.0.1",
        ],
    },
)
