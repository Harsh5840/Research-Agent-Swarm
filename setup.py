from setuptools import setup, find_packages

setup(
    name="research-assistant",
    version="0.1.0",
    description="Autonomous AI Research Assistant",
    author="Research Team",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "langchain>=0.3.27",
        "langchain-openai>=0.3.29",
        "langchain-community>=0.3.27",
        "langchain-core>=0.3.0",
        "openai",
        "chromadb",
        "pymupdf",
        "tqdm",
        "python-dotenv>=1.1.1",
        "requests>=2.32.4",
        "fitz>=0.0.1.dev2",
        "pathlib>=1.0.1",
    ],
    extras_require={
        "dev": [
            "pytest",
            "black",
            "flake8",
        ],
    },
    entry_points={
        "console_scripts": [
            "research=apps.cli.main:main",
        ],
    },
) 