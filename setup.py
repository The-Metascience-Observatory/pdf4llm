"""
Setup script for PDF-to-Markdown pipeline.

Install with:
    pip install -e .

Or:
    pip install -e ".[dev]"  # with development dependencies
"""

from setuptools import setup, find_packages

setup(
    name="pdf2mdjson",
    version="1.0.0",
    description="PDF-to-Markdown and JSON converter optimized for LLM analysis of scientific papers",
    author="Metascience Observatory",
    python_requires=">=3.9",
    packages=find_packages(),
    install_requires=[
        "requests>=2.31.0",
        "lxml>=4.9.0",
        "pydantic>=2.0.0",
        "pymupdf>=1.23.0",
        "pdfplumber>=0.10.0",
        "tqdm>=4.66.0",
        "click>=8.0.0",
        "Pillow>=10.0.0",  # For chart image processing
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "isort>=5.12.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "pdf2mdjson=pdf2mdjson.cli:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering",
        "Topic :: Text Processing :: Markup",
    ],
)
