"""Setup script for inoculation-probing package."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="inoculation-probing",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Experimental framework for testing inoculation prompts in linear probing",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mahadikprasad15/inoculation-probing",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.35.0",
        "datasets>=2.14.0",
        "huggingface-hub>=0.19.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "scikit-learn>=1.3.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "tqdm>=4.65.0",
        "pandas>=2.0.0",
    ],
)
