from setuptools import setup, find_packages

setup(
    name="spikingbrain",
    version="0.1.0",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.1.0",
        "transformers>=4.38.0",
        "accelerate>=0.27.0",
        "numpy>=1.24.0",
        "pydantic>=2.0.0",
        "tabulate>=0.9.0",
        "rich>=13.0.0",
    ],
)
