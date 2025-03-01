from setuptools import setup, find_packages

setup(
    name="data-valuation",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy",
        "torch",
        "transformers",
        "scikit-learn",
        "wandb",
        "pytest",
        "tiktoken",
        "sentencepiece",
    ],
    python_requires=">=3.11",
)