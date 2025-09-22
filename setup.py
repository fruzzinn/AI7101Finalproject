"""
Setup configuration for Expresso Churn Prediction ML System.

Educational Focus: Demonstrates proper Python package configuration for ML projects.
"""

from setuptools import setup, find_packages

# Read requirements from requirements.txt
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

# Read long description from README (will be created later)
try:
    with open("README.md", "r", encoding="utf-8") as fh:
        long_description = fh.read()
except FileNotFoundError:
    long_description = "Expresso Churn Prediction ML System - Academic Project"

setup(
    name="expresso-churn-prediction",
    version="1.0.0",
    author="AI7101 Final Project Team",
    author_email="student@university.edu",
    description="Machine Learning system for predicting customer churn in telecommunications",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/kidamongus/AI7101finalproject",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Education",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.11",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.5.0",
            "isort>=5.12.0",
        ],
        "docs": [
            "sphinx>=7.1.0",
            "sphinx-rtd-theme>=1.3.0",
        ],
        "advanced": [
            "xgboost>=1.7.0",
            "lightgbm>=4.0.0",
            "optuna>=3.3.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "churn-predict=cli.main:main",
            "churn-data=cli.data_commands:main",
            "churn-model=cli.model_commands:main",
            "churn-business=cli.business_commands:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.json", "*.yaml", "*.yml"],
    },
    keywords="machine-learning, churn-prediction, telecommunications, data-science, education",
    project_urls={
        "Bug Reports": "https://github.com/kidamongus/AI7101finalproject/issues",
        "Source": "https://github.com/kidamongus/AI7101finalproject",
        "Documentation": "https://github.com/kidamongus/AI7101finalproject/docs",
    },
)