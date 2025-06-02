from setuptools import setup, find_packages

setup(
    name="bond_risk_analyzer",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scipy>=1.7.0",
        "requests>=2.26.0",
        "python-dotenv>=0.19.0",
        "yfinance>=0.1.63",
        "pytest>=6.2.5",
        "tenacity>=8.0.1",
        "pydantic>=2.0.0",
        "python-dateutil>=2.8.2",
        "typing-extensions>=4.0.0",
        "streamlit>=1.10.0",
        "plotly>=5.3.1",
        "investpy>=1.0.8"  # For real bond data
    ],
    python_requires=">=3.8",
    author="Bond Risk Analyzer Team",
    description="A comprehensive bond risk analysis tool",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)