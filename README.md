# Bond Risk Analyzer

A comprehensive tool for analyzing corporate bonds and sovereign risks, providing detailed risk metrics, survival analysis, and stress testing capabilities.

## Features

### 1. Risk Analysis
- Default probability calculations
- Yield spread computation
- Combined risk scoring
- Credit Value at Risk (VaR)
- Expected shortfall analysis

### 2. Survival Analysis
- Probability of default estimation
- Loss given default calculations
- Exposure at default tracking
- Time to default projections
- Stress test scenarios:
  - Severe recession
  - Moderate stress
  - Recovery scenarios

### 3. Debt Analysis
- Debt capacity calculation
- Industry benchmarking
- Peer comparison
- Growth projections
- Risk recommendations

### 4. Data Integration
- Financial Market Place (FMP) API integration
- Yahoo Finance fallback
- World Bank economic indicators
- Country risk data
- Real-time market data

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/bond-risk-analyzer.git
cd bond-risk-analyzer
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
export FMP_API_KEY=your_api_key
export WORLDBANK_API_KEY=your_api_key  # Optional
```

## Usage

1. Run the Streamlit app:
```bash
streamlit run src/app.py
```

2. Enter bond information:
- ISIN/CUSIP
- Company name
- Face value
- Maturity date

3. View analysis results:
- Risk metrics
- Debt analysis
- Survival analysis
- Macro benchmarking
- Peer comparison
- Recommendations

## Project Structure

```
bond-risk-analyzer/
├── src/
│   ├── app.py                 # Main Streamlit application
│   ├── config.py             # Configuration settings
│   ├── data/                 # Data fetching modules
│   │   ├── fetcher.py       # Main data fetcher
│   │   ├── country_fetcher.py # Country data fetcher
│   │   └── bond_fetcher.py  # Bond data fetcher
│   ├── risk/                 # Risk analysis modules
│   │   ├── metrics.py       # Risk metrics calculation
│   │   ├── model_survival.py # Survival analysis model
│   │   ├── simulations.py   # Monte Carlo simulations
│   │   └── debt_analyzer.py # Debt analysis
│   ├── models/              # Data models
│   │   ├── company.py      # Company financials model
│   │   └── country.py      # Country risk model
│   └── macro/               # Macro-economic analysis
│       └── worldbank_loader.py # World Bank data loader
├── tests/                    # Test suite
├── requirements.txt          # Python dependencies
└── README.md                # This file
```

## Testing

Run the test suite:
```bash
pytest tests/
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Financial Market Place (FMP) API
- World Bank Data API
- Yahoo Finance API
- Streamlit for the UI framework 