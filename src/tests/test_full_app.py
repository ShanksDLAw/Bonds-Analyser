import unittest
import sys
from pathlib import Path
import streamlit as st
from datetime import datetime
import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv

# Add the src directory to the Python path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.data.fetcher import BondRiskFetcher, APIError
from src.models.company import CompanyFinancials, Bond
from src.app import format_percentage, format_basis_points

class TestFullApplication(unittest.TestCase):
    """Test suite for the full Bond Risk Analyzer application"""

    @classmethod
    def setUpClass(cls):
        """Set up test environment"""
        load_dotenv()
        if not os.getenv('FMP_API_KEY'):
            raise ValueError("FMP_API_KEY not found in environment variables")
        cls.fetcher = BondRiskFetcher()

    def test_data_pipeline(self):
        """Test the complete data pipeline"""
        print("\n🔍 Testing Data Pipeline...")
        
        try:
            # 1. Fetch company data
            raw_data = self.fetcher.get_company_data('AAPL')
            self.assertIsNotNone(raw_data)
            print("✅ Successfully fetched company data")
            
            # 2. Create CompanyFinancials object
            company = CompanyFinancials.from_api_data(raw_data)
            self.assertIsNotNone(company)
            print("✅ Successfully created CompanyFinancials object")
            
            # 3. Check company attributes
            self.assertEqual(company.symbol, raw_data['profile']['symbol'])
            self.assertEqual(company.name, raw_data['profile']['companyName'])
            print("✅ Company attributes are correct")
            
            # 4. Check bond data
            self.assertTrue(len(company.bonds) > 0)
            print(f"✅ Successfully loaded {len(company.bonds)} bonds")
            
            # 5. Calculate risk metrics
            risk_metrics = company.get_risk_metrics()
            self.assertIsNotNone(risk_metrics)
            print("✅ Successfully calculated risk metrics")
            
            # Print risk metrics
            for risk_type, score in risk_metrics.items():
                print(f"  - {risk_type}: {score:.2%}")
            
        except Exception as e:
            print(f"❌ Error in data pipeline: {str(e)}")
            raise

    def test_formatting_functions(self):
        """Test formatting functions"""
        print("\n🔍 Testing Formatting Functions...")
        
        # Test percentage formatting
        self.assertEqual(format_percentage(0.1234, 2), "12.34%")
        self.assertEqual(format_percentage(None), "N/A")
        print("✅ Percentage formatting works correctly")
        
        # Test basis points formatting
        self.assertEqual(format_basis_points(0.0123), "123bps")
        self.assertEqual(format_basis_points(None), "N/A")
        print("✅ Basis points formatting works correctly")

    def test_risk_calculations(self):
        """Test risk calculation components"""
        print("\n🔍 Testing Risk Calculations...")
        
        try:
            # Get test data
            raw_data = self.fetcher.get_company_data('MSFT')
            company = CompanyFinancials.from_api_data(raw_data)
            
            # Test individual risk components
            credit_risk = company._calculate_credit_risk()
            self.assertGreaterEqual(credit_risk, 0)
            self.assertLessEqual(credit_risk, 1)
            print(f"✅ Credit risk calculation: {credit_risk:.2%}")
            
            market_risk = company._calculate_market_risk()
            self.assertGreaterEqual(market_risk, 0)
            self.assertLessEqual(market_risk, 1)
            print(f"✅ Market risk calculation: {market_risk:.2%}")
            
            liquidity_risk = company._calculate_liquidity_risk()
            self.assertGreaterEqual(liquidity_risk, 0)
            self.assertLessEqual(liquidity_risk, 1)
            print(f"✅ Liquidity risk calculation: {liquidity_risk:.2%}")
            
            duration_risk = company._calculate_duration_risk()
            self.assertGreaterEqual(duration_risk, 0)
            self.assertLessEqual(duration_risk, 1)
            print(f"✅ Duration risk calculation: {duration_risk:.2%}")
            
        except Exception as e:
            print(f"❌ Error in risk calculations: {str(e)}")
            raise

    def test_data_consistency(self):
        """Test data consistency across the pipeline"""
        print("\n🔍 Testing Data Consistency...")
        
        try:
            # Test with multiple companies
            companies = ['AAPL', 'MSFT', 'GOOGL']
            results = []
            
            for symbol in companies:
                raw_data = self.fetcher.get_company_data(symbol)
                company = CompanyFinancials.from_api_data(raw_data)
                risk_metrics = company.get_risk_metrics()
                
                results.append({
                    'symbol': symbol,
                    'data': raw_data,
                    'company': company,
                    'risk_metrics': risk_metrics
                })
                print(f"✅ Successfully processed {symbol}")
            
            # Verify data consistency
            for result in results:
                # Check data structure
                self.assertIn('profile', result['data'])
                self.assertIn('quote', result['data'])
                self.assertIn('ratios', result['data'])
                self.assertIn('bonds', result['data'])
                
                # Check risk metrics
                self.assertIn('credit_risk', result['risk_metrics'])
                self.assertIn('market_risk', result['risk_metrics'])
                self.assertIn('liquidity_risk', result['risk_metrics'])
                self.assertIn('duration_risk', result['risk_metrics'])
                self.assertIn('overall_risk', result['risk_metrics'])
                
                print(f"✅ Data structure verified for {result['symbol']}")
                
        except Exception as e:
            print(f"❌ Error in data consistency test: {str(e)}")
            raise

    def test_error_handling(self):
        """Test error handling throughout the pipeline"""
        print("\n🔍 Testing Error Handling...")
        
        # Test invalid symbol
        try:
            raw_data = self.fetcher.get_company_data('INVALID_SYMBOL_123')
            self.assertIsNotNone(raw_data)  # Should get synthetic data
            print("✅ Handled invalid symbol correctly")
        except Exception as e:
            print(f"❌ Failed to handle invalid symbol: {str(e)}")
            raise
        
        # Test missing data handling
        try:
            raw_data = self.fetcher.get_company_data('AAPL')
            # Simulate missing data
            del raw_data['ratios']
            company = CompanyFinancials.from_api_data(raw_data)
            self.assertIsNotNone(company)
            print("✅ Handled missing data correctly")
        except Exception as e:
            print(f"❌ Failed to handle missing data: {str(e)}")
            raise

def run_tests():
    """Run all tests"""
    print("\n🧪 Starting Full Application Tests...")
    print("=" * 50)
    
    # Run tests
    unittest.main(argv=[''], verbosity=2, exit=False)

if __name__ == '__main__':
    run_tests() 