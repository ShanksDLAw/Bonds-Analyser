"""Test module for the cleaner package.

This module demonstrates how to use the cleaner package to clean and standardize
financial data from various sources.
"""

import os
import sys
import json
from pprint import pprint

# Add the project root to the Python path when running this file directly
if __name__ == "__main__":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    sys.path.insert(0, project_root)

from src.cleaner.bond_cleaner import BondDataCleaner
from src.cleaner.company_cleaner import CompanyDataCleaner
from src.tests.fetcher import BondRiskFetcher

def test_bond_data_cleaning():
    """Test cleaning bond data from different sources."""
    print("\n=== Testing Bond Data Cleaning ===")
    
    # Create instances
    fetcher = BondRiskFetcher()
    bond_cleaner = BondDataCleaner()
    
    # Test company with bonds (Apple as an example)
    company_symbol = "AAPL"
    
    # Get bond data from company-bond-list endpoint
    raw_bonds = fetcher._fmp_request(f"company-bond-list/{company_symbol}")
    print(f"\nRaw bonds from company-bond-list: {len(raw_bonds)}")
    
    # Clean the bond data
    cleaned_bonds = bond_cleaner.clean_company_bond_list(raw_bonds)
    print(f"Cleaned bonds from company-bond-list: {len(cleaned_bonds)}")
    
    if cleaned_bonds:
        print("\nSample cleaned bond:")
        pprint(cleaned_bonds[0])
    
    # Get bond data from debt-profile endpoint
    debt_profile = fetcher._fmp_request(f"debt-profile/{company_symbol}")
    print(f"\nRaw debt profile available: {bool(debt_profile)}")
    
    # Clean the debt profile bond data
    cleaned_debt_bonds = bond_cleaner.clean_debt_profile_bonds(debt_profile, company_symbol)
    print(f"Cleaned bonds from debt-profile: {len(cleaned_debt_bonds)}")
    
    if cleaned_debt_bonds:
        print("\nSample cleaned debt profile bond:")
        pprint(cleaned_debt_bonds[0])
    
    # Merge bond data from both sources
    merged_bonds = bond_cleaner.merge_bond_sources(cleaned_bonds, cleaned_debt_bonds)
    print(f"\nMerged bonds: {len(merged_bonds)}")
    print(f"Source of merged bonds: {merged_bonds[0]['source'] if merged_bonds else 'N/A'}")

def test_company_data_cleaning():
    """Test cleaning company financial data."""
    print("\n=== Testing Company Data Cleaning ===")
    
    # Create instances
    fetcher = BondRiskFetcher()
    company_cleaner = CompanyDataCleaner()
    
    # Test company (Apple as an example)
    company_symbol = "AAPL"
    
    # Get company profile data
    profile_data = fetcher._fmp_request(f"profile/{company_symbol}")[0]
    print(f"\nRaw profile data available: {bool(profile_data)}")
    
    # Clean the profile data
    cleaned_profile = company_cleaner.clean_company_profile(profile_data)
    print("\nCleaned profile data:")
    print(f"Company: {cleaned_profile.get('company_name')}")
    print(f"Market Cap: ${cleaned_profile.get('market_cap', 0):,.2f}")
    print(f"Industry: {cleaned_profile.get('industry')}")
    
    # Get financial ratios data
    ratios_data = fetcher._fmp_request(f"ratios-ttm/{company_symbol}")[0]
    print(f"\nRaw ratios data available: {bool(ratios_data)}")
    
    # Clean the ratios data
    cleaned_ratios = company_cleaner.clean_financial_ratios(ratios_data)
    print("\nCleaned financial ratios:")
    print(f"P/E Ratio: {cleaned_ratios.get('pe_ratio', 0):.2f}")
    print(f"Debt to Equity: {cleaned_ratios.get('debt_to_equity', 0):.2f}")
    print(f"Current Ratio: {cleaned_ratios.get('current_ratio', 0):.2f}")
    
    # Get balance sheet data
    balance_sheets = fetcher._fmp_request(f"balance-sheet-statement/{company_symbol}?limit=1")
    balance_sheet_data = balance_sheets[0] if balance_sheets else {}
    print(f"\nRaw balance sheet data available: {bool(balance_sheet_data)}")
    
    # Clean the balance sheet data
    cleaned_balance_sheet = company_cleaner.clean_balance_sheet(balance_sheet_data)
    print("\nCleaned balance sheet data:")
    print(f"Long-term Debt: ${cleaned_balance_sheet.get('long_term_debt', 0):,.2f}")
    print(f"Total Debt: ${cleaned_balance_sheet.get('total_debt', 0):,.2f}")
    print(f"Net Debt: ${cleaned_balance_sheet.get('net_debt', 0):,.2f}")
    print(f"Debt to Assets Ratio: {cleaned_balance_sheet.get('debt_to_assets_ratio', 0):.2f}")
    
    # Combine all financial data
    combined_data = company_cleaner.combine_financial_data(
        cleaned_profile, cleaned_ratios, cleaned_balance_sheet
    )
    print("\nCombined financial data:")
    print(f"Company: {combined_data.get('company_name')}")
    print(f"Market Cap: ${combined_data.get('market_cap', 0):,.2f}")
    print(f"P/E Ratio: {combined_data.get('pe_ratio', 0):.2f}")
    print(f"Total Debt: ${combined_data.get('total_debt', 0):,.2f}")
    print(f"Debt to Assets: {combined_data.get('debt_to_assets_ratio', 0):.2f}")
    print(f"Long-term Debt % of Total: {combined_data.get('long_term_debt_to_total_debt', 0) * 100:.1f}%")

def test_integrated_workflow():
    """Test the complete integrated workflow using cleaners."""
    print("\n=== Testing Integrated Workflow ===")
    
    # Create instances
    fetcher = BondRiskFetcher()
    bond_cleaner = BondDataCleaner()
    company_cleaner = CompanyDataCleaner()
    
    # Test company
    company_name = "Microsoft"
    print(f"\nAnalyzing company: {company_name}")
    
    # Step 1: Get company symbol
    symbol = fetcher._search_company_symbol(company_name)
    if not symbol:
        print(f"No symbol found for {company_name}")
        return
    print(f"Found symbol: {symbol}")
    
    # Step 2: Get financial data
    profile, ratios, balance_sheet = fetcher._get_financials(symbol)
    
    # Step 3: Clean financial data
    cleaned_profile = company_cleaner.clean_company_profile(profile)
    cleaned_ratios = company_cleaner.clean_financial_ratios(ratios)
    cleaned_balance_sheet = company_cleaner.clean_balance_sheet(balance_sheet)
    
    # Step 4: Get bond data from both sources
    raw_bonds = fetcher._fmp_request(f"company-bond-list/{symbol}")
    cleaned_bonds = bond_cleaner.clean_company_bond_list(raw_bonds)
    
    debt_profile = fetcher._fmp_request(f"debt-profile/{symbol}")
    cleaned_debt_bonds = bond_cleaner.clean_debt_profile_bonds(debt_profile, symbol)
    
    # Step 5: Merge bond data
    merged_bonds = bond_cleaner.merge_bond_sources(cleaned_bonds, cleaned_debt_bonds)
    
    # Step 6: Combine all financial data
    combined_data = company_cleaner.combine_financial_data(
        cleaned_profile, cleaned_ratios, cleaned_balance_sheet
    )
    
    # Add bonds to combined data
    combined_data['bonds'] = merged_bonds
    combined_data['bond_count'] = len(merged_bonds)
    
    # Display results
    print("\nCompany Analysis Results:")
    print(f"Company: {combined_data.get('company_name')}")
    print(f"Symbol: {combined_data.get('symbol')}")
    print(f"Market Cap: ${combined_data.get('market_cap', 0):,.2f}")
    print(f"P/E Ratio: {combined_data.get('pe_ratio', 0):.2f}")
    print(f"Total Debt: ${combined_data.get('total_debt', 0):,.2f}")
    print(f"Debt to Assets: {combined_data.get('debt_to_assets_ratio', 0):.2f}")
    print(f"Number of Bonds: {combined_data.get('bond_count', 0)}")
    
    if combined_data.get('bond_count', 0) > 0:
        print("\nBond Information:")
        for i, bond in enumerate(combined_data.get('bonds', [])[:3]):  # Show first 3 bonds
            print(f"\nBond {i+1}:")
            print(f"Symbol: {bond.get('symbol')}")
            print(f"Maturity: {bond.get('maturity_date')}")
            print(f"Coupon: {bond.get('coupon_rate', 0):.2f}%")
            print(f"YTM: {bond.get('yield_to_maturity', 0):.2f}%")
            print(f"Rating: {bond.get('credit_rating')}")
        
        if combined_data.get('bond_count', 0) > 3:
            print(f"\n... and {combined_data.get('bond_count', 0) - 3} more bonds")

# Run tests if executed directly
if __name__ == "__main__":
    print("\n==== Bond Risk Analyzer Cleaner Tests ====\n")
    
    # Run individual tests
    test_bond_data_cleaning()
    test_company_data_cleaning()
    test_integrated_workflow()
    
    print("\n==== All tests completed ====\n")