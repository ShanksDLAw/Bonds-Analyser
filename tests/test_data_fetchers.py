import unittest
from unittest.mock import patch, MagicMock
from src.data.country_fetcher import IMFClient, WorldBankClient, CountryDataManager
import logging
import json

logging.basicConfig(level=logging.INFO)

class TestDataFetchers(unittest.TestCase):
    def setUp(self):
        self.test_countries = ['USA', 'GBR', 'JPN', 'DEU']
        
        # Mock IMF response for each indicator
        self.mock_imf_responses = {
            'NGDP_RPCH': {
                'values': {
                    'USA': {'2023': 2.5},
                    'GBR': {'2023': 1.8},
                    'JPN': {'2023': 1.0},
                    'DEU': {'2023': 1.5}
                }
            },
            'PCPIPCH': {
                'values': {
                    'USA': {'2023': 3.0},
                    'GBR': {'2023': 2.5},
                    'JPN': {'2023': 0.8},
                    'DEU': {'2023': 2.2}
                }
            },
            'NGDP_FX': {
                'values': {
                    'USA': {'2023': 500000},
                    'GBR': {'2023': 150000},
                    'JPN': {'2023': 300000},
                    'DEU': {'2023': 200000}
                }
            }
        }
        
        # Mock World Bank response
        self.mock_wb_response = [
            {'page': 1, 'pages': 1, 'per_page': 1, 'total': 1},
            [{'value': 2.5, 'date': '2023', 'indicator': {'id': 'NY.GDP.MKTP.KD.ZG'}}]
        ]
        
        # Mock Treasury response
        self.mock_treasury_response = {
            'data': [
                {
                    'security_desc': '10-Year Treasury Constant Maturity Rate',
                    'avg_interest_rate_amt': '3.5'
                }
            ]
        }
        
        # Initialize clients with mocked responses
        with patch('requests.Session') as mock_session:
            mock_get = MagicMock()
            mock_get.status_code = 200
            
            def mock_json(*args, **kwargs):
                url = mock_session.return_value.get.call_args[0][0]
                if 'imf.org' in url:
                    indicator = url.split('/')[-2]
                    return self.mock_imf_responses.get(indicator, {'values': {}})
                elif 'treasury.gov' in url:
                    return self.mock_treasury_response
                else:
                    return self.mock_wb_response
            
            mock_get.json = mock_json
            mock_session.return_value.get.return_value = mock_get
            
            self.imf = IMFClient()
            self.wb = WorldBankClient()
            self.manager = CountryDataManager()
    
    @patch('requests.Session')
    def test_imf_client(self, mock_session):
        """Test IMF data fetching"""
        mock_get = MagicMock()
        mock_get.status_code = 200
        
        def mock_json(*args, **kwargs):
            url = mock_session.return_value.get.call_args[0][0]
            indicator = url.split('/')[-2]
            return self.mock_imf_responses.get(indicator, {'values': {}})
        
        mock_get.json = mock_json
        mock_session.return_value.get.return_value = mock_get
        
        for country in self.test_countries:
            data = self.imf.get_all_indicators(country)
            self.assertIsInstance(data, dict)
            self.assertTrue(len(data) > 0, f"No IMF data returned for {country}")
            # Check data types
            for value in data.values():
                self.assertIsInstance(value, (int, float))
    
    @patch('requests.Session')
    def test_world_bank_client(self, mock_session):
        """Test World Bank data fetching"""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = self.mock_wb_response
        mock_session.return_value.get.return_value = mock_response
        
        for country in self.test_countries:
            data = self.wb.get_all_indicators(country)
            self.assertIsInstance(data, dict)
            self.assertTrue(len(data) > 0, f"No World Bank data returned for {country}")
            # Check data types
            for value in data.values():
                self.assertIsInstance(value, (int, float))
    
    @patch('requests.Session')
    def test_country_manager(self, mock_session):
        """Test comprehensive data fetching"""
        mock_get = MagicMock()
        mock_get.status_code = 200
        
        def mock_json(*args, **kwargs):
            url = mock_session.return_value.get.call_args[0][0]
            if 'imf.org' in url:
                indicator = url.split('/')[-2]
                return self.mock_imf_responses.get(indicator, {'values': {}})
            elif 'treasury.gov' in url:
                return self.mock_treasury_response
            else:
                return self.mock_wb_response
        
        mock_get.json = mock_json
        mock_session.return_value.get.return_value = mock_get
        
        for country in self.test_countries:
            data = self.manager.get_comprehensive_metrics(country)
            self.assertIsInstance(data, dict)
            
            # Check structure
            self.assertIn('macro', data)
            self.assertIn('market', data)
            self.assertIn('risk', data)
            
            # Check if we have some data in each category
            for category in ['macro', 'market']:
                self.assertTrue(len(data[category]) > 0, 
                              f"No {category} data for {country}")
            
            # For risk data, we expect it for non-USA countries
            if country != 'USA':
                self.assertTrue(len(data['risk']) > 0,
                              f"No risk data for {country}")
            else:
                # For USA, we expect FX reserves but no sovereign spread
                if len(data['risk']) > 0:
                    self.assertNotIn('sovereign_spread', data['risk'],
                                   "USA should not have sovereign spread")
    
    @patch('requests.Session')
    def test_sovereign_spread_calculation(self, mock_session):
        """Test sovereign spread calculation"""
        mock_get = MagicMock()
        mock_get.status_code = 200
        
        def mock_json(*args, **kwargs):
            url = mock_session.return_value.get.call_args[0][0]
            if 'imf.org' in url:
                indicator = url.split('/')[-2]
                return self.mock_imf_responses.get(indicator, {'values': {}})
            elif 'treasury.gov' in url:
                return self.mock_treasury_response
            else:
                return [
                    {'page': 1, 'pages': 1, 'per_page': 1, 'total': 1},
                    [{'value': 0.035, 'date': '2023'}]  # 3.5% yield
                ]
        
        mock_get.json = mock_json
        mock_session.return_value.get.return_value = mock_get
        
        # Get US data first
        us_data = self.manager.get_comprehensive_metrics('USA')
        # Test other countries
        for country in [c for c in self.test_countries if c != 'USA']:
            data = self.manager.get_comprehensive_metrics(country)
            if 'sovereign_spread' in data['risk']:
                spread = data['risk']['sovereign_spread']
                self.assertIsInstance(spread, (int, float, type(None)))
                if spread is not None:
                    self.assertGreaterEqual(spread, -0.1)  # Reasonable lower bound
                    self.assertLessEqual(spread, 0.2)      # Reasonable upper bound

if __name__ == '__main__':
    unittest.main() 