import requests
from bs4 import BeautifulSoup
import supabase
import os
from datetime import datetime, timedelta
import logging
from typing import List, Dict
from dotenv import load_dotenv

load_dotenv()  # .env 파일 로드

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class DisclosureCollector:
    def __init__(self):
        self.base_url = "https://kpub.knia.or.kr"
        self.supabase_url = os.getenv('SUPABASE_URL')
        self.supabase_key = os.getenv('SUPABASE_KEY')
        self.supabase = supabase.create_client(self.supabase_url, self.supabase_key)
        
        # Insurance comparison URLs by type
        self.insurance_links = {
            'savings': f"{self.base_url}/savingDisc/savingDisc/savingDiscSummary.do",
            'fire_insurance': f"{self.base_url}/productDisc/longTermGuarantee/fireInsurance.do",
            'accident_insurance': f"{self.base_url}/productDisc/longTermGuarantee/teethInsurance.do",
            'pension_savings': f"{self.base_url}/productDisc/pensionSaving/pensionSavingDisclosure.do",
            'medical_insurance': f"{self.base_url}/productDisc/lostHealth/lostHealthDisclosure.do",
            'car_insurance': f"{self.base_url}/carInsuranceDisc/compare/carCompare.do",
            'other_insurance': f"{self.base_url}/etcDisc/inCompSale/compDiscInsSale.do"
        }
        
        # Mapping of Korean names to keys for easy lookup
        self.insurance_types = {
            '저축성보험': 'savings',
            '장기화재/종합보험': 'fire_insurance',
            '장기상해/기타보험': 'accident_insurance',
            '연금저축보험': 'pension_savings',
            '실손의료보험': 'medical_insurance',
            '자동차보험': 'car_insurance',
            '기타/기타비교공시': 'other_insurance'
        }
        
    def get_insurance_link(self, insurance_type: str) -> str:
        """Get the comparison URL for a specific insurance type
        
        Args:
            insurance_type (str): Either the Korean name or English key of the insurance type
            
        Returns:
            str: The URL for the insurance comparison page
            
        Raises:
            ValueError: If the insurance type is not found
        """
        # Try to get by Korean name first, then by key
        key = self.insurance_types.get(insurance_type, insurance_type)
        if key in self.insurance_links:
            return self.insurance_links[key]
        raise ValueError(f"Unknown insurance type: {insurance_type}")
        
    def _parse_js_object(self, js_str: str):
        """Parse a JavaScript object literal string into a Python dictionary
        
        Args:
            js_str (str): JavaScript object literal string
            
        Returns:
            dict: Parsed Python dictionary
        """
        import re
        import json
        
        try:
            # Try to parse as JSON first (in case it's already valid JSON)
            return json.loads(js_str)
        except json.JSONDecodeError:
            # If not valid JSON, try to convert from JS object literal
            try:
                # Remove trailing semicolon if present
                if js_str.endswith(';'):
                    js_str = js_str[:-1]
                
                # Handle unquoted property names (convert to JSON)
                def replacer(match):
                    key = match.group(1)
                    # Check if the key needs quotes (not already quoted and not a number)
                    if not (key.startswith('"') or key[0].isdigit()):
                        return f'"{key}"'
                    return key
                
                # Match unquoted keys followed by a colon
                pattern = r'([a-zA-Z_$][a-zA-Z0-9_$]*)\s*:'
                json_str = re.sub(pattern, replacer, js_str)
                
                # Convert single quotes to double quotes for JSON
                json_str = json_str.replace("'", '"')
                
                # Handle unquoted values (like true, false, null)
                json_str = json_str.replace(': true', ': true').replace(':false', ':false')
                json_str = json_str.replace(': true', ': true').replace(':false', ':false')
                
                # Parse the JSON string
                return json.loads(json_str)
                
            except Exception as e:
                logging.warning(f"Failed to parse JavaScript object: {e}")
                return {'content': js_str}
    
    def _make_api_request(self, endpoint: str, payload: dict, referer: str) -> dict:
        """Make a request to the KNIA API
        
        Args:
            endpoint (str): API endpoint (e.g., '/popup/disclosureList.do')
            payload (dict): Request payload
            referer (str): Referer URL for the request
            
        Returns:
            dict: Parsed JSON response or raw content if not JSON
            
        Raises:
            Exception: If the request fails
        """
        url = f"{self.base_url}{endpoint}"
        
        # Headers for the API request
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'application/json, text/javascript, */*; q=0.01',
            'Content-Type': 'application/x-www-form-urlencoded; charset=UTF-8',
            'X-Requested-With': 'XMLHttpRequest',
            'Origin': self.base_url,
            'Referer': referer
        }
        
        # Disable SSL warnings
        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        
        # Configure session to not verify SSL
        session = requests.Session()
        session.verify = False
        
        try:
            logging.info(f"Making request to {url}")
            response = session.post(
                url,
                headers=headers,
                data=payload,
                timeout=30
            )
            
            # Check if the response is successful
            response.raise_for_status()
            
            # Get the response text
            response_text = response.text.strip()
            
            # Try to parse as JSON first
            try:
                result = response.json()
                logging.info(f"Successfully received JSON response with {len(result) if isinstance(result, (list, dict)) else 'unknown'} items")
                return result
            except ValueError:
                # If not JSON, try to parse as JavaScript object literal
                try:
                    result = self._parse_js_object(response_text)
                    if isinstance(result, dict) and 'content' not in result:
                        logging.info(f"Successfully parsed JavaScript object with {len(result) if isinstance(result, (list, dict)) else 'unknown'} items")
                    return result
                except Exception as e:
                    logging.warning(f"Failed to parse response as JavaScript object: {e}")
                    return {'content': response_text}
                
        except requests.exceptions.RequestException as e:
            error_msg = f"API request failed: {e}"
            if hasattr(e, 'response') and e.response is not None:
                error_msg += f"\nStatus code: {e.response.status_code}"
                try:
                    error_msg += f"\nResponse: {e.response.text[:500]}"
                except:
                    pass
            logging.error(error_msg)
            raise Exception(error_msg) from e
    
    def get_insurance_data(self, insurance_type: str, page: int = 1, page_size: int = 20) -> dict:
        """Get insurance data by type
        
        Args:
            insurance_type (str): Type of insurance (can be Korean name or English key)
            page (int, optional): Page number. Defaults to 1.
            page_size (int, optional): Number of items per page. Defaults to 20.
            
        Returns:
            dict: The API response
            
        Raises:
            ValueError: If the insurance type is not found
            Exception: If the API request fails
        """
        # Get the insurance type key
        key = self.insurance_types.get(insurance_type, insurance_type)
        if key not in self.insurance_links:
            raise ValueError(f"Unknown insurance type: {insurance_type}")
        
        # Define the payload based on insurance type
        payload = {
            'tabType': '1',
            'tptyCode': 'PB11',  # This might need to be dynamic based on insurance type
            'refreshYn': '',
            'detailYn': '',
            'pCode': '',
            'channel': '',
            'prdNm': '',
            'payNm': '',
            'payReason': '',
            '__sort__': '1 asc',
            'pageIndex': str(page),
            'pageUnit': str(page_size)
        }
        
        # Make the API request
        return self._make_api_request(
            endpoint='/popup/disclosureList.do',
            payload=payload,
            referer=self.insurance_links[key]
        )
    
    def get_fire_insurance_data(self, page: int = 1, page_size: int = 20) -> dict:
        """Fetch fire insurance data from the API
        
        Args:
            page (int, optional): Page number. Defaults to 1.
            page_size (int, optional): Number of items per page. Defaults to 20.
            
        Returns:
            dict: The API response
            
        Raises:
            Exception: If the API request fails
        """
        return self.get_insurance_data('fire_insurance', page, page_size)
    
    def get_disclosure_list(self, date: str, insurance_type: str = None) -> List[Dict]:
        """Get disclosure list for a specific date and optionally filtered by insurance type
        
        Args:
            date (str): Date in YYYY-MM-DD format
            insurance_type (str, optional): Insurance type to filter by. Defaults to None.
            
        Returns:
            List[Dict]: List of disclosure items
        """
        if insurance_type:
            url = self.get_insurance_link(insurance_type)
        else:
            url = f"{self.base_url}/disclosure/list.do"
            
        params = {
            'date': date,
            'page': 1
        }
        
        # Common headers for web requests
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Sec-Fetch-User': '?1',
            'Cache-Control': 'max-age=0',
            'Referer': self.base_url
        }
        
        try:
            # First try with SSL verification
            try:
                response = requests.get(url, params=params, headers=headers, verify=True)
                response.raise_for_status()
            except requests.exceptions.SSLError as ssl_err:
                logging.warning(f"SSL verification failed, retrying without verification: {ssl_err}")
                # Retry without SSL verification (use with caution in production)
                import urllib3
                urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
                response = requests.get(url, params=params, headers=headers, verify=False)
                response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            disclosures = []
            # Find disclosure items in the HTML
            items = soup.find_all('div', class_='disclosure-item')
            for item in items:
                disclosure = {
                    'company_name': item.find('span', class_='company').text.strip(),
                    'disclosure_date': date,
                    'disclosure_type': item.find('span', class_='type').text.strip(),
                    'title': item.find('h3', class_='title').text.strip(),
                    'file_url': self.base_url + item.find('a')['href'] if item.find('a') else None,
                    'file_name': item.find('a').text.strip() if item.find('a') else None
                }
                disclosures.append(disclosure)
            
            return disclosures
            
        except Exception as e:
            logging.error(f"Error fetching disclosure list: {e}")
            return []
            
    def download_file(self, url: str, file_name: str) -> str:
        """Download disclosure file and return content"""
        try:
            response = requests.get(url)
            response.raise_for_status()
            return response.text
        except Exception as e:
            logging.error(f"Error downloading file {file_name}: {e}")
            return None
            
    def store_in_supabase(self, disclosures: List[Dict]):
        """Store disclosure information in Supabase"""
        try:
            for disclosure in disclosures:
                # Download file content if available
                if disclosure['file_url']:
                    disclosure['content'] = self.download_file(
                        disclosure['file_url'],
                        disclosure['file_name']
                    )
                
                # Insert into Supabase
                self.supabase.table('disclosure_info').insert(disclosure).execute()
                logging.info(f"Stored disclosure: {disclosure['title']}")
                
        except Exception as e:
            logging.error(f"Error storing in Supabase: {e}")
            
    def collect_daily_disclosures(self):
        """Collect and store daily disclosures"""
        today = datetime.now().strftime('%Y-%m-%d')
        logging.info(f"Starting disclosure collection for {today}")
        
        try:
            disclosures = self.get_disclosure_list(today)
            if disclosures:
                self.store_in_supabase(disclosures)
                logging.info(f"Successfully collected {len(disclosures)} disclosures")
            else:
                logging.info("No disclosures found for today")
                
        except Exception as e:
            logging.error(f"Error in daily collection: {e}")
            
    def run(self):
        """Run the collector"""
        self.collect_daily_disclosures()

if __name__ == "__main__":
    collector = DisclosureCollector()
    collector.run()
