import requests
import json

def test_fire_insurance_api():
    url = "https://kpub.knia.or.kr/popup/disclosureList.do"
    
    # Headers to mimic a browser request
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept': 'application/json, text/javascript, */*; q=0.01',
        'Content-Type': 'application/x-www-form-urlencoded; charset=UTF-8',
        'X-Requested-With': 'XMLHttpRequest',
        'Origin': 'https://kpub.knia.or.kr',
        'Referer': 'https://kpub.knia.or.kr/productDisc/longTermGuarantee/fireInsurance.do'
    }
    
    # Payload for the POST request
    payload = {
        'tabType': '1',
        'tptyCode': 'PB11',
        'refreshYn': '',
        'detailYn': '',
        'pCode': '',
        'channel': '',
        'prdNm': '',
        'payNm': '',
        'payReason': '',
        '__sort__': '1 asc'
    }
    
    # Disable SSL verification warnings
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    
    # Create a session with SSL verification disabled
    session = requests.Session()
    session.verify = False
    
    try:
        print("Sending request to:", url)
        response = session.post(url, headers=headers, data=payload, timeout=30)
        print(f"Status Code: {response.status_code}")
        
        # Try to parse as JSON
        try:
            data = response.json()
            print("Response (first 500 chars):")
            print(json.dumps(data, indent=2, ensure_ascii=False)[:500])
            
            # Save full response to file
            with open('api_response.json', 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            print("\nFull response saved to: api_response.json")
            
        except ValueError:
            print("Response is not JSON. Content:")
            print(response.text[:1000])  # Show first 1000 chars of non-JSON response
            
    except Exception as e:
        print(f"Error: {str(e)}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Response status: {e.response.status_code}")
            print(f"Response content: {e.response.text[:1000]}")

if __name__ == "__main__":
    test_fire_insurance_api()
