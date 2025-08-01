import os
import sys
import json
from dotenv import load_dotenv

# Add the parent directory to the path so we can import app
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.services.disclosure_collector import DisclosureCollector

def main():
    # Load environment variables
    load_dotenv()
    
    # Initialize the collector
    collector = DisclosureCollector()
    
    try:
        print("Fetching fire insurance data...")
        data = collector.get_fire_insurance_data()
        
        # Print a summary of the response
        print("\nAPI Response Summary:")
        print(f"Response type: {type(data).__name__}")
        
        if isinstance(data, dict):
            print("\nTop-level keys:")
            for key in data.keys():
                print(f"- {key}")
                
            # If there's a list of items, show how many
            for key, value in data.items():
                if isinstance(value, list):
                    print(f"\nFound {len(value)} items in '{key}'")
                    if value:
                        print("First item structure:")
                        print(json.dumps(value[0], indent=2, ensure_ascii=False)[:500] + "...")
        
        # Save full response to file for inspection
        output_file = "fire_insurance_response.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"\nFull response saved to: {output_file}")
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
