
import requests
import json

def debug_polymarket():
    print("Fetching LARGE list of markets for client-side filtering...")
    try:
        response = requests.get(
            "https://gamma-api.polymarket.com/markets",
            params={
                "active": "true", 
                "closed": "false", 
                "limit": 500, # Fetch more
                # "order": "volume24hr" # Optional: try to get high volume ones
            },
            timeout=15
        )
        markets = response.json()
        print(f"Fetched {len(markets)} markets.")
        
        btc_markets = [m for m in markets if 'bitcoin' in m.get('question', '').lower() or 'btc' in m.get('question', '').lower()]
        
        print(f"\nFound {len(btc_markets)} Bitcoin markets:")
        for m in btc_markets[:20]:
            print(f" - {m.get('question')} (ID: {m.get('id')})")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    debug_polymarket()
