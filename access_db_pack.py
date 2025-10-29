# access_db_pack.py (Exploration Version)
import requests
from requests.auth import HTTPBasicAuth
import pandas as pd
import json
import os
from dotenv import load_dotenv

# --- Configuration ---
load_dotenv()
WYSOUT_USER = os.getenv("WYSCOUT_USER")
WYSOUT_PASS = os.getenv("WYSCOUT_PASS")
if not WYSOUT_USER or not WYSOUT_PASS:
    print("❌ Error: Wyscout credentials not found in .env file.")
    exit()

COMPETITION_ID = 43324 # Liga 3 Portugal
SEASON_ID = 191782    # 2025/2026
# You might need a specific team or player ID for some tests
EXAMPLE_TEAM_ID = 2697 # Example: Need a valid Team ID from Liga 3
EXAMPLE_PLAYER_ID = 3801 # Example: Need a valid Player ID

BASE_URL_V3 = "https://apirest.wyscout.com/v3"
auth = HTTPBasicAuth(WYSOUT_USER, WYSOUT_PASS)

# --- Helper Function for API Calls ---
def make_api_call(endpoint):
    """Makes a GET request to a Wyscout API endpoint and returns the JSON."""
    url = f"{BASE_URL_V3}/{endpoint}"
    print(f"\n--- Calling Endpoint: {url} ---")
    try:
        r = requests.get(url, auth=auth, timeout=20)
        r.raise_for_status() # Raise HTTP errors
        print(f"✅ Status Code: {r.status_code}")
        response_data = r.json()
        print("--- Response Structure (First Element or Keys): ---")
        if isinstance(response_data, list) and response_data:
            print(json.dumps(response_data[0], indent=4)) # Print first item if it's a list
        elif isinstance(response_data, dict):
            # Print keys or the whole dict if small
            if len(str(response_data)) < 1000:
                 print(json.dumps(response_data, indent=4))
            else:
                 print("Keys:", list(response_data.keys()))
        else:
             print("Response format not list or dict.")

        return response_data
    except requests.exceptions.HTTPError as errh:
        print(f"❌ HTTP Error: {errh}")
        print(f"   Response: {errh.response.text}")
    except requests.exceptions.RequestException as err:
        print(f"❌ Request Error: {err}")
    except json.JSONDecodeError:
        print(f"❌ Error decoding JSON response.")
        print(f"   Raw Response: {r.text if 'r' in locals() else 'Request failed'}")
    return None # Return None on failure

# --- Functions to Explore Specific Endpoints ---

def explore_competitions():
    """Gets a list of all competitions."""
    data = make_api_call("competitions")
    # You could add code here to save to CSV or further process

def explore_competition_seasons(comp_id):
    """Gets seasons for a specific competition."""
    data = make_api_call(f"competitions/{comp_id}/seasons")

def explore_season_teams(season_id):
    """Gets teams for a specific season."""
    data = make_api_call(f"seasons/{season_id}/teams")
    if data and isinstance(data.get('teams'), list):
         print("\n--- Team Names Found: ---")
         teams_list = [team.get('name') for team in data['teams'] if team.get('name')]
         print(teams_list)
         # Save to CSV if needed
         # pd.DataFrame(data['teams']).to_csv(f"season_{season_id}_teams.csv", index=False)

def explore_team_details(team_id):
    """Gets details for a specific team."""
    data = make_api_call(f"teams/{team_id}")

def explore_player_details(player_id):
    """Gets details for a specific player."""
    data = make_api_call(f"players/{player_id}")


# --- Main Execution ---
if __name__ == "__main__":
    print("Starting API Exploration...")

    # --- CHOOSE WHICH ENDPOINTS TO TEST ---
    # Uncomment the lines below one by one to explore different data

    explore_competitions()
    explore_competition_seasons(COMPETITION_ID)
    explore_season_teams(SEASON_ID)
    explore_team_details(EXAMPLE_TEAM_ID) # Requires a valid team ID
    explore_player_details(EXAMPLE_PLAYER_ID) # Requires a valid player ID

    print("\nAPI Exploration Finished.")