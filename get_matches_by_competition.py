# get_matches_by_competition.py
import requests
from requests.auth import HTTPBasicAuth
import pandas as pd
import json

# --- ⚠️ PASTE YOUR CREDENTIALS HERE ---
username = "ggm0zzt-jidg1g5bv-ofdye2m-huk6ii8kkd"
password = ",Xzas52XAavPLHNK8sSJLJNhHEP!NY"
# ------------------------------------

# --- The IDs we know from our tests ---
competition_id = 43324
season_id_to_find = 191782
# ------------------------------------

url = f"https://apirest.wyscout.com/v3/competitions/{competition_id}/matches"
auth = HTTPBasicAuth(username, password)

print(f"Attempting to fetch matches for competitionId: {competition_id}...")

try:
    r = requests.get(url, auth=auth, timeout=15)
    print(f"-> Status Code: {r.status_code}")

    if r.status_code == 200:
        all_matches_data = r.json().get("matches", [])
        
        if not all_matches_data:
            print("-> The API returned an empty list of matches for this competition.")
        else:
            # Filter the list to find matches belonging to our specific season
            season_matches = [
                match for match in all_matches_data 
                if match.get("seasonId") == season_id_to_find
            ]
            
            print(f"\n✅ Success! Found {len(season_matches)} matches for seasonId {season_id_to_find}.")
            
            if season_matches:
                # Create a DataFrame and save it
                matches_df = pd.json_normalize(season_matches)
                filename = f"match_ids_for_season_{season_id_to_find}.csv"
                matches_df.to_csv(filename, index=False)
                print(f"-> A list of these matches has been saved to '{filename}'")
    else:
        print("\n❌ FAILED. The API returned an error.")
        print("   Response:", r.text)

except requests.exceptions.RequestException as e:
    print(f"\n❌ An error occurred: {e}")