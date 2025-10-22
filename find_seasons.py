# find_seasons.py
import requests
from requests.auth import HTTPBasicAuth
from tqdm import tqdm
import pandas as pd

# --- ⚠️ IMPORTANT: PASTE YOUR API CREDENTIALS HERE ---
username = "ggm0zzt-jidg1g5bv-ofdye2m-huk6ii8kkd"
password = ",Xzas52XAavPLHNK8sSJLJNhHEP!NY"
# ----------------------------------------------------

base_url = "https://apirest.wyscout.com/v4" # v4 is best for season info
auth = HTTPBasicAuth(username, password)

# Define a wide range of seasons to scan. You can adjust this.
season_range_to_test = range(197000, 198000) # Example: Scan 1000 recent seasons

accessible_seasons = []

print(f"Scanning seasons to find which ones are accessible...")
for season_id in tqdm(season_range_to_test):
    url = f"{base_url}/seasons/{season_id}"
    try:
        r = requests.get(url, auth=auth, timeout=5) # Added a timeout
        if r.status_code == 200 and r.json().get('competitionName'):
            season_data = r.json()
            accessible_seasons.append(season_data)
    except requests.exceptions.RequestException:
        continue # Ignore timeouts or connection errors

print(f"\nScan complete. You have access to {len(accessible_seasons)} seasons in the tested range.")

if accessible_seasons:
    df = pd.DataFrame(accessible_seasons)
    print("\nHere are the seasons you can access:")
    print(df[['seasonId', 'competitionName', 'name']].to_string())
    df.to_csv("accessible_seasons.csv", index=False)
    print("\n✅ A full list has been saved to 'accessible_seasons.csv'")
else:
    print("\n❌ No accessible seasons found. Please double-check your API credentials.")