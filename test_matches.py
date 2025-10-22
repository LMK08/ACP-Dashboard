# test_matches.py
import requests
from requests.auth import HTTPBasicAuth
import json

# --- Use the same credentials from your main script ---
username = "ggm0zzt-jidg1g1bv-ofdye2m-huk6ii8kkd"
password = ",Xzas52XAavPLHNK8sSJLJNhHEP!NY"
# ----------------------------------------------------

season_id = 191782
url = f"https://apirest.wyscout.com/v3/seasons/{season_id}/matches"
auth = HTTPBasicAuth(username, password)

print(f"Checking for matches in season {season_id} at URL: {url}")

try:
    r = requests.get(url, auth=auth, timeout=10)
    print(f"-> Status Code: {r.status_code}")

    # Pretty-print the JSON response
    response_data = r.json()
    print("-> Full API Response:")
    print(json.dumps(response_data, indent=4))

    # Check if the 'matches' list is empty
    if "matches" in response_data and not response_data["matches"]:
        print("\n✅ The API call worked, but the 'matches' list is empty.")
        print("   This confirms that there is no match data available for this season.")
    elif "matches" in response_data and response_data["matches"]:
         print("\n✅ Success! The API returned match data.")
    else:
        print("\n❌ The response did not contain the expected 'matches' key.")

except requests.exceptions.RequestException as e:
    print(f"\n❌ An error occurred: {e}")