# test_single_season.py
import requests
from requests.auth import HTTPBasicAuth

# --- ⚠️ IMPORTANT: VERY CAREFULLY DOUBLE-CHECK AND PASTE YOUR CREDENTIALS HERE ---
# Look for typos. The password starts with a comma and ends with an exclamation mark.
username = "ggm0zzt-jidg1g5bv-ofdye2m-huk6ii8kkd"
password = ",Xzas52XAavPLHNK8sSJLJNhHEP!NY"
# --------------------------------------------------------------------------------

season_id_to_test = 191782
base_url = f"https://apirest.wyscout.com/v4/seasons/{season_id_to_test}"
auth = HTTPBasicAuth(username, password)

print(f"Attempting to access seasonId: {season_id_to_test}...")

try:
    r = requests.get(base_url, auth=auth, timeout=10)

    # Check the result
    if r.status_code == 200 and r.json().get('competitionName'):
        season_data = r.json()
        print("\n✅ SUCCESS! Access granted.")
        print(f"   Season ID: {season_data.get('seasonId')}")
        print(f"   Competition: {season_data.get('competitionName')}")
        print(f"   Season Name: {season_data.get('name')}")
    else:
        print(f"\n❌ FAILED. Access was denied for season {season_id_to_test}.")
        print(f"   Status Code: {r.status_code}")
        print(f"   Response: {r.text}")
        print("\n   This almost always means the username or password is incorrect.")

except requests.exceptions.RequestException as e:
    print(f"\n❌ An error occurred: {e}")