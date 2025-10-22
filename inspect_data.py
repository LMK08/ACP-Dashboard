# inspect_data.py (v2)
import pickle
import pandas as pd

try:
    with open('all_match_data.pkl', 'rb') as f:
        all_match_data = pickle.load(f)

    print(f"✅ Data file loaded. Type: {type(all_match_data)}")

    if not all_match_data:
        print("❌ The loaded data dictionary is empty.")
    elif not isinstance(all_match_data, dict):
         print("❌ The loaded data is not a dictionary as expected.")
    else:
        print(f"   Number of matches found in data: {len(all_match_data)}")
        
        # Check the first few match IDs
        match_ids_to_check = list(all_match_data.keys())[:3] # Check up to 3 matches
        print(f"   Inspecting data for Match IDs: {match_ids_to_check}")

        for i, match_id in enumerate(match_ids_to_check):
            print(f"\n--- Checking Match {i+1} (ID: {match_id}) ---")
            match_data = all_match_data.get(match_id)

            if not match_data:
                print("   Match data not found for this ID.")
                continue

            # Check Team Stats
            if 'team_stats' in match_data and match_data['team_stats']:
                print("   --- Team Stats ---")
                if isinstance(match_data['team_stats'], dict) and match_data['team_stats']:
                    first_category = list(match_data['team_stats'].keys())[0]
                    first_df = match_data['team_stats'][first_category]
                    print(f"      First category: '{first_category}'")
                    if isinstance(first_df, pd.DataFrame):
                         print(f"      Columns: {first_df.columns.tolist()}")
                         print(f"      Index: {first_df.index.tolist()}")
                         print(f"      Example Row:\n{first_df.head(1).to_string()}")
                    else:
                         print("      Data for first category is not a DataFrame.")
                else:
                    print("      'team_stats' is not a non-empty dictionary.")
            else:
                print("   ❌ 'team_stats' key missing or empty.")

            # Check Player Stats
            if 'player_stats' in match_data and match_data['player_stats']:
                 print("   --- Player Stats ---")
                 if isinstance(match_data['player_stats'], dict) and 'home' in match_data['player_stats']:
                      home_df = match_data['player_stats']['home']
                      if isinstance(home_df, pd.DataFrame) and not home_df.empty:
                           print(f"      Home Player Columns: {home_df.columns.tolist()}")
                           print(f"      Example Row (Home):\n{home_df.head(1).to_string()}")
                      else:
                           print("      Home player stats DataFrame is not valid or is empty.")
                 else:
                      print("      'player_stats' dictionary is invalid or missing 'home' key.")
            else:
                print("   ❌ 'player_stats' key missing or empty.")

except FileNotFoundError:
    print("❌ Error: 'all_match_data.pkl' not found. Please run process_data.py first.")
except Exception as e:
    print(f"❌ An error occurred during inspection: {e}")