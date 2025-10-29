# check_scores.py (Filter for Atlético CP)
import pandas as pd

team_to_check = "Atlético CP"

try:
    df = pd.read_parquet('matches_summary.parquet')

    # Filter for matches involving the specified team
    atletico_matches_df = df[
        (df['home_team'] == team_to_check) | (df['away_team'] == team_to_check)
    ].copy() # Use .copy() to avoid SettingWithCopyWarning if you modify it later

    if atletico_matches_df.empty:
        print(f"No matches found involving {team_to_check}.")
    else:
        print(f"--- All Matches Involving {team_to_check} ---")
        # Display relevant columns, including Gameweek for context
        print(atletico_matches_df[['Gameweek', 'home_team', 'away_team', 'score']].to_string())

except FileNotFoundError:
    print("❌ Error: 'matches_summary.parquet' not found.")
except Exception as e:
    print(f"❌ An error occurred: {e}")