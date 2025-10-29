# check_order.py (Natural Order Check - No Date)
import pandas as pd

try:
    df = pd.read_parquet('matches_summary.parquet')

    print("--- First 15 Matches (Natural Order in File) ---")
    # Removed 'date' from the list of columns to print
    print(df[['matchId', 'home_team', 'away_team', 'Gameweek']].head(15).to_string())

    print("\n--- Last 15 Matches (Natural Order in File) ---")
    # Removed 'date' from the list of columns to print
    print(df[['matchId', 'home_team', 'away_team', 'Gameweek']].tail(15).to_string())

except FileNotFoundError:
    print("❌ Error: 'matches_summary.parquet' not found.")
except Exception as e:
    print(f"❌ An error occurred: {e}")