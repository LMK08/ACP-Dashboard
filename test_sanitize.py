import pandas as pd
import numpy as np

def _safe_str(val):
    """The helper function we added to process_data.py"""
    if val is None:
        return None
    if isinstance(val, dict):
        # Handle the specific case where Wyscout returns a dict for a name
        return str(val.get('name', val.get('shortName', val)))
    return str(val)

def test_pandas_crash():
    print("🧪 STARTING TEST: Handling Dictionary values in Pandas Groupby\n")

    # 1. CREATE MESSY DATA
    # This simulates what happens when the API returns a dict for 'teamName'
    # and we pass it directly to the dataframe.
    data = [
        {'playerId': 101, 'minutes': 90, 'teamName': 'Braga II'},          # Clean string
        {'playerId': 102, 'minutes': 45, 'teamName': {'en': 'Paredes'}},   # ❌ DIRTY DICT (Causes crash)
        {'playerId': 101, 'minutes': 10, 'teamName': 'Braga II'},          # Duplicate player
    ]
    
    print("1. Created Dummy Data with mixed types (String vs Dict)...")
    df_raw = pd.DataFrame(data)
    print(df_raw)
    print("\n--------------------------------------------------")

    # 2. ATTEMPT CRASH (The "Before" Scenario)
    print("2. Attempting GroupBy WITHOUT fixing the data...")
    try:
        # This is the exact line that crashed in your error log
        # Grouping by a column that contains a dictionary raises TypeError
        bad_group = df_raw.groupby(['playerId', 'teamName'])['minutes'].sum()
        print("❌ SURPRISE: It worked? (This is unexpected if data is truly dirty)")
    except TypeError as e:
        print(f"✅ CRASH CONFIRMED: {e}")
        print("   (This proves the 'unhashable type: dict' error exists)")

    print("\n--------------------------------------------------")

    # 3. APPLY FIX (The "After" Scenario)
    print("3. Applying '_safe_str' fix to sanitize columns...")
    
    # Apply the helper function to the problem column
    df_raw['teamName'] = df_raw['teamName'].apply(_safe_str)
    
    print("   Data is now clean:")
    print(df_raw)
    
    print("\n4. Retrying GroupBy WITH clean data...")
    try:
        good_group = df_raw.groupby(['playerId', 'teamName'])['minutes'].sum().reset_index()
        print("✅ SUCCESS! GroupBy worked perfectly.")
        print(good_group)
    except Exception as e:
        print(f"❌ FAILED AGAIN: {e}")

if __name__ == "__main__":
    test_pandas_crash()