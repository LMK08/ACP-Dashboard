import pandas as pd

def check_event_columns():
    try:
        # Load the events file
        df = pd.read_parquet('raw_events.parquet')
        
        print("✅ loaded raw_events.parquet")
        print(f"Columns: {list(df.columns)}")
        
        # Check for any column that looks like a position
        pos_cols = [c for c in df.columns if 'pos' in c.lower() or 'role' in c.lower()]
        print(f"Potential Position Columns: {pos_cols}")
        
        # Check for team name column
        if 'team.name' in df.columns:
            print("✅ 'team.name' exists!")
            print("Sample:", df['team.name'].dropna().unique()[:3])
        else:
            print("❌ 'team.name' MISSING")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    check_event_columns()