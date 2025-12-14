import pandas as pd
import numpy as np
import os

def optimize_file(filename):
    print(f"📉 Optimizing {filename}...")
    
    if not os.path.exists(filename):
        print(f"   ⚠️ {filename} not found.")
        return

    # Load the full file
    df = pd.read_parquet(filename)
    original_mem = df.memory_usage(deep=True).sum() / 1024**2
    
    # 1. DROP UNUSED COLUMNS (Keep only what matches current data)
    if 'events' in filename:
        try:
            current_df = pd.read_parquet('raw_events.parquet')
            keep_cols = set(current_df.columns)
            # Ensure critical columns are always kept
            keep_cols.update(['matchId', 'player.id', 'team.name', 'player.name'])
            
            # Identify columns to drop
            existing_cols = [c for c in df.columns if c in keep_cols]
            df = df[existing_cols]
        except:
            print("   Could not load reference 'raw_events.parquet', skipping column drop.")

    # 2. DOWNCAST NUMBERS (Float64 -> Float32)
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = df[col].astype('float32')
    
    for col in df.select_dtypes(include=['int64']).columns:
        df[col] = pd.to_numeric(df[col], downcast='integer')

    # 3. OPTIMIZE STRINGS (Categoricals) - WITH SAFETY CHECK
    for col in df.select_dtypes(include=['object']).columns:
        try:
            # Check if values are unhashable (lists/arrays) before converting
            # We take a sample to check type
            sample = df[col].dropna().iloc[0] if not df[col].dropna().empty else None
            if isinstance(sample, (list, np.ndarray, dict)):
                continue # Skip complex columns (tags, positions, etc.)

            num_unique = len(df[col].unique())
            num_total = len(df)
            if num_unique / num_total < 0.5: 
                df[col] = df[col].astype('category')
        except TypeError:
            # If unique() fails (unhashable type), just skip this column
            continue
        except Exception:
            continue

    # Save back to the same filename
    df.to_parquet(filename)
    
    new_mem = df.memory_usage(deep=True).sum() / 1024**2
    print(f"   ✅ Done! Reduced from {original_mem:.2f} MB to {new_mem:.2f} MB")

if __name__ == "__main__":
    optimize_file('historical_events.parquet')
    optimize_file('historical_matches.parquet')
    optimize_file('raw_events.parquet')
    print("\n🚀 All files optimized. You can now push to GitHub.")