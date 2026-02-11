import pandas as pd
import sys
import os
import argparse

def main():
    parser = argparse.ArgumentParser(description="Shuffle rows of a CSV file.")
    parser.add_argument("input_file", help="Path to the input CSV file")
    args = parser.parse_args()

    input_file = args.input_file
    
    # Derive output filename
    base, ext = os.path.splitext(input_file)
    output_file = f"{base}_shuffled{ext}"

    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found.")
        sys.exit(1)

    print(f"Reading {input_file}...")
    try:
        df = pd.read_csv(input_file)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        sys.exit(1)
    
    original_count = len(df)
    print(f"Original row count: {original_count}")

    print("Shuffling rows...")
    # Shuffle only the rows, reset index to avoid saving the old index
    # frac=1 means sample 100% of the rows (shuffle)
    # random_state=42 ensures reproducibility
    df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"Saving to {output_file}...")
    df_shuffled.to_csv(output_file, index=False)

    # Verification
    print("\n--- Verification ---")
    
    # 1. Check file existence
    if not os.path.exists(output_file):
        print("Error: Output file was not created.")
        sys.exit(1)
    
    # 2. Check row count
    try:
        df_new = pd.read_csv(output_file)
    except Exception as e:
        print(f"Error reading output CSV for verification: {e}")
        sys.exit(1)
        
    new_count = len(df_new)
    print(f"New row count: {new_count}")
    
    if original_count != new_count:
        print(f"FAILURE: Row counts do not match! ({original_count} vs {new_count})")
        sys.exit(1)
    else:
        print("SUCCESS: Row counts match.")

    # 3. Check content integrity
    print("Verifying content integrity...")
    
    cols = list(df.columns)
    
    # Sorting
    df_sorted = df.sort_values(by=cols).reset_index(drop=True)
    df_new_sorted = df_new.sort_values(by=cols).reset_index(drop=True)
    
    try:
        pd.testing.assert_frame_equal(df_sorted, df_new_sorted)
        print("SUCCESS: Content integrity verified (all rows preserved).")
    except AssertionError as e:
        print(f"FAILURE: Content mismatch!\n{e}")
        sys.exit(1)

    print("\nShuffling and verification complete successfully.")

if __name__ == "__main__":
    main()
