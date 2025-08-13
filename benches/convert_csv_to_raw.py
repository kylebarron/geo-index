import pandas as pd
import numpy as np
import os
import glob

def convert_csv_to_raw(csv_file):
    """Convert a CSV file with bounding box data to raw binary format."""
    print(f"Processing {csv_file}...")
    
    # Read CSV file - assuming no header and 4 columns: minx, miny, maxx, maxy
    df = pd.read_csv(csv_file, header=None, names=['minx', 'miny', 'maxx', 'maxy'])
    
    print(f"Loaded {len(df)} bounding boxes")
    
    # Convert to numpy array with float64 (double precision)
    bounds_array = df.values.astype(np.float64)
    
    print(f"Array shape: {bounds_array.shape}")
    
    # Convert to bytes in C-contiguous order (same as the original script)
    buf = bounds_array.tobytes("C")
    
    # Create output filename by replacing .csv with .raw
    raw_file = csv_file.replace('.csv', '.raw')
    
    # Write to raw file
    with open(raw_file, "wb") as f:
        f.write(buf)
    
    print(f"Wrote {len(buf)} bytes to {raw_file}")
    print()

def main():
    """Convert all CSV files in the current directory to raw format."""
    # Get all CSV files in the current directory
    csv_files = glob.glob("*.csv")
    
    if not csv_files:
        print("No CSV files found in the current directory.")
        return
    
    print(f"Found {len(csv_files)} CSV files:")
    for csv_file in csv_files:
        print(f"  - {csv_file}")
    print()
    
    # Convert each CSV file
    for csv_file in csv_files:
        try:
            convert_csv_to_raw(csv_file)
        except Exception as e:
            print(f"Error processing {csv_file}: {e}")
            print()

if __name__ == "__main__":
    main()
