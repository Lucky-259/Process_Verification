import pandas as pd
import os
import sys

def convert_json_to_parquet(json_file, output_dir=None):
    """
    Convert a JSON file to a Parquet file.
    
    Args:
        json_file (str): Path to the input JSON file.
        output_dir (str): Path to the output dictionary.
    """
    if output_dir is None:
        output_dir = os.path.dirname(json_file)
    base_name = os.path.basename(json_file).split('.')[0]

    # Read the JSON file into a DataFrame
    print(f"Reading {json_file}...")
    df = pd.read_json(json_file, orient='records')

    # Get the parquet file path
    parquet_path = os.path.join(output_dir, f"{base_name}.parquet")
    
    # Write the DataFrame to a Parquet file
    print(f"Saving to PARQUET: {parquet_path}")
    df.to_parquet(parquet_path)

    # Print basic info
    print(f"\nConverted file with {len(df)} rows and {len(df.columns)} columns")
    
    return parquet_path

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python convert_json_to_parquet.py <json_file_path> [output_directory]")
        sys.exit(1)
    
    json_file = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None
    
    parquet_path = convert_json_to_parquet(json_file, output_dir)
    print(f"\nFiles saved to:\n- {parquet_path}")