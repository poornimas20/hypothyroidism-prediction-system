"""
Script to download UCI Hypothyroid Dataset
"""

import urllib.request
import os

def download_dataset():
    """Download the UCI Hypothyroid Dataset"""
    
    # URLs
    data_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/thyroid-disease/hypothyroid.data"
    names_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/thyroid-disease/hypothyroid.names"
    
    data_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Download data file
    print("Downloading hypothyroid.data...")
    try:
        urllib.request.urlretrieve(data_url, os.path.join(data_dir, "hypothyroid.data"))
        print("✓ hypothyroid.data downloaded successfully")
    except Exception as e:
        print(f"✗ Error downloading hypothyroid.data: {e}")
    
    # Download names file
    print("Downloading hypothyroid.names...")
    try:
        urllib.request.urlretrieve(names_url, os.path.join(data_dir, "hypothyroid.names"))
        print("✓ hypothyroid.names downloaded successfully")
    except Exception as e:
        print(f"✗ Error downloading hypothyroid.names: {e}")
    
    print("\nDataset download complete!")
    print(f"Files saved to: {data_dir}")

if __name__ == "__main__":
    download_dataset()
