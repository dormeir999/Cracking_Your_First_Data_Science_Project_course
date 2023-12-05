import os

print(os.getcwd())
# # Import data, set target
# filename = "winequalityN.csv"
# if filename in os.listdir():
#     df = pd.read_csv(filename)
# elif filename in
# else:
#     df = pd.read_csv("/usercode/" + filename)
#
#
# target = 'quality'

import pandas as pd
import os


import pandas as pd
import os

def import_data(filename: str = "winequalityN.csv", data_dir: str = "../data/raw/") -> pd.DataFrame:
    """
    Imports data from a specified file located in a given directory.

    Parameters:
    - filename (str): Name of the file to be imported. Defaults to "winequalityN.csv".
    - data_dir (str): Relative path to the directory containing the data file. Defaults to "../data/raw/".

    Returns:
    - pd.DataFrame: DataFrame containing the imported data.
    """
    # Construct the full file path
    file_path = os.path.join(data_dir, filename)

    # Check if the file exists
    if not os.path.isfile(file_path):  # Try the educative coding environment location
        file_path = "/usercode/" + filename
        if not os.path.isfile(file_path):
            print(f"File not found: {file_path}")
            raise FileNotFoundError(f"File not found: {file_path}")

    # Read and return the data
    print("Data imported successfully!")
    return pd.read_csv(file_path)


def main():
    # Load the data
    try:
        df = import_data()
        # Proceed with training or other operations using df
    except FileNotFoundError as e:
        print(e)
        # Handle the error appropriately (e.g., exit the script or log the error)

if __name__ == "__main__":
    main()