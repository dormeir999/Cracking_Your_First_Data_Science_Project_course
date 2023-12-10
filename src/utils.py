import os
import pandas as pd


def print_function_name(func):
    """
    Decorator that prints the name of the function when it is called.

    Parameters:
    - func (callable): The function to be decorated.

    Returns:
    - callable: The wrapped function.
    """

    def wrapper(*args, **kwargs):
        print(f"{func.__name__}()")
        return func(*args, **kwargs)

    return wrapper

@print_function_name
def import_data(filename: str = "winequalityN.csv", data_dir: str = "data/raw/") -> pd.DataFrame:
    """
    Imports data from a specified file located in a given directory.

    Parameters:
    - filename (str): Name of the file to be imported. Defaults to "winequalityN.csv".
    - data_dir (str): Relative path to the directory containing the data file. Defaults to "../data/raw/".

    Returns:
    - pd.DataFrame: DataFrame containing the imported data.
    """
    # Determine the path to the directory containing this script
    module_dir = os.getcwd()
    if os.path.split(os.getcwd())[-1] == 'src':
        os.chdir("..")
        module_dir = os.getcwd()
    # Construct the path to the data file
    data_dir = os.path.join(module_dir, data_dir)
    file_path = os.path.join(data_dir, filename)
    print("Attempting to load data from:", file_path)

    # Check if the file exists
    if not os.path.isfile(file_path):  # If doesn't exit, try the educative coding environment location
        file_path = "/usercode/" + filename
        if not os.path.isfile(file_path):
            print(f"File not found: {file_path}")
            raise FileNotFoundError(f"File not found: {file_path}")

    # Read and return the data
    print("Data imported successfully!")
    return pd.read_csv(file_path)

@print_function_name
def transform_numeric_target_feature_to_binary(the_df: pd.DataFrame, target_col: str = 'quality',
                                               threshold: int = 7) -> pd.DataFrame:
    """
   Transform a numeric target feature in a DataFrame into a binary representation.

   Parameters:
   - the_df (pd.DataFrame): DataFrame containing the target feature.
   - target_col (str): Name of the target column. Defaults to 'quality'.
   - threshold (int): Threshold value for binarization. Defaults to 7.

   Returns:
   - pd.DataFrame: Modified DataFrame with the target feature binarized.
   """
    the_df[target_col] = (the_df[target_col] >= threshold) * 1

    return the_df