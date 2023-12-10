from utils import print_function_name, import_data, transform_numeric_target_feature_to_binary
import pickle
from sklearn.base import BaseEstimator
import numpy as np
import os
import pandas as pd

@print_function_name
def load_model_from_pickle(filename='wine_quality_classifier.model', verbose=True) -> BaseEstimator:
    """
    Loads a scikit-learn model from a pickle file.

    Parameters:
    filename (str): The path and name of the file from which the model will be loaded.

    Returns:
    BaseEstimator: The scikit-learn model loaded from the file.
    """
    with open(filename, 'rb') as file:
        model = pickle.load(file)
    if verbose:
        print(f"Loaded {filename} succesfully!")
    return model



@print_function_name
def import_train_statistics(filename='wine_quality_train_statistics.csv', verbose=True):
    the_train_statistics = pd.read_csv(filename, index_col=0)
    if verbose:
        print(f"Successfully loaded {filename}!")
    return the_train_statistics

from scipy import stats
import pandas as pd

def test_if_new_data_is_similar(train_stats, new_data, alpha=0.05):
    """
    Compares the mean of numerical features between provided train statistics and new data.

    Parameters:
    - train_stats (pd.DataFrame): A DataFrame with summary statistics of the train data (from df.describe()).
    - new_data (pd.DataFrame): The new incoming data to be compared.
    - alpha (float): Significance level for the t-test.

    Returns:
    - pd.DataFrame: A DataFrame with means of train and new data, their differences,
                    and a statistical test result if the means are not different.
    """
    comparison_results = pd.DataFrame()

    # Extract mean and std from train stats
    train_means = train_stats['mean']
    train_std = train_stats['std']

    # Get means for new data
    new_data_means = new_data.describe().T['mean']

    # Calculate the mean differences
    comparison_results['train_mean'] = train_means
    comparison_results['new_data_mean'] = new_data_means
    comparison_results['difference'] = new_data_means - train_means

    for feature in comparison_results.index:
        if feature in new_data.columns:
            # Perform a two-sample t-test for difference in means
            t_statistic, p_value = stats.ttest_ind_from_stats(
                mean1=train_means[feature], std1=train_std[feature], nobs1=train_stats['count'][feature],
                mean2=new_data_means[feature], std2=new_data[feature].std(), nobs2=len(new_data)
            )
            comparison_results.loc[feature, 'means_statistically_same'] = p_value > alpha

    return comparison_results



@print_function_name
def main():
    print("### Running a complete prediction pipline for wine quality. . . ")

    ## Import model and train statistics

    # Let's make sure the model is in the directory
    filename = 'wine_quality_classifier.model'
    print(filename if filename in os.listdir() else "The model is not in the directory")

    # import model
    model_prod = load_model_from_pickle(filename)
    print("This is our pre-trained, loaded model:")
    print(model_prod)
    print(model_prod.get_params())

    # import train statistics
    filename='wine_quality_train_statistics.csv'
    train_statistics = import_train_statistics(filename)
    print("These are our train statistics:")
    print(train_statistics)

    ## Import our dataset
    # Load the data
    try:
        df = import_data()
        # Proceed with training or other operations using df
    except FileNotFoundError as e:
        print(e)
        # Handle the error appropriately (e.g., exit the script or log the error)
        exit()

    ## Define target
    target = 'quality'
    if target in df:
        df = transform_numeric_target_feature_to_binary(df, target_col=target, threshold=7)

    ## Test if new data statistics are different than train
    alpha = 0.01  # significance level
    comparison_results = test_if_new_data_is_similar(train_statistics, df, alpha=alpha)
    print(comparison_results)



if __name__ == "__main__":
    main()
