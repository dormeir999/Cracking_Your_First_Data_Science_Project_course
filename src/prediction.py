import sys
import os

# Add the 'src' directory to the Python path - needed when running from jupyter environment, etc.
sys.path.append(os.path.abspath('src'))
sys.path.append(os.path.abspath('../src'))


from utils import *
import pickle
from sklearn.base import BaseEstimator
import numpy as np
import os
import pandas as pd
from scipy import stats
from statsmodels.stats.proportion import proportions_ztest

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
    Compares the train data statistics with new incoming data for both numerical and categorical features.

    Parameters:
    - train_stats (pd.DataFrame): A DataFrame with summary statistics of the train data (from df.describe()).
    - new_data (pd.DataFrame): The new incoming data to be compared.
    - alpha (float): Significance level for the statistical tests.

    Returns:
    - pd.DataFrame: A DataFrame with comparison results including means, modes, and statistical test outcomes.
    """
    comparison_results = pd.DataFrame()

    # Numerical: Extract mean and std from train stats
    train_means = train_stats['mean']
    train_std = train_stats['std']

    # Categorical: Modes and frequencies
    train_modes = train_stats['top']
    train_mode_freq = train_stats['freq']

    # Process each feature
    for feature in train_stats.index:
        if feature in new_data.columns:
            if new_data[feature].dtype == 'object':  # Categorical feature
                mode_new_data = new_data[feature].mode()[0]
                freq_new_data = new_data[feature].value_counts()[mode_new_data]
                count_new_data = len(new_data[feature])

                # Compare number of unique values and the mode
                comparison_results.loc[feature, 'train_mode'] = train_modes[feature]
                comparison_results.loc[feature, 'new_data_mode'] = mode_new_data
                comparison_results.loc[feature, 'modes_match'] = (train_modes[feature] == mode_new_data)

                # Compare frequency proportions
                prop_train = train_mode_freq[feature] / train_stats.loc[feature, 'count']
                prop_new_data = freq_new_data / count_new_data
                comparison_results.loc[feature, 'freq_proportion_train'] = prop_train
                comparison_results.loc[feature, 'freq_proportion_new_data'] = prop_new_data

                # Statistical test for proportion difference
                z_score, p_value = proportions_ztest([train_mode_freq[feature], freq_new_data], [train_stats.loc[feature, 'count'], count_new_data])
                comparison_results.loc[feature, 'proportions_statistically_same'] = p_value > alpha

            else:  # Numerical feature
                new_data_mean = new_data[feature].mean()
                comparison_results.loc[feature, 'train_mean'] = train_means[feature]
                comparison_results.loc[feature, 'new_data_mean'] = new_data_mean
                comparison_results.loc[feature, 'difference'] = new_data_mean - train_means[feature]

                # Perform a two-sample t-test for difference in means
                t_statistic, p_value = stats.ttest_ind_from_stats(
                    mean1=train_means[feature], std1=train_std[feature], nobs1=train_stats['count'][feature],
                    mean2=new_data_mean, std2=new_data[feature].std(), nobs2=len(new_data)
                )
                comparison_results.loc[feature, 'means_statistically_same'] = p_value > alpha
    # report problems
    tests_cols = ['proportions_statistically_same', 'means_statistically_same']
    statistically_not_the_same = comparison_results.loc[
        comparison_results[comparison_results[tests_cols] == False].dropna(how='all').index].dropna(axis=1)
    if len(statistically_not_the_same) > 0:
        print(f"""These new data features are not statistically similar to the train features: 
              {statistically_not_the_same}""")

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

    ## replaces spaces in feature names with underscores:
    df = replace_columns_spaces_with_underscores(df)
    ## Test if new data statistics are different than train
    alpha = 0.01  # significance level
    comparison_results = test_if_new_data_is_similar(train_statistics, df, alpha=alpha)




if __name__ == "__main__":
    print(os.getcwd())
    main()
