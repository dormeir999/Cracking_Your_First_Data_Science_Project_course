import sys
import os
import pickle
from sklearn.base import BaseEstimator
import os
from statsmodels.stats.proportion import proportions_ztest
from scipy import stats
import pandas as pd

# Add the 'src' directory to the Python path - needed when running from jupyter environment, etc.
sys.path.append(os.path.abspath('src'))
sys.path.append(os.path.abspath('../src'))

from utils import *

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


@print_function_name
def test_if_new_data_is_similar(the_train_statistics, new_data, alpha=0.05):
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
    train_means = the_train_statistics['mean']
    train_std = the_train_statistics['std']

    # Categorical: Modes and frequencies
    train_modes = the_train_statistics['top']
    train_mode_freq = the_train_statistics['freq']

    tests_cols = []
    # Process each feature
    for feature in the_train_statistics.index:
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
                prop_train = train_mode_freq[feature] / the_train_statistics.loc[feature, 'count']
                prop_new_data = freq_new_data / count_new_data
                comparison_results.loc[feature, 'freq_proportion_train'] = prop_train
                comparison_results.loc[feature, 'freq_proportion_new_data'] = prop_new_data

                # Statistical test for proportion difference
                z_score, p_value = proportions_ztest([train_mode_freq[feature], freq_new_data], [the_train_statistics.loc[feature, 'count'], count_new_data])
                comparison_results.loc[feature, 'proportions_statistically_same'] = p_value > alpha
                tests_cols = tests_cols + ['proportions_statistically_same'] if not 'proportions_statistically_same' in tests_cols else tests_cols

            else:  # Numerical feature
                new_data_mean = new_data[feature].mean()
                comparison_results.loc[feature, 'train_mean'] = train_means[feature]
                comparison_results.loc[feature, 'new_data_mean'] = new_data_mean
                comparison_results.loc[feature, 'difference'] = new_data_mean - train_means[feature]

                # Perform a two-sample t-test for difference in means
                t_statistic, p_value = stats.ttest_ind_from_stats(
                    mean1=train_means[feature], std1=train_std[feature], nobs1=the_train_statistics['count'][feature],
                    mean2=new_data_mean, std2=new_data[feature].std(), nobs2=len(new_data)
                )
                comparison_results.loc[feature, 'means_statistically_same'] = p_value > alpha
                tests_cols = tests_cols + ['means_statistically_same'] if not 'means_statistically_same' in tests_cols else tests_cols
    # report problems
    rows_of_features_not_similar = comparison_results[tests_cols] == False
    if rows_of_features_not_similar.sum().sum() > 0:
        statistically_not_the_same = comparison_results.loc[
        comparison_results[rows_of_features_not_similar].dropna(how='all').index].dropna(axis=1)
        print(f"""These new data features are not statistically similar to the train features: 
              {statistically_not_the_same}""")

    return comparison_results

@print_function_name
def drop_features_not_in_train_statistics(the_df, the_train_statistics):
    """
    Drops columns from df that are not listed in the index of the_train_statistics, if they do exist.
    Otherwise, reports: "No unknown features in new data"

    Parameters:
    - df (pd.DataFrame): The DataFrame from which columns will be dropped.
    - the_train_statistics (pd.DataFrame): A DataFrame containing statistics,
      where the index should be the column names of the original DataFrame.

    Returns:
    - pd.DataFrame: A DataFrame with only the columns that were present in the_train_statistics.
    """
    # Get a list of columns from the train statistics
    train_cols = the_train_statistics.index.tolist()

    # Identify columns in df that are not in the train statistics
    df_cols_not_in_train_statistics = [col for col in the_df.columns if col not in train_cols]

    if len(df_cols_not_in_train_statistics) > 0:
        # Drop the identified columns from df
        the_df = the_df.drop(columns=df_cols_not_in_train_statistics)
    else:
        print("No unknown features in new data")

    return the_df


@print_function_name
def main():
    print("### Running a complete prediction pipline for wine quality. . . ")
    ## Import our dataset
    # Load the data
    try:
        df = import_data()
        # Proceed with training or other operations using df
    except FileNotFoundError as e:
        print(e)
        # Handle the error appropriately (e.g., exit the script or log the error)
        exit()

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
    filename = 'wine_quality_train_statistics.csv'
    train_statistics = import_train_statistics(filename)
    print("These are our train statistics:")
    print(train_statistics)

    ## Define target
    target = 'quality'
    if target in df:
        df = transform_numeric_target_feature_to_binary(df, target_col=target, threshold=7)

    ## replaces spaces in feature names with underscores:
    df = replace_columns_spaces_with_underscores(df)

    ## drop features that were not part of train dataset
    df = drop_features_not_in_train_statistics(df, train_statistics)

    ## Test if new data statistics are different than train
    alpha = 0.01  # significance level
    comparison_results = test_if_new_data_is_similar(train_statistics, df, alpha=alpha)

    # Create new data statistics
    new_data_statistics = df.describe(include='all').T
    len_new_data = len(df)

    ## Missing Values imputation
    new_data_statistics['has_na'] = (new_data_statistics['count'] < len_new_data) * 1
    add_print = True

    # before imputing missing values, mark categories NA's as NA string
    categorical_features = get_train_features_with_property(train_statistics, 'is_categorical_to_drop')
    df, _ = replace_categoricals_missing_values_with_NA_string(df,
                                                            categorical_features=categorical_features,
                                                            NA_string=NA_string)
    # impute missing values
    df = imputate_missing_values('new_data', df, train_statistics, add_print=add_print)

    ## Handle Categoricals
    df, _ = one_hot_encode_categoricals(df, train_statistics,
                                            drop_one=drop_one)

    ## Handle Outliers
    # Apply outlier indicators
    # get train outlier columns
    train_outiler_cols = get_train_features_with_suffix(train_statistics, the_suffix=outlier_col_suffix)
    # if outliers exist in train, add outlier indicators to val and test in those specific features
    if len(train_outiler_cols) > 0:
        df, new_data_outlier_cols = add_outlier_indicators_on_features(df, the_train_statistics=train_statistics,
                                                      X_train_numeric_features=train_outiler_cols,
                                                      outlier_col_suffix=outlier_col_suffix)
        # Validate outliers detection: Test if train outlier statistics are different from new data outlier statistics
        comparison_results = test_if_new_data_is_similar(train_statistics, df[new_data_outlier_cols], alpha=alpha)

    # Impute outliers features (the function already checks if they exist)
    df = winsorize_outliers(df, train_statistics)
    pass

if __name__ == "__main__":
    print(os.getcwd())
    main()

#%%
