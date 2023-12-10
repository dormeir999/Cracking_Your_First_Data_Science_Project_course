from datetime import datetime
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_curve, roc_auc_score, RocCurveDisplay, confusion_matrix, classification_report, f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from scipy import stats
from scipy.stats import skew, kurtosis
from functools import partial
from sklearn.manifold import TSNE


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


@print_function_name
def split_dataset(the_df: pd.DataFrame, target_col: str, the_test_size: float = 0.2, the_random_state: int = 42) -> (
        pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame):
    """
    Split a DataFrame into training, validation, and testing sets, while maintaining similar target distribution in all.

    Parameters:
    - the_df (pd.DataFrame): DataFrame to be split.
    - the_target (str): Name of the target column.
    - the_test_size (float): Proportion of the dataset to include in the test split. Defaults to 0.2.
    - the_random_state (int): Controls the shuffling for reproducible output. Defaults to 42.

    Returns:
    - tuple: Six DataFrames in the following order: X_train, X_test, X_val, y_train, y_test, y_val.
    """
    the_X = the_df.drop(columns=target_col)
    the_y = the_df[[target_col]]

    # Split into train and test
    the_X_train, the_X_test, the_y_train, the_y_test = train_test_split(
        the_X, the_y, test_size=the_test_size, stratify=the_y, shuffle=True, random_state=the_random_state
    )

    # Split train into train and validation
    the_X_train, the_X_val, the_y_train, the_y_val = train_test_split(
        the_X_train, the_y_train, test_size=the_test_size, stratify=the_y_train, shuffle=True,
        random_state=the_random_state
    )

    return the_X_train, the_X_test, the_X_val, the_y_train, the_y_test, the_y_val


@print_function_name
def test_if_significant(p_value, alpha=0.05, print_text=False):
    """
    Tests if a given p-value is significant based on a specified alpha level.

    Parameters:
    - p_value (float): The p-value to be tested.
    - alpha (float, optional): The significance level threshold. Defaults to 0.05.
    - print_text (bool, optional): If True, prints whether the means are significantly different. Defaults to False.

    Returns:
    - bool: True if p_value is less than alpha (significant), else False.
    """
    if p_value < alpha:
        if print_text:
            print(f"The means of the two populations are significantly different (alpha={alpha}).")
        return True
    else:
        if print_text:
            print("The means of the two populations are NOT significantly different (alpha={alpha}).")
        return False


@print_function_name
def test_if_features_statistically_different(the_X_train, dfs_dict, alpha=0.05):
    '''Check if mean of numerical features in X_train and dfs_dict are statistically the same, for specified significance level
       return a df means, their difference and an answer the the question:
       Are the means statistically not different?'''
    # get train numeric features means
    train_val_outlier_means = the_X_train.describe().T.add_suffix('_train')['mean_train']
    for the_df_name, the_df in dfs_dict.items():
        # get other df numeric features means
        the_df_name = the_df_name.split("X_")[1]
        X_df_outlier_means = the_df.describe().T.add_suffix(f'_{the_df_name}')[f'mean_{the_df_name}']
        # concat the other df means to the same dataframe
        train_val_outlier_means = pd.concat([train_val_outlier_means, X_df_outlier_means], axis=1)
        # calc the mean for both, just to get a sense of the size of difference
        train_val_outlier_means['difference'] = (
                train_val_outlier_means['mean_train'] - train_val_outlier_means[f'mean_{the_df_name}']).round(3)
        for feature in train_val_outlier_means.index:
            # test the normality of the difference
            statatistic, p_value = stats.shapiro(the_X_train[feature] - the_df[feature])
            train_val_outlier_means.loc[
                feature, f'{the_df_name} difference is normal with {int((1 - alpha) * 100)}% significance'] = not test_if_significant(
                p_value, alpha=alpha)
            # perform the two-sample t-test for difference in means,
            t_statistic, p_value = stats.ttest_ind(the_X_train[feature], the_df[feature])
            train_val_outlier_means.loc[
                feature, f'{the_df_name} mean is the same with {int((1 - alpha) * 100)}% significance'] = not test_if_significant(
                p_value, alpha=alpha)
    return train_val_outlier_means


@print_function_name
def imputate_missing_values(dataset_name, dataset, the_train_statistics, n_rows_to_show=5, add_print=True):
    """
    Impute missing values in a dataset using the mean values from training statistics.

    Parameters:
    - dataset_name (str): Name of the dataset being processed.
    - dataset (pd.DataFrame): The dataset for imputation.
    - the_train_statistics (pd.DataFrame): Training statistics for reference.
    - n_rows_to_show (int): Number of rows to display for demonstration. Defaults to 5.
    - add_print (bool): Flag to print the demonstration rows. Defaults to True.

    Returns:
    - pd.DataFrame: Dataset with missing values imputed.
    """
    # get a dict of missing values per each feature
    mean_values = the_train_statistics["mean"].to_dict()
    # impute missing values, and save those indexes
    missing_indexes = dataset[dataset.isna().any(axis=1)].index
    if add_print:
        to_show = dataset.loc[missing_indexes][:n_rows_to_show]
        print(f"# First {n_rows_to_show} original {dataset_name} missing values:\n{to_show}\n")
    # fill the missing values in X_train with the mean values
    dataset = dataset.fillna(value=mean_values)
    if add_print:
        to_show = dataset.loc[missing_indexes][:n_rows_to_show]
        print(f"# First {n_rows_to_show} imputed {dataset_name} missing values:\n{to_show}\n")
        missing_values = dataset.isna().sum()
        print(f"# The number of missing values in columns in {dataset_name}:\n{missing_values}\n")
    return dataset


@print_function_name
def one_hot_encode_categoricals(the_df, categorical_features=None, drop_one=True, categories_to_use_from_train=None):
    """
    Perform one-hot encoding on categorical features in a DataFrame.

    Parameters:
    - the_df (pd.DataFrame): DataFrame to be encoded.
    - categorical_features (list): List of categorical feature names. Defaults to ['type'].
    - drop_one (bool): Flag to drop the first category to avoid multicollinearity. Defaults to True.
    - categories_to_use_from_train (list): List of categories used in training for consistency.

    Returns:
    - tuple: Modified DataFrame and a list of new category columns.
    """
    # Create the numerical encodings
    if categorical_features is None:
        categorical_features = ['type']
    one_hot_encodings = pd.get_dummies(the_df[categorical_features], drop_first=drop_one)
    # if val or test, filter categories to the ones that were used in train
    if categories_to_use_from_train:
        one_hot_encodings = one_hot_encodings[categories_to_use_from_train]
    one_hot_encodings_categories = one_hot_encodings.columns.tolist()
    # Add the encodings to the dataset
    the_df = pd.concat([the_df, one_hot_encodings], axis=1)
    # Drop the original categorical_features
    the_df = the_df.drop(columns=categorical_features)
    return the_df, one_hot_encodings_categories


@print_function_name
def add_new_features_statistics_to_train_statistics(the_train, the_train_statistics, new_features):
    """
    Add descriptive statistics for newly created features to the training statistics table.

    Parameters:
    - the_train (pd.DataFrame): The training dataset.
    - the_train_statistics (pd.DataFrame): DataFrame containing statistics for training data.
    - new_features (list): List of new feature names added to the dataset.

    Returns:
    - pd.DataFrame: Updated training statistics with new features included.
    """
    train_new_features_statistics = the_train[new_features].describe(include='all').T
    the_train_statistics = pd.concat([the_train_statistics, train_new_features_statistics], axis=0)
    return the_train_statistics


@print_function_name
def add_binary_property_to_train_statistics(the_train_statistics, the_property, features_list_with_property):
    """
    Add a binary property to a group of features in the training statistics table.

    Parameters:
    - the_train_statistics (pd.DataFrame): DataFrame of training statistics.
    - the_property (str): Name of the binary property to be added.
    - features_list_with_property (list): List of features to which the property applies.

    Returns:
    - pd.DataFrame: Updated training statistics with the new property.
    """
    if not the_property in the_train_statistics:
        the_train_statistics[the_property] = 0
    if len(features_list_with_property) == 1:
        features_list_with_property = features_list_with_property[0]
    the_train_statistics.loc[features_list_with_property, the_property] = 1
    return the_train_statistics


@print_function_name
def get_train_feautres_with_property(the_train_statistics, the_property):
    """
    Extracts a list of features from the train statistics DataFrame that have a specified property.

    Parameters:
    - the_train_statistics (pd.DataFrame): DataFrame containing training statistics.
    - the_property (str): The property based on which to filter features.

    Returns:
    - list: List of feature names that have the specified property.
    """
    the_features = the_train_statistics[the_train_statistics[the_property] == 1].index.tolist()
    return the_features


@print_function_name
def add_kurtosis_skew_statistics(the_df, the_train_statistics):
    """
    Add kurtosis and skew statistics to the training statistics table.

    Parameters:
    - the_df (pd.DataFrame): DataFrame containing the data.
    - the_train_statistics (pd.DataFrame): DataFrame of training statistics.

    Returns:
    - pd.DataFrame: Updated training statistics with kurtosis and skewness metrics.
    """
    the_train_statistics['kurtosis'] = the_df.apply(kurtosis)
    the_train_statistics['skew'] = the_df.apply(skew)
    the_train_statistics['is_many_outliers'] = (the_train_statistics['kurtosis'] >= 3) * 1
    the_train_statistics['is_right_skew'] = (the_train_statistics['skew'] > 0) * 1
    return the_train_statistics


@print_function_name
def add_outlier_indicator(the_df: pd.DataFrame, the_feature: pd.Series, the_train_statistics: pd.DataFrame,
                          outlier_col_suffix='is_outlier', is_train=False) -> pd.DataFrame:
    """
    Add an outlier indicator column for a specific feature in the DataFrame.
    Outliers are defined as points distant more than 3 std from the mean.

    Parameters:
    - the_df (pd.DataFrame): DataFrame to be processed.
    - the_feature (str): Name of the feature to check for outliers.
    - the_train_statistics (pd.DataFrame): Training statistics for reference.
    - outlier_col_suffix (str): Suffix for the outlier column name. Defaults to 'is_outlier'.
    - is_train (bool): Flag to indicate if the dataset is training data. Defaults to False.

    Returns:
    - tuple: Modified DataFrame and the name of the new outlier column.
    """
    outlier_col = the_feature + "_" + outlier_col_suffix
    # Create is_outlier col if it doesn't exist, and fill with 0 (i.e. no outliers)
    if not outlier_col in the_df:
        the_df[outlier_col] = 0
    # The formula for calculating a z-score is: Z = (X - μ) / σ
    X = the_df[the_feature]
    mu = the_train_statistics.loc[the_feature, 'mean']
    sigma = the_train_statistics.loc[the_feature, 'std']
    obs_z_scores = (X - mu) / sigma
    # Get all rows with outliers
    outliers = obs_z_scores.abs() > 3
    # Mark outliers
    if sum(outliers) > 0:
        the_df.loc[outliers, outlier_col] = 1
    else:
        if is_train:  # if train and no outliers, drop column. if val or test, keep zeros.
            the_df = the_df.drop(columns=outlier_col)

    return the_df, outlier_col


@print_function_name
def add_outlier_indicators_on_features(the_df: pd.DataFrame, the_train_statistics: pd.DataFrame,
                                       X_train_numeric_features: List = None,
                                       outlier_col_suffix='is_outlier') -> pd.DataFrame:
    """
    Add outlier indicator columns for multiple features in the DataFrame.

    Parameters:
    - the_df (pd.DataFrame): DataFrame to be processed.
    - the_train_statistics (pd.DataFrame): Training statistics for reference.
    - X_train_numeric_features (List): List of numeric feature names to check for outliers.
    - outlier_col_suffix (str): Suffix for the outlier column names. Defaults to 'is_outlier'.

    Returns:
    - tuple: Modified DataFrame and a list of new outlier column names.
    """
    # If the_features not defined (first run on train), filter out non-numeric features and run on all
    if not X_train_numeric_features:
        is_train = True
        categories = get_train_feautres_with_property(the_train_statistics, 'is_category')
        X_train_numeric_features = [col for col in the_df.columns if not col in categories]
    else:
        is_train = False  # either validation or test
    new_outlier_cols = []
    for feature in X_train_numeric_features:
        the_df, new_outlier_col = add_outlier_indicator(the_df, feature, the_train_statistics,
                                                        outlier_col_suffix=outlier_col_suffix, is_train=is_train)
        new_outlier_cols = new_outlier_cols + [new_outlier_col]
    return the_df, new_outlier_cols


@print_function_name
def get_train_features_with_suffix(the_train_statistics, remove_suffix=True, the_suffix='is_outlier'):
    """
    Extracts a list of features from the train statistics DataFrame that have a specified suffix.

    Parameters:
    - the_train_statistics (pd.DataFrame): DataFrame containing training statistics.
    - remove_suffix (bool, optional): If True, removes the suffix from the feature names. Defaults to True.
    - the_suffix (str, optional): The suffix to filter features. Defaults to 'is_outlier'.

    Returns:
    - list: List of feature names with the specified suffix, optionally without the suffix.
    """
    the_train_statistics_features = the_train_statistics.index.to_list()
    feautres_with_suffix = [feature for feature in the_train_statistics_features if feature.endswith(the_suffix)]
    if remove_suffix:
        feautres_with_suffix = [feature.split("_" + the_suffix)[0] for feature in feautres_with_suffix]
    return feautres_with_suffix


@print_function_name
def add_winsorization_values_to_train_statistics(the_X_train, the_train_statistics, percentiles=None):
    """
    Add winsorization percentile values to the training statistics for outlier handling.

    Parameters:
    - the_X_train (pd.DataFrame): The training dataset.
    - the_train_statistics (pd.DataFrame): Training statistics for reference.
    - percentiles (list): List of percentiles to be used for winsorization. Defaults to [.05, .95].

    Returns:
    - pd.DataFrame: Updated training statistics with winsorization values.
    """
    if percentiles is None:
        percentiles = [.05, .95]
    winsorization_values = the_X_train.describe(include='all', percentiles=[.05, .95]).T
    percentile_col_names = [str(col).split(".")[1].replace("0", "") + "%" for col in percentiles]
    the_train_statistics = the_train_statistics.join(winsorization_values[percentile_col_names], how='left')
    return the_train_statistics


@print_function_name
def winsorize_outliers(the_df, the_train_statistics, percentiles=None, outlier_col_suffix='is_outlier'):
    """
    Apply winsorization to handle outliers in the DataFrame based on training statistics.

    Parameters:
    - the_df (pd.DataFrame): DataFrame containing the data.
    - the_train_statistics (pd.DataFrame): Training statistics with winsorization values.
    - percentiles (list): List of percentiles used for winsorization. Defaults to [.05, .95].

    Returns:
    - pd.DataFrame: The DataFrame after applying winsorization to outliers.
    """
    # extract original outlier call and is_outliers cols
    if percentiles is None:
        percentiles = [.05, .95]
    remove_suffix = False
    train_outlier_cols = get_train_features_with_suffix(the_train_statistics, the_suffix=outlier_col_suffix,
                                                        remove_suffix=remove_suffix)
    remove_suffix = True
    train_orig_outlier_cols = get_train_features_with_suffix(the_train_statistics, the_suffix=outlier_col_suffix,
                                                             remove_suffix=remove_suffix)
    outlier_cols_mapper = dict(zip(train_orig_outlier_cols, train_outlier_cols))
    # extract winsorization values
    percentile_col_names = [str(col).split(".")[1].replace("0", "") + "%" for col in percentiles]
    winsorization_values = the_train_statistics.loc[train_orig_outlier_cols, percentile_col_names].T
    # replace min/max outliers with min_winzor/max_winzor
    for orig_col, is_outlier_col in outlier_cols_mapper.items():
        min_winzor = winsorization_values[orig_col].min()
        max_winzor = winsorization_values[orig_col].max()
        outlier_rows = the_df[is_outlier_col] == 1
        min_outliers = the_df[orig_col] <= min_winzor
        max_outliers = the_df[orig_col] >= max_winzor
        the_df.loc[(outlier_rows) & (min_outliers), orig_col] = min_winzor
        the_df.loc[(outlier_rows) & (max_outliers), orig_col] = max_winzor

    return the_df


@print_function_name
def add_polynomial_features(the_df, features_to_add=None, suffix='_squared'):
    """
    Add polynomial features to the DataFrame.

    Parameters:
    - the_df (pd.DataFrame): DataFrame to be processed.
    - features_to_add (list): List of feature names to apply polynomial transformation. Defaults to None.
    - suffix (str): Suffix to be added to the new polynomial feature names. Defaults to '_squared'.

    Returns:
    - tuple: Modified DataFrame and a list of new polynomial feature names.
    """
    if features_to_add is None:
        features_to_add = ['alcohol', 'density']
    new_features_list = []
    for feature in features_to_add:
        new_feature = feature + suffix
        the_df[new_feature] = the_df[feature] ** 2
        new_features_list = new_features_list + [new_feature]
    return the_df, new_features_list


@print_function_name
def add_combination_features_wine_dataset(the_df, features_to_add=None):
    """
    Add combination features to the DataFrame based on specified formulas or operations of wine dataset features.

    Parameters:
    - the_df (pd.DataFrame): DataFrame to be processed.
    - features_to_add (list): List of feature combinations to be added. Defaults to None.

    Returns:
    - tuple: Modified DataFrame and a list of new combination feature names.
    """
    if features_to_add is None:
        features_to_add = ['total acidity', 'combined sulfur dioxide', 'mso2']
    new_features_list = features_to_add
    for feature in features_to_add:
        if feature == 'total acidity':
            the_df[feature] = the_df['fixed acidity'] + the_df['volatile acidity']
        if feature == 'combined sulfur dioxide':
            the_df[feature] = the_df['total sulfur dioxide'] - the_df['free sulfur dioxide']
        if feature == 'mso2':
            the_df[feature] = (1 + 10 ** (the_df['pH'] - 1.81))
    return the_df, new_features_list


@print_function_name
def add_interaction_features(the_df, features_tuples_list_to_add=None):
    """
    Add interaction features to the DataFrame by multiplying pairs of existing features.

    Parameters:
    - the_df (pd.DataFrame): DataFrame to be processed.
    - features_tuples_list_to_add (list): List of tuples, each containing a pair of features for interaction. Defaults to None.

    Returns:
    - tuple: Modified DataFrame and a list of new interaction feature names.
    """
    if features_tuples_list_to_add is None:
        features_tuples_list_to_add = [('total sulfur dioxide', 'alcohol'),
                                       ('chlorides', 'volatile acidity'),
                                       ('density', 'citric acid')
                                       ]
    new_features_list = []
    for feature1, feature2 in features_tuples_list_to_add:
        new_feature = feature1 + "_X_" + feature2
        the_df[new_feature] = the_df[feature1] * the_df[feature2]
        new_features_list = new_features_list + [new_feature]
    return the_df, new_features_list


@print_function_name
def add_ratio_features(the_df, features_tuples_list_to_add=None):
    """
    Add ratio features to the DataFrame by dividing pairs of existing features.

    Parameters:
    - the_df (pd.DataFrame): DataFrame to be processed.
    - features_tuples_list_to_add (list): List of tuples, each containing a pair of features for ratio calculation. Defaults to None.

    Returns:
    - tuple: Modified DataFrame and a list of new ratio feature names.
    """
    if features_tuples_list_to_add is None:
        features_tuples_list_to_add = [('total acidity', 'free sulfur dioxide'),
                                       ('free sulfur dioxide', 'total sulfur dioxide'),
                                       ('total acidity', 'pH'),
                                       ('citric acid', 'fixed acidity'),
                                       ('alcohol', 'density'),
                                       ('residual sugar', 'chlorides'),
                                       ]
    new_features_list = []
    for feature1, feature2 in features_tuples_list_to_add:
        new_feature = feature1 + "_/_" + feature2
        the_df[new_feature] = the_df[feature1] / the_df[feature2]
        new_features_list = new_features_list + [new_feature]
    return the_df, new_features_list


@print_function_name
def engineer_new_features(the_df, add_functions_list=None, features_to_add_list=None):
    """
    Engineer new features by applying a series of transformation functions to the DataFrame.

    Parameters:
    - the_df (pd.DataFrame): DataFrame to be processed.
    - add_functions_list (list): List of functions to apply for feature engineering. Defaults to None, which means:
    add_polynomial_features, add_combination_features_wine_dataset, add_interaction_features, add_ratio_features.
    - features_to_add_list (list): List of features to be added by each function. Defaults to None.

    Returns:
    - tuple: Modified DataFrame and a list of all new engineered feature names.
    """
    if add_functions_list is None:
        add_functions_list = [add_polynomial_features,
                              add_combination_features_wine_dataset,
                              add_interaction_features,
                              add_ratio_features]
    if features_to_add_list is None:
        features_to_add_list = [None,
                                None,
                                None,
                                None]
    new_features_list = []
    for a_func, features_to_add in zip(add_functions_list, features_to_add_list):
        the_df, func_new_features_list = a_func(the_df, features_to_add)
        new_features_list = new_features_list + func_new_features_list
    return the_df, new_features_list


@print_function_name
def get_multicollinear_features(features_list, df_multicollinear_corr, add_print=False):
    """
    Identify multicollinear features from a given list based on a correlation matrix.

    Parameters:
    - features_list (list): List of feature names to check for multicollinearity.
    - df_multicollinear_corr (pd.DataFrame): DataFrame containing the correlation matrix.
    - add_print (bool): Flag to enable printing of results. Defaults to False.

    Returns:
    - list: List of multicollinear features.
    """
    multicollinear_feautres = {}
    for feature in features_list:
        feature_val_counts = df_multicollinear_corr[feature].value_counts()
        if len(feature_val_counts) > 0:
            features_high_corr = df_multicollinear_corr[feature].dropna()
            if add_print:
                print(f"\n{features_high_corr}")
            multicollinear_feautres[feature] = features_high_corr.index.tolist()
    engineered_correlated_feautures = pd.DataFrame.from_dict(multicollinear_feautres,
                                                             orient='index').reset_index().drop(
        columns='index').stack().drop_duplicates().values
    return engineered_correlated_feautures


@print_function_name
def drop_high_correlation_features(the_df, the_train_statistics, method='pearson', high_correlation_threshold=0.9,
                                   add_print=False):
    """
    Drop features from the DataFrame that have high correlation with other features.

    Parameters:
    - the_df (pd.DataFrame): DataFrame to be processed.
    - the_train_statistics (pd.DataFrame): Training statistics for reference.
    - method (str): Method for correlation calculation (e.g., 'pearson'). Defaults to 'pearson'.
    - high_corrleation_threshold (float): Threshold for considering a feature as highly correlated. Defaults to 0.9.
    - add_print (bool): Flag to enable printing of results. Defaults to False.

    Returns:
    - tuple: Modified DataFrame and a list of dropped feature names due to high correlation.
    """
    if add_print:
        print(f"\n# Each feature's high pearson correlations (at least {high_correlation_threshold}):")
    df_corr = the_df.corr(method=method)
    # Get all high correlations features' sets
    df_corr = df_corr[(df_corr.abs() >= high_correlation_threshold) & (df_corr.abs() != 1)].dropna(how='all')
    # get engineered highly correlated features
    orig_features = the_train_statistics[the_train_statistics['is_engineered'] == 0].index
    orig_features = [col for col in orig_features if col in df_corr]
    engineered_correlated_feautures = get_multicollinear_features(orig_features, df_corr, add_print=add_print)
    if add_print:
        print(
            f"\nThere are {len(engineered_correlated_feautures)} high correlated engineered feautres (>={high_correlation_threshold}):\n{engineered_correlated_feautures}")
    # drop engineered highly correlated features
    the_df = the_df.drop(columns=engineered_correlated_feautures)
    if add_print:
        print(
            f"After dropping highly correlated engineered features (>={high_correlation_threshold}, there are {len(the_df.columns)} features in dataset")
    all_correlated_dropped_features = engineered_correlated_feautures
    # get remaining highly correlated features
    remaining_features = the_train_statistics[the_train_statistics['is_engineered'] == 1].index
    remaining_features = [col for col in remaining_features if col in df_corr if col in the_df]
    remaining_correlated_feautures = get_multicollinear_features(remaining_features, df_corr, add_print=add_print)
    if add_print:
        print(
            f"There are {len(remaining_correlated_feautures)} high correlated remaining feautres (>={high_correlation_threshold}):\n{remaining_correlated_feautures}")
    if len(remaining_correlated_feautures) > 0:
        # drop remaining highly correlated features
        the_df = the_df.drop(columns=remaining_correlated_feautures)
        if add_print:
            print(
                f"After dropping highly correlated remaining features (>={high_correlation_threshold}, there are {len(the_df.columns)} features in dataset")
        all_correlated_dropped_features = all_correlated_dropped_features + remaining_correlated_feautures
    return the_df, all_correlated_dropped_features


@print_function_name
def standardize_df_using_train_statistics(the_df, the_train_statistics, add_print=False):
    """
    Standardize a DataFrame using mean and standard deviation from training statistics.

    Parameters:
    - the_df (pd.DataFrame): DataFrame to be standardized.
    - the_train_statistics (pd.DataFrame): Training statistics with mean and std values.
    - add_print (bool): If True, prints the standardized features' means and standard deviations. Defaults to False.

    Returns:
    - pd.DataFrame: Standardized DataFrame.
    """
    for feature in the_df:
        mu = the_train_statistics.loc[the_train_statistics.index == feature, 'mean'][0]
        sigma = the_train_statistics.loc[the_train_statistics.index == feature, 'std'][0]
        the_df[feature] = (the_df[feature] - mu) / sigma
    if add_print:
        print("\n # The standardized features means and standard deviations:\n")
        print(the_df.agg(['mean', 'std']).round(3))
    return the_df


@print_function_name
def upsample_target_minority(the_df, the_train_statistics, random_state=42):
    """
    Upsample the minority class in the target feature to balance the dataset.

    Parameters:
    - the_df (pd.DataFrame): DataFrame containing features and target.
    - the_train_statistics (pd.DataFrame): Training statistics for reference.
    - random_state (int): Seed for random number generator. Defaults to 42.

    Returns:
    - tuple: Feature DataFrame and target Series after upsampling.
    """
    # Extract target name from train_statistics
    target = the_train_statistics[the_train_statistics['is_target'] == 1].index[0]
    # get the number of rows of majority class and the classes values
    target_mean = the_train_statistics.loc[the_train_statistics.index == target, 'mean'][0]
    if target_mean >= 0.5:
        majority_pct = target_mean
        minority_class = 0
        majority_class = 1
    else:
        majority_pct = 1 - target_mean
        minority_class = 1
        majority_class = 0

    df_size = len(the_df)
    majority_N_rows = int(np.floor(majority_pct * df_size))
    # Resample the minorty class (with replacemnt) to the number of rows in majority class
    the_df_minorty = the_df[the_df[target] == minority_class]
    the_df_majority = the_df[the_df[target] == majority_class]
    the_df_minorty = the_df_minorty.sample(majority_N_rows, random_state=random_state, replace=True)
    # Concat the rows back together, and shuffle the df
    the_df = pd.concat([the_df_minorty, the_df_majority], axis=0)
    the_df = the_df.sample(frac=1)
    # split the features and the target to two dfs
    the_target_df = the_df[target]
    the_target_features = the_df.drop(columns=target)
    return the_target_features, the_target_df


@print_function_name
def add_target_to_train_statistics(the_train_target, the_train_statistics, target='quality'):
    """
    Add target feature statistics to the training statistics DataFrame.

    Parameters:
    - the_train_target (pd.Series): Series containing the target data.
    - the_train_statistics (pd.DataFrame): Training statistics DataFrame.
    - target (str): Name of the target feature. Defaults to 'quality'.

    Returns:
    - pd.DataFrame: Updated training statistics including target feature statistics.
    """
    # get target statistics
    target_statistics = the_train_target.describe(include='all').T
    # add target statistics to train statistics
    the_train_statistics = pd.concat([target_statistics, the_train_statistics], axis=0)
    # add is_target property
    the_train_statistics = add_binary_property_to_train_statistics(the_train_statistics, 'is_target', target)
    # move is_target property to first column
    the_train_statistics = pd.concat(
        [the_train_statistics['is_target'], the_train_statistics.drop(columns=['is_target'])], axis=1)
    return the_train_statistics


@print_function_name
def replace_columns_spaces_with_underscores(the_df):
    """
    Replace spaces in DataFrame column names with underscores.

    Parameters:
    - the_df (pd.DataFrame): DataFrame whose column names need modification.

    Returns:
    - pd.DataFrame: DataFrame with updated column names.
    """
    the_df.columns = the_df.columns.str.replace("_/_", "/")
    the_df.columns = the_df.columns.str.replace(" ", "_")
    return the_df


@print_function_name
def drop_features(the_df, features_to_drop, errors='ignore'):
    """
    Drop specified features from a DataFrame.

    Parameters:
    - the_df (pd.DataFrame): DataFrame from which features are to be dropped.
    - features_to_drop (list): List of feature names to drop.
    - errors (str): Error handling strategy, 'ignore' or 'raise'. Defaults to 'ignore'.

    Returns:
    - pd.DataFrame: DataFrame after dropping specified features.
    """
    the_df = the_df.drop(columns=features_to_drop, errors=errors)
    return the_df


@print_function_name
def drop_features_with_train_statistics_property(the_df, the_train_statistics, property_list, errors='ignore'):
    """
    Drop features from a DataFrame based on a specific property in the training statistics.

    Parameters:
    - the_df (pd.DataFrame): DataFrame to be processed.
    - the_train_statistics (pd.DataFrame): Training statistics containing property information.
    - property_list (list): List of properties used to determine which features to drop.
    - errors (str): Error handling strategy, 'ignore' or 'raise'. Defaults to 'ignore'.

    Returns:
    - pd.DataFrame: DataFrame after dropping features based on the specified property.
    """
    for type_of_features_to_drop in property_list:
        features_to_drop = get_train_feautres_with_property(the_train_statistics, type_of_features_to_drop)
        the_df = drop_features(the_df, features_to_drop=features_to_drop, errors=errors)
    return the_df


@print_function_name
def move_cols_to_first(the_df, cols):
    """
    Move specified columns to the beginning of a DataFrame.

    Parameters:
    - the_df (pd.DataFrame): The DataFrame to be modified.
    - cols (list): List of column names to move to the beginning.

    Returns:
    - pd.DataFrame: The DataFrame with specified columns moved to the beginning.
    """
    other_cols = the_df.columns[~the_df.columns.isin(cols)]
    the_df = pd.concat([the_df[cols], the_df[other_cols]], axis=1)
    return the_df


@print_function_name
def reorder_feature_importance_by_abs_values(the_feature_importance, importance_col='importance'):
    """
    Reorder feature importance by absolute values.

    Parameters:
    - the_feature_importance (pd.DataFrame): DataFrame containing features and their importance.
    - importance_col (str): Column name indicating the importance of features. Defaults to 'importance'.

    Returns:
    - pd.DataFrame: DataFrame with features reordered based on their importance in absolute terms.
    """
    feature_importance_index = the_feature_importance.copy(deep=True).abs().sort_values(by=importance_col,
                                                                                        ascending=False).index
    the_feature_importance = the_feature_importance.loc[feature_importance_index]
    return the_feature_importance


@print_function_name
def get_permutation_importance(the_model, the_X_train, the_X_val, the_y_val, random_state=0):
    """
    Calculate permutation importance for features in a trained model.

    Parameters:
    - the_model: Trained machine learning model.
    - the_X_train (pd.DataFrame): Training feature set.
    - the_X_val (pd.DataFrame): Validation feature set.
    - the_y_val (pd.Series): Validation target.
    - random_state (int): Seed for random number generator. Defaults to 0.

    Returns:
    - pd.DataFrame: DataFrame containing features and their permutation importance.
    """
    imps = permutation_importance(the_model, the_X_val.values, the_y_val.values, random_state=random_state)
    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(the_X_train)
    feature_names = count_vect.get_feature_names_out()
    the_feature_importance = zip(feature_names, imps.importances_mean)
    the_feature_importance = sorted(the_feature_importance, key=lambda x: x[1], reverse=True)
    the_feature_importance = pd.DataFrame(the_feature_importance)
    the_feature_importance.columns = ['feature', 'importance']
    the_feature_importance = the_feature_importance.set_index('feature')
    the_feature_importance = reorder_feature_importance_by_abs_values(the_feature_importance,
                                                                      importance_col='importance')
    return the_feature_importance


@print_function_name
# Fix plot_feature_importance, so it'll receive plot axis as input, for plotting several graphs one above the other
def plot_feature_importance(the_feature_importance, ax, show_sign_color=True, show_only_important=True, model_name=""):
    """
    Plot feature importance on the provided axis object.

    Parameters:
    - the_feature_importance (pd.DataFrame): DataFrame containing features and their importance.
    - ax (matplotlib.axes.Axes): The axis object to plot on.
    - show_sign_color (bool): If True, different colors will be used to indicate positive and negative importance. Defaults to True.
    - show_only_important (bool): If True, only features with non-zero importance are plotted. Defaults to True.
    - model_name (str): Name of the model for title. Defaults to an empty string.

    Returns:
    - None: The function plots the feature importance on the provided axis.
    """
    the_feature_importance = the_feature_importance.copy(deep=True)
    if show_sign_color:
        the_feature_importance['color'] = np.where(the_feature_importance['importance'] >= 0, '#88CCEE', '#CC79A7')
        the_feature_importance['importance'] = the_feature_importance['importance'].abs()
    the_feature_importance = the_feature_importance.sort_values(by='importance', ascending=True)
    if show_only_important:
        the_feature_importance = the_feature_importance[the_feature_importance['importance'] > 0]
    if show_sign_color:
        the_feature_importance['importance'].plot.barh(color=the_feature_importance['color'].values, ax=ax,
                                                       title=f'{model_name} Feature importances: blue=positive, red=negative')
    else:
        the_feature_importance['importance'].plot.barh(ax=ax, title=f'{model_name} Feature importances')
    plt.tight_layout()


@print_function_name
def plot_roc_curves(the_y_train, the_y_prob_train, the_y_val, the_y_prob_val, ax, model_name='basline'):
    """
    Plot ROC curves for training and validation sets.

    Parameters:
    - the_y_train (pd.Series): Actual target values for the training set.
    - the_y_prob_train (np.array): Predicted probabilities for the training set.
    - the_y_val (pd.Series): Actual target values for the validation set.
    - the_y_prob_val (np.array): Predicted probabilities for the validation set.
    - ax (matplotlib.axes.Axes): The axis object to plot on.
    - model_name (str): Name of the model for title. Defaults to 'baseline'.

    Returns:
    - None: The function plots ROC curves on the provided axis.
    """
    fpr_train, tpr_train, thresholds_train = roc_curve(the_y_train, the_y_prob_train)
    fpr_val, tpr_val, thresholds_val = roc_curve(the_y_val, the_y_prob_val)
    auc_train = roc_auc_score(the_y_train, the_y_prob_train)
    auc_val = roc_auc_score(the_y_val, the_y_prob_val)
    RocCurveDisplay(fpr=fpr_train, tpr=tpr_train, roc_auc=auc_train, estimator_name='Train').plot(ax=ax)
    RocCurveDisplay(fpr=fpr_val, tpr=tpr_val, roc_auc=auc_val, estimator_name='Validation').plot(ax=ax)
    ax.set_title(f"{model_name} ROC curve".capitalize())
    ax.plot([0, 1], [0, 1], 'r--', label='Random classifier')
    ax.legend(loc='lower right')
    plt.tight_layout()


@print_function_name
def get_model_metrics(the_y_val, the_y_pred, the_y_prob_val, the_feature_importance, model, model_params,
                      model_name='baseline', existing_model_metrics=None, export_to_csv=False, filename='model_metrics.csv'):
    """
    Generate and retrieve model performance metrics, including confusion matrix, classification report, lift and AUC.

    Parameters:
    - the_y_val (pd.Series): Actual target values from the validation set.
    - the_y_pred (np.array): Predicted target values from the validation set.
    - the_y_prob_val (np.array): Predicted probabilities for the validation set.
    - the_feature_importance (pd.DataFrame): DataFrame of feature importances.
    - model: The trained model.
    - model_params (dict): Parameters used in the model.
    - model_name (str): Name of the model for identification. Defaults to 'baseline'.
    - existing_model_metrics (pd.DataFrame): DataFrame of existing model metrics to append to. Defaults to None.
    - export_to_csv (bool): Whether to export the metrics to a CSV file. Defaults to False.

    Returns:
    - pd.DataFrame: DataFrame containing the updated model metrics.
    """
    # if not existing_model_metrics, import existing metrics from folder if it exists, else create a new dataframe.
    # if existing_model_metrics - use the existing model metrics, in the end we'll append current results to existing.
    if existing_model_metrics is None:
        if filename in os.listdir():
            the_model_metrics = pd.read_csv(filename, index_col=0)
        else:
            the_model_metrics = pd.DataFrame()
    else:
        the_model_metrics = existing_model_metrics.copy(deep=True)
    # create confusion matrix
    conf_matrix = pd.DataFrame(confusion_matrix(the_y_val, the_y_pred))
    conf_matrix = conf_matrix.stack().to_frame().T
    conf_matrix.columns = ['TN', 'FP', 'FN', 'TP']
    # create classification report
    class_report = pd.DataFrame(classification_report(the_y_val, the_y_pred, output_dict=True)).drop(
        columns=['macro avg', 'weighted avg'])
    class_report = pd.concat(
        [class_report[class_report.index == 'support'], class_report[class_report.index != 'support']], axis=0)
    class_report = class_report.stack().to_frame().T
    class_report = pd.concat([conf_matrix, class_report], axis=1)
    class_report[[('support', '1'), ('support', '0')]] = class_report[[('support', '1'), ('support', '0')]].astype(int)
    ## create distribution 1, lift 1, lift 0
    class_report[('distribution', '1')] = class_report[('support', '1')] / (
            class_report[('support', '0')] + class_report[('support', '1')])
    class_report[('lift', '1')] = class_report[('precision', '1')] / class_report[('distribution', '1')]
    class_report[('lift', '0')] = class_report[('precision', '0')] / (1 - class_report[('distribution', '1')])
    class_report = class_report.rename(columns={('support', 'accuracy'): 'accuracy'}).drop(
        columns=[('f1-score', 'accuracy'), ('recall', 'accuracy'), ('precision', 'accuracy')])
    ## add AUC
    class_report['AUC'] = roc_auc_score(the_y_val, the_y_prob_val)
    ## reorder columns
    class_report = move_cols_to_first(class_report,
                                      [('support', '1'), ('support', '0'), 'TP', 'FP', 'TN', 'FN', 'accuracy',
                                       ('distribution', '1'), ('precision', '1'), ('lift', '1'), ('recall', '1'),
                                       ('f1-score', '1'), 'AUC', ('precision', '0'), ('lift', '0'), ('recall', '0')])
    # add feature importance
    the_feature_importance = the_feature_importance if isinstance(model_name, dict) else \
        the_feature_importance.round(3).to_dict()['importance']
    class_report['feature_importance'] = [the_feature_importance]
    # add modeling metdata
    class_report['model'] = [model]
    class_report['hyper_parameters'] = [model_params]
    class_report['train_timestamp'] = pd.to_datetime(datetime.now().strftime("%d-%m-%Y %H:%M:%S"), dayfirst=True)
    class_report = move_cols_to_first(class_report, ['train_timestamp', 'model', 'hyper_parameters'])
    # set model name
    model_name = model_name if isinstance(model_name, list) else [model_name]
    class_report.index = model_name
    # append current results to either new or existing dataframe
    if len(the_model_metrics) > 0:
        class_report.columns = the_model_metrics.columns
    the_model_metrics = pd.concat([the_model_metrics, class_report], axis=0)
    if export_to_csv:
        the_model_metrics.to_csv(filename)
    return the_model_metrics


@print_function_name
def train_evaluate_plot_report_sklearn_classification_model(the_model, the_X_train, the_y_train, the_X_val, the_y_val,
                                                            the_model_name=None, export_metrics_to_csv=True,
                                                            to_plot=True, plot_time='first', axs=None,
                                                            filename='model_metrics.csv'):
    """
    Train, evaluate, plot, and report metrics for a Scikit-learn classification model.

    Parameters:
    - the_model: Scikit-learn model to be trained.
    - the_X_train (pd.DataFrame): Training data features.
    - the_y_train (pd.Series): Training data target.
    - the_X_val (pd.DataFrame): Validation data features.
    - the_y_val (pd.Series): Validation data target.
    - the_model_name (str): Name of the model. Used for plotting and reporting. Defaults to None.
    - export_metrics_to_csv (bool): If True, export metrics to a CSV file. Defaults to True.
    - to_plot (bool): If True, plot feature importance and ROC curves. Defaults to True.
    - plot_time (str): When to plot ('first', 'second', or 'unique'). Defaults to 'first'.
    - axs: Matplotlib axes array for plotting. Defaults to None.

    Returns:
    - The trained model metrics as a DataFrame, along with axes if plotting is enabled.
    """
    the_model = the_model.fit(the_X_train.values, the_y_train.values)
    y_pred = the_model.predict(the_X_val.values)

    if hasattr(the_model, 'feature_importances_'):  # Tree based algorithms
        feature_importance = the_model.feature_importances_
        feature_importance = pd.DataFrame({'feature': the_X_train.columns, 'importance': feature_importance})
        feature_importance = feature_importance.set_index('feature')
    elif hasattr(the_model, 'coef_'):  # logistic regression coefficients
        coefficients = the_model.coef_[0]
        feature_importance = pd.DataFrame({'feature': the_X_train.columns, 'importance': coefficients})
        feature_importance = feature_importance.set_index('feature')
    else:  # permutation importance
        feature_importance = get_permutation_importance(the_model, the_X_train, the_X_val, the_y_val)

    if the_model_name is None:
        the_model_name = str(the_model)
    the_y_prob_train = the_model.predict_proba(the_X_train.values)[:, [1]]
    the_y_prob_val = the_model.predict_proba(the_X_val.values)[:, [1]]

    if to_plot:
        if plot_time == 'unique':
            fig, axs = plt.subplots(2, 1, figsize=(8, 10))  # Adjust the size as necessary
            first_axes = 0
            second_axes = 1
        elif plot_time == 'first':
            fig, axs = plt.subplots(4, 1, figsize=(8, 20))  # Adjust the size as necessary
            first_axes = 0
            second_axes = 1
        elif plot_time == 'second':
            first_axes = 2
            second_axes = 3
        plot_feature_importance(feature_importance, axs[first_axes], show_sign_color=True, show_only_important=True,
                                model_name=the_model_name)
        plot_roc_curves(the_y_train, the_y_prob_train, the_y_val, the_y_prob_val, axs[second_axes],
                        model_name=the_model_name)

    model_params = the_model.get_params()
    the_model_metrics = get_model_metrics(the_y_val, y_pred, the_y_prob_val,
                                          feature_importance, model=the_model,
                                          model_params=model_params,
                                          model_name=the_model_name,
                                          export_to_csv=export_metrics_to_csv,
                                          filename=filename)
    return the_model_metrics, axs


@print_function_name
def get_important_features(the_feature_importance, importance_threshold=0.001):
    """
    Extract a list of important features based on a specified importance threshold.

    Parameters:
    - the_feature_importance (pd.DataFrame): DataFrame containing feature importances.
    - importance_threshold (float): Threshold value to consider a feature as important. Defaults to 0.001.

    Returns:
    - list: List of important feature names.
    """
    the_important_features = the_feature_importance[the_feature_importance > importance_threshold].dropna()
    the_important_features = the_important_features.index.tolist()
    return the_important_features


@print_function_name
def retrain_on_entire_dataset(the_df_X, the_df_y, the_model, the_model_name='entire dataset (no validation)'):
    """
    Retrain a model on the entire dataset (without a validation set) and plot feature importances.

    Parameters:
    - the_df_X (pd.DataFrame): Feature set for the entire dataset.
    - the_df_y (pd.Series): Target set for the entire dataset.
    - the_model: Machine learning model to be retrained.
    - the_model_name (str): Name of the model for plotting purposes. Defaults to 'entire dataset (no validation)'.

    Returns:
    - None: The function retrains the model and plots feature importances.
    """
    the_model = the_model.fit(the_df_X.values, the_df_y.values)

    if hasattr(the_model, 'feature_importances_'):  # Tree based algorithms
        feature_importance = the_model.feature_importances_
        feature_importance = pd.DataFrame({'feature': the_df_X.columns, 'importance': feature_importance})
        feature_importance = feature_importance.set_index('feature')
    elif hasattr(the_model, 'coef_'):  # logistic regression coefficients
        coefficients = the_model.coef_[0]
        feature_importance = pd.DataFrame({'feature': the_df_X.columns, 'importance': coefficients})
        feature_importance = feature_importance.set_index('feature')
    else:  # permutation importance
        feature_importance = get_permutation_importance(the_model, the_df_X, the_df_X, the_df_y)

    fig, axs = plt.subplots(figsize=(8, 6))  # Adjust the size as necessary
    plot_feature_importance(feature_importance, axs, show_sign_color=True, show_only_important=True,
                            model_name=the_model_name)

    return the_model


@print_function_name
def transform_target_to_series(the_df, target_col='quality'):
    """
    Transform and extract the target column from a DataFrame into a Series.

    Parameters:
    - the_df (pd.DataFrame): DataFrame containing the target column.
    - target_col (str): Name of the target column. Defaults to 'quality'.

    Returns:
    - pd.Series: Series containing the target data.
    """
    if type(the_df) == pd.DataFrame:
        the_df = the_df[target_col]
    return the_df


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


def get_AUC_score(the_X_val, the_y_val, the_model):
    """
    Computes the AUC (Area Under the Curve) score for a given model using validation data.

    Parameters:
    - the_X_val (pd.DataFrame): DataFrame containing validation features.
    - the_y_val (pd.Series): Series containing validation target values.
    - the_model: The trained model object.

    Returns:
    - float: The AUC score of the model on the validation data.
    """
    the_score = roc_auc_score(the_y_val, the_model.predict_proba(the_X_val)[:, 1])
    return the_score


@print_function_name
def find_best_model_greedy_feed_forward(the_X_train, the_y_train, the_X_val, the_y_val, the_model, the_metric_fn,
                                        the_metric_name, print_each_iteration=False, plot_iterations=True,
                                        n_features=None):
    """
    Applies greedy feed-forward feature selection to find the best performing model based on a specified metric.

    Parameters:
    - the_X_train (pd.DataFrame): DataFrame containing training features.
    - the_y_train (pd.Series): Series containing training target values.
    - the_X_val (pd.DataFrame): DataFrame containing validation features.
    - the_y_val (pd.Series): Series containing validation target values.
    - the_model: The model object to be trained and evaluated.
    - the_metric_fn: Function to evaluate the model.
    - the_metric_name (str): Name of the metric used for evaluation.
    - print_each_iteration (bool, optional): If True, prints score for each iteration. Defaults to False.
    - plot_iterations (bool, optional): If True, plots the metric scores across iterations. Defaults to True.
    - n_features (int, optional): Number of top features to consider. Defaults to considering all features.

    Returns:
    - Tuple: The best model after greedy feature selection and the set of selected features.
    """
    if n_features:
        the_X_train = the_X_train.iloc[:, :n_features]
        the_X_val = the_X_val.iloc[:, :n_features]
    selected_features = []  # List to store selected features
    n_trained_models = 0
    iteration_scores = []
    best_model_per_num_features = {}
    for num_features in range(1, len(the_X_train.columns) + 1):
        print(f"Training models with {num_features} features...")
        best_feature = None  # To store the best feature for the current iteration
        best_iteration_score = 0  # To store the best score for the current iteration
        for feature in the_X_train.columns:
            if feature not in selected_features:  # Avoid features already selected
                feature_combo = selected_features + [feature]
                X_train_selected = the_X_train[feature_combo]
                X_val_selected = the_X_val[feature_combo]

                # Train your model using X_selected and y_train
                the_model.fit(X_train_selected, the_y_train)
                n_trained_models += 1
                # Evaluate your model (you can use any evaluation metric)
                iteration_score = the_metric_fn(the_X_val=X_val_selected, the_y_val=the_y_val, the_model=the_model)
                if print_each_iteration:
                    print("iteration_score", iteration_score)
                iteration_scores += [iteration_score]

                # Check if the current iteration score is better
                if iteration_score > best_iteration_score:
                    best_iteration_score = iteration_score
                    best_feature = feature
                    best_feature_combo = feature_combo

        # If a better feature is found for this iteration, add it to selected features
        if best_feature is not None:
            selected_features.append(best_feature)

        # Update best model per num_features
        best_model_per_num_features.update({f'{num_features}': {'best_achieved_score': best_iteration_score,
                                                                'best_set_of_features': best_feature_combo}})
        print(f"best_feature_combo:", best_feature_combo)

    print(f"{n_trained_models} models were trained")

    if plot_iterations:
        best_model_per_num_features = pd.DataFrame(best_model_per_num_features).T
        best_model_per_num_features['best_achieved_score'] = best_model_per_num_features['best_achieved_score'].astype(
            float)
        best_n_features = best_model_per_num_features['best_achieved_score'].idxmax()
        best_overall_score = \
            best_model_per_num_features[best_model_per_num_features.index == best_n_features][
                'best_achieved_score'].values[
                0]
        the_best_set_of_features = best_model_per_num_features[best_model_per_num_features.index == best_n_features][
            'best_set_of_features'].values[0]
        best_model_per_num_features.best_achieved_score.plot(
            title=f'Best achieved forward selection {the_metric_name} for {str(the_model)} is {np.round(best_overall_score, 3)} with {best_n_features} features')
        plt.tight_layout()

    # recreate best model
    X_train_selected = the_X_train[the_best_set_of_features]
    X_val_selected = the_X_val[the_best_set_of_features]
    the_best_model = the_model.fit(X_train_selected, the_y_train)
    iteration_score = the_metric_fn(the_X_val=X_val_selected, the_y_val=the_y_val, the_model=the_best_model)
    assert iteration_score == best_overall_score
    return the_best_model, the_best_set_of_features


@print_function_name
def find_best_model_greedy_feed_backward(the_X_train, the_y_train, the_X_val, the_y_val, the_model, the_metric_fn,
                                         the_metric_name, print_each_iteration=False, plot_iterations=True,
                                         n_features=None):
    """
    Applies greedy feed-backward feature elimination to find the best performing model based on a specified metric.

    Starting with all features, this function iteratively removes the least contributing feature (based on the evaluation metric) until the desired number of features is reached or no further improvement is observed.

    Parameters:
    - the_X_train (pd.DataFrame): DataFrame containing training features.
    - the_y_train (pd.Series): Series containing training target values.
    - the_X_val (pd.DataFrame): DataFrame containing validation features.
    - the_y_val (pd.Series): Series containing validation target values.
    - the_model: The model object to be trained and evaluated.
    - the_metric_fn: Function to evaluate the model.
    - the_metric_name (str): Name of the metric used for evaluation.
    - print_each_iteration (bool, optional): If True, prints score for each iteration. Defaults to False.
    - plot_iterations (bool, optional): If True, plots the metric scores across iterations. Defaults to True.
    - n_features (int, optional): Maximum number of top features to consider. Defaults to considering all features.

    Returns:
    - Tuple: The best model after greedy feature elimination and the set of remaining features.

    The function trains multiple models by progressively removing the least significant features based on the provided evaluation metric. The process continues until the specified number of features is reached or removing more features does not improve the model performance.
    """
    if n_features:
        n_features += 1
        the_X_train = the_X_train.iloc[:, :n_features]
        the_X_val = the_X_val.iloc[:, :n_features]
    selected_features = list(the_X_train.columns)  # Start with all features
    n_trained_models = 0
    iteration_scores = []
    best_model_per_num_features = {}
    for num_features in range(len(the_X_train.columns) + 1, 2, -1):
        print(f"Training models with {num_features - 2} features...")
        worst_feature = None  # To store the worst feature for the current iteration
        worst_iteration_score = 1  # To store the worst score for the current iteration
        for feature in selected_features:
            feature_combo = [feat for feat in selected_features if feat != feature]
            X_train_selected = the_X_train[feature_combo]
            X_val_selected = the_X_val[feature_combo]

            # Train your model using X_selected and y_train
            the_model.fit(X_train_selected, the_y_train)
            n_trained_models += 1
            # Evaluate your model (you can use any evaluation metric)
            iteration_score = the_metric_fn(the_X_val=X_val_selected, the_y_val=the_y_val, the_model=the_model)
            if print_each_iteration:
                print("iteration_score", iteration_score)
            iteration_scores += [iteration_score]

            # Check if the current iteration score is worse
            if iteration_score < worst_iteration_score:
                worst_iteration_score = iteration_score
                worst_feature = feature
                worst_feature_combo = feature_combo

        # If a worse feature is found for this iteration, remove it from selected features
        if worst_feature is not None:
            selected_features.remove(worst_feature)

        # Update best model per num_features
        best_model_per_num_features.update({f'{num_features - 2}': {'best_achieved_score': worst_iteration_score,
                                                                    'best_set_of_features': worst_feature_combo}})
        print(f"best_set_of_features:", worst_feature_combo)

    print(f"{n_trained_models} models were trained")

    if plot_iterations:
        best_model_per_num_features = pd.DataFrame(best_model_per_num_features).T
        best_model_per_num_features['best_achieved_score'] = best_model_per_num_features['best_achieved_score'].astype(
            float)
        best_n_features = best_model_per_num_features['best_achieved_score'].idxmax()
        best_overall_score = \
            best_model_per_num_features[best_model_per_num_features.index == best_n_features][
                'best_achieved_score'].values[
                0]
        the_best_set_of_features = best_model_per_num_features[best_model_per_num_features.index == best_n_features][
            'best_set_of_features'].values[0]
        best_model_per_num_features.best_achieved_score.plot(
            title=f'Best achieved backward selection {the_metric_name} for {str(the_model)} is {np.round(best_overall_score, 3)} with {best_n_features} features')
        plt.tight_layout()

    # recreate best model
    X_train_selected = the_X_train[the_best_set_of_features]
    X_val_selected = the_X_val[the_best_set_of_features]
    the_best_model = the_model.fit(X_train_selected, the_y_train)
    iteration_score = the_metric_fn(the_X_val=X_val_selected, the_y_val=the_y_val, the_model=the_best_model)
    assert iteration_score == best_overall_score
    return the_best_model, the_best_set_of_features


@print_function_name
def optimize_model_complexity_early_stopping(the_model, the_X_train,
                                             the_y_train, the_X_val, the_y_val, the_hyper_parameter,
                                             the_values_range, metric_fn=get_AUC_score, metric_name='AUC',
                                             diff_periods=1, train_optimization_value=0.02):
    """
    Optimizes a model's hyperparameter by monitoring early stopping criteria on training and validation datasets.

    This function iterates over a range of hyperparameter values to find the optimal setting where the performance
    improvement either becomes negligible or starts decreasing. It uses the concept of early stopping to prevent overfitting.

    Parameters:
    - the_model: The model object to be optimized.
    - the_X_train (pd.DataFrame): DataFrame containing training features.
    - the_y_train (pd.Series): Series containing training target values.
    - the_X_val (pd.DataFrame): DataFrame containing validation features.
    - the_y_val (pd.Series): Series containing validation target values.
    - the_hyper_parameter (str): The name of the hyperparameter to be optimized.
    - the_values_range (iterable): A range of values to test for the hyperparameter.
    - metric_fn (function, optional): Function to evaluate the model. Defaults to get_AUC_score.
    - metric_name (str, optional): Name of the evaluation metric. Defaults to 'AUC'.
    - diff_periods (int, optional): The number of periods for calculating the difference in metric scores. Defaults to 1.
    - train_optimization_value (float, optional): The threshold for minimal acceptable improvement in training score. Defaults to 0.02.

    Returns:
    - pd.DataFrame: DataFrame containing metric scores for training and validation sets across the tested hyperparameter values.
    - The optimal hyperparameter value based on validation score.
    - The optimal hyperparameter value based on training score improvement threshold.

    The function trains the model for each hyperparameter value and evaluates it on both training and validation datasets.
    It plots the metric scores to visually determine the point of diminishing returns or performance degradation, aiding in the selection of the optimal hyperparameter value.
    """
    model_params = the_model.get_params()
    the_values_scores_on_train = {}
    the_values_scores_on_val = {}
    for value in the_values_range:
        print(f'Training {the_hyper_parameter}={value} . . .')
        model_params[the_hyper_parameter] = value
        the_model.set_params(**model_params)
        the_model = the_model.fit(the_X_train.values, the_y_train.values)
        the_values_scores_on_train[value] = metric_fn(the_X_train.values, the_y_train, the_model)
        the_values_scores_on_val[value] = metric_fn(the_X_val.values, the_y_val, the_model)
    res = pd.DataFrame(the_values_scores_on_train.values(), the_values_scores_on_train.keys()).rename(
        columns={0: 'Train'})
    res['Validation'] = the_values_scores_on_val.values()
    res['validation_diff'] = res.Validation.diff(periods=diff_periods)
    res['train_diff'] = res.Train.diff(periods=diff_periods)
    if len(res[res['validation_diff'] <= 0]):  # get value before the decrease in validation
        first_non_negative_diff_value = res[res.index < res[res['validation_diff'] <= 0].index[0]].index[-1]
    else:  # no decrease in validation - get last value
        first_non_negative_diff_value = res.index[-1]
    if len(res[res['train_diff'] <= train_optimization_value]):  # get value before the decrease in validation
        train_first_non_negative_diff_value = \
            res[res.index < res[res['train_diff'] <= train_optimization_value].index[0]].index[-1]
    else:  # no decrease in validation - get last value
        train_first_non_negative_diff_value = res.index[-1]
    title = f'''{str(the_model)} **{metric_name}**:
    First {diff_periods} preiods non-increasing validation value: {the_hyper_parameter}={first_non_negative_diff_value},
    and first train increase value of less than {train_optimization_value} is: {the_hyper_parameter}={train_first_non_negative_diff_value}'''
    res.drop(columns=['validation_diff', 'train_diff']).plot(title=title, figsize=(8, 6))
    plt.xlabel(the_hyper_parameter)
    plt.ylabel(metric_name)
    plt.legend()
    plt.xticks(res.index.tolist())
    plt.axvline(first_non_negative_diff_value, linestyle='--', color='orange')
    plt.axvline(train_first_non_negative_diff_value, linestyle='--')
    plt.tight_layout()
    plt.show()
    return res, first_non_negative_diff_value, train_first_non_negative_diff_value

@print_function_name
def move_threshold(the_model, the_X_train, the_y_train, the_X_val, the_y_val, metric_fn=f1_score,
                   metric_name='f1-score', to_plot=True):
    """
    Determines the optimal decision threshold for a classification model to maximize a specified metric.

    This function iterates over a range of potential decision thresholds for classifying instances.
    It calculates the desired metric (such as F1-score) for each threshold to identify the one that
    yields the best performance on the training dataset. It also evaluates the model performance on
    the validation dataset using the identified threshold.

    Parameters:
    - the_model: The trained model object.
    - the_X_train (pd.DataFrame): DataFrame containing training features.
    - the_y_train (pd.Series): Series containing training target values.
    - the_X_val (pd.DataFrame): DataFrame containing validation features.
    - the_y_val (pd.Series): Series containing validation target values.
    - metric_fn (function, optional): The metric function used for optimization (e.g., f1_score). Defaults to f1_score.
    - metric_name (str, optional): The name of the metric being optimized. Defaults to 'f1-score'.
    - to_plot (bool, optional): Flag to plot the metric scores against different thresholds. Defaults to True.

    Returns:
    - The optimal decision threshold value.
    - The best metric score achieved on the training set.

    The function first predicts the probability scores for the training set, then iterates over a range
    of thresholds to find the optimal one based on the specified metric. It also evaluates the metric
    score on the validation set using the optimal threshold and optionally plots the metric scores across
    all tested thresholds.
    """
    thresholds = np.linspace(0, 1, 100)
    calculated_metrics = {}
    # Calc metric on each threshold value
    for threshold in thresholds:
        proba = the_model.predict_proba(the_X_train.values)[:, 1]
        the_y_pred = (proba >= threshold) * 1
        calculated_metrics[threshold] = metric_fn(the_y_train, the_y_pred)
    calculated_metrics = pd.DataFrame.from_dict(calculated_metrics, orient='index').rename(columns={0: metric_name})
    # Find best metric value and threshold
    the_best_threshold = calculated_metrics.idxmax().values[0]
    the_best_score = calculated_metrics.max().values[0]
    # Get validation score
    proba = the_model.predict_proba(the_X_val.values)[:, 1]
    y_pred = (proba >= the_best_threshold) * 1
    validation_score = metric_fn(the_y_val, y_pred)
    if to_plot:
        # Plot results
        calculated_metrics.plot(
            title=f'''Train best {metric_name} is {the_best_score.round(2)}, for decision threshold={the_best_threshold.round(2)},
For Validation, {metric_name}={validation_score.round(2)}''')
        plt.xlabel("Decision Thresholds")
        plt.ylabel(metric_name)
        plt.show()
    return the_best_threshold, the_best_score

@print_function_name
def plot_tsne(the_model, the_X_val, title='Predictions Visualized with t-SNE'):
    """
    Visualizes the predictions of a model using t-Distributed Stochastic Neighbor Embedding (t-SNE).

    This function applies t-SNE to reduce the dimensionality of the validation set to two dimensions
    and plots these points, colored by the model's predictions. This visualization can help in
    understanding how the model is grouping or separating different data points.

    Parameters:
    - the_model: The trained model used for making predictions.
    - the_X_val: The validation dataset (features) on which predictions are made.
    - title (str, optional): The title for the plot. Defaults to 'Predictions Visualized with t-SNE'.

    Returns:
    - None: The function creates a plot but does not return any value.
    """
    # Convert X_val and y_val to numpy arrays
    X_val_array = np.array(the_X_val)

    # Apply t-SNE to reduce the dimensionality of X_val to 2
    tsne = TSNE(n_components=2, random_state=0)
    X_val_tsne = tsne.fit_transform(X_val_array)

    # Use the original k-NN model to predict the classes for X_val
    y_pred = the_model.predict(X_val_array)

    # Plot the t-SNE-transformed data points, coloring them according to the predicted classes
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_val_tsne[:, 0], X_val_tsne[:, 1], c=y_pred, edgecolors='k', marker='o', s=50)
    plt.legend(*scatter.legend_elements(), title="Predicted Classes")
    plt.title(f"{str(the_model)} {title}")
    plt.xlabel("t-SNE component 1")
    plt.ylabel("t-SNE component 2")
    plt.show()

@print_function_name
def train_eval_plot_classification_algos_two_datasets(the_X_train, the_y_train, the_X_val, the_y_val, the_train_statistics,
                                                      the_algos=None,
                                                      features_to_drop_from_train_statistics_in_first='is_engineered',
                                                      model_name_in_second="_engineered_features",
                                                      export_metrics_to_csv=True, to_plot=True,
                                                      filename='model_metrics.csv'):
    """
    Trains, evaluates, and plots a set of classification algorithms on two datasets: one with and one without engineered features,
    or possible without 'features_to_drop_from_train_statistics_in_first'.

    This function trains a set of classification models on two versions of a dataset: one that includes engineered features
    and one that does not. It evaluates the models, plots performance metrics, and optionally exports these metrics to a CSV file.
    It also visualizes the dataset using t-SNE to provide insights into how the models are performing.

    Parameters:
    - the_X_train, the_y_train: The training dataset and labels.
    - the_X_val, the_y_val: The validation dataset and labels.
    - the_train_statistics: Statistics of the training dataset, used to identify features to drop.
    - the_algos (list, optional): List of classification algorithms to train. Defaults to common classifiers if None.
    - features_to_drop_from_train_statistics_in_first (str): Feature property to identify for dropping in the first dataset. Defaults to 'is_engineered'.
    - model_name_in_second (str): Suffix for model names in the second training round. Defaults to "_engineered_features".
    - export_metrics_to_csv (bool): Whether to export the model metrics to a CSV file. Defaults to True.
    - to_plot (bool): Whether to plot the evaluation metrics. Defaults to True.
    - filename (str): Name of the CSV file to export metrics to. Defaults to 'model_metrics.csv'.

    Returns:
    - None: The function trains the models and performs plots but does not return any value.
    """
    if the_algos is None:
        the_algos = [KNeighborsClassifier(), SVC(probability=True), LogisticRegression(), DecisionTreeClassifier(),
                     RandomForestClassifier(), GradientBoostingClassifier(), MLPClassifier()]
        print(f"# going to train on: {the_algos}")
    # Create dataset without engineered features
    if not features_to_drop_from_train_statistics_in_first is None:
        if not isinstance(features_to_drop_from_train_statistics_in_first, List):
            types_of_features_to_drop = [features_to_drop_from_train_statistics_in_first]
        else:
            types_of_features_to_drop = features_to_drop_from_train_statistics_in_first
        dropping_errors = 'ignore'
        # drop engineered features:
        drop_features_with_train_statistics_property_fn = partial(
            drop_features_with_train_statistics_property, the_train_statistics=the_train_statistics,
            property_list=types_of_features_to_drop, errors=dropping_errors)
        X_train_orig = drop_features_with_train_statistics_property_fn(the_X_train)
        X_val_orig = drop_features_with_train_statistics_property_fn(the_X_val)

    if to_plot:
        # prepare tsne plot function
        plot_tsne_fn = partial(plot_tsne, the_X_val=the_X_val)

    # create a partial function for all models that will be trained on the dataset WITHOUT engineered features
    if not features_to_drop_from_train_statistics_in_first is None:
        plot_time = 'first'
        train_evaluate_plot_report_sklearn_classification_model_original_features = partial(
            train_evaluate_plot_report_sklearn_classification_model, the_X_train=X_train_orig, the_y_train=the_y_train,
            the_X_val=X_val_orig, the_y_val=the_y_val, export_metrics_to_csv=export_metrics_to_csv, to_plot=to_plot,
            plot_time=plot_time, filename=filename)
        # create a partial function for all models that will be trained on the dataset WITH engineered features
        plot_time = 'second'
    else:
        plot_time = 'unique'

    train_evaluate_plot_report_sklearn_classification_model_engineered_features = partial(
        train_evaluate_plot_report_sklearn_classification_model, the_X_train=the_X_train, the_y_train=the_y_train,
        the_X_val=the_X_val,
        the_y_val=the_y_val, export_metrics_to_csv=export_metrics_to_csv, plot_time=plot_time, filename=filename)

    # train models with all algos
    for algo in the_algos:
        the_model = algo
        model_name_engineered = str(the_model) + model_name_in_second
        if not features_to_drop_from_train_statistics_in_first is None:
            model_metrics, axs = train_evaluate_plot_report_sklearn_classification_model_original_features(the_model=the_model)
            _, _ = train_evaluate_plot_report_sklearn_classification_model_engineered_features(
                the_model=the_model, the_model_name=model_name_engineered, axs=axs)
        else:
            _, _ = train_evaluate_plot_report_sklearn_classification_model_engineered_features(
                the_model=the_model, the_model_name=model_name_engineered)
        if to_plot:
            plot_tsne_fn(the_model)

@print_function_name
def main():
    KEEP_WORKING_DIR = True
    if KEEP_WORKING_DIR:
        orig_dir = os.getcwd()

    METRICS_FILENAME = 'model_metrics.csv'

    # Load the data
    try:
        df = import_data()
        # Proceed with training or other operations using df
    except FileNotFoundError as e:
        print(e)
        # Handle the error appropriately (e.g., exit the script or log the error)
        exit()
    add_print = False
    ## Define target
    target = 'quality'

    df = transform_numeric_target_feature_to_binary(df, target_col=target, threshold=7)
    X_train, X_test, X_val, y_train, y_test, y_val = split_dataset(df, target_col=target)

    # Import the warnings module
    import warnings

    # Ignore the UserWarning from scipy.stats
    # the warnings are saying we have less the 5000 samples so the noramlity test isn't accurate
    warnings.filterwarnings("ignore", category=UserWarning, module="scipy.stats")

    # Show all columns (don't replace some with "...")
    pd.set_option('display.max_columns', None)

    ## Test if train statistics are different than val and test statistics
    alpha = 0.01  # significance level
    dfs_dict_to_test = {'X_val': X_val, 'X_test': X_test}
    train_val_outlier_means_test = test_if_features_statistically_different(X_train, dfs_dict_to_test, alpha=alpha)
    print('\n# Test if train, validation and test sets means are statisically not different:\n',
          train_val_outlier_means_test)

    ## Missing Values imputation

    # Create a variable to hold all train statistics,
    # and insert the features with missing data, and transpose the matrix
    train_statistics = X_train.describe(include='all').T
    # Add a column to flag features with missing values
    len_X_train = len(X_train)
    train_statistics['has_na'] = (train_statistics['count'] < len_X_train) * 1

    # for train, for valiation and for test
    # Impute missing values
    X_train = imputate_missing_values('X_train', X_train, train_statistics, add_print=add_print)
    X_val = imputate_missing_values('X_val', X_val, train_statistics, add_print=add_print)
    X_test = imputate_missing_values('X_test', X_test, train_statistics, add_print=add_print)

    ## Handle Categoricals

    # Define parameters for the functions: cateorical features names, and if should one categoy be dropped for every feature
    categorical_features = ['type']
    drop_one = True
    # Encode categoricals for train, get back the train categories
    X_train, train_categories = one_hot_encode_categoricals(X_train, categorical_features=categorical_features,
                                                            drop_one=drop_one)

    # Add new train cateogories statistics to train_statistics

    # Add proprty 'is_categorical_to_drop' to original cateogorical features
    train_statistics = add_binary_property_to_train_statistics(train_statistics, 'is_categorical_to_drop',
                                                               categorical_features)
    # Add proprty 'is_category' to newly created categories one-hot-encoding features
    train_statistics = add_new_features_statistics_to_train_statistics(X_train, train_statistics, train_categories)
    train_statistics = add_binary_property_to_train_statistics(train_statistics, 'is_category', train_categories)

    # Get category features and categories of train, from train_statistics
    categorical_features = get_train_feautres_with_property(train_statistics, 'is_categorical_to_drop')
    categories_to_use_from_train = get_train_feautres_with_property(train_statistics, 'is_category')
    # Create a partial function, pre-set with these parameters, to be run on both X_val and X_test

    one_hot_encode_categoricals_fn = partial(one_hot_encode_categoricals, categorical_features=categorical_features,
                                             categories_to_use_from_train=categories_to_use_from_train)
    X_val, _ = one_hot_encode_categoricals_fn(X_val)
    X_test, _ = one_hot_encode_categoricals_fn(X_test)

    ## add kurtosis and skew statistics
    train_statistics = add_kurtosis_skew_statistics(X_train, train_statistics)

    ## Handle Outliers

    # Detect Outliers
    # Add outlier column indicator, having 1 for outlier rows
    X_train_numeric_features = None  # When none, assume train dataset and find all relevent columns
    outlier_col_suffix = 'is_outlier'
    X_train, train_outiler_cols = add_outlier_indicators_on_features(X_train, train_statistics,
                                                                     X_train_numeric_features=X_train_numeric_features,
                                                                     outlier_col_suffix=outlier_col_suffix)

    # Update outlier statistics to train_statistics
    train_statistics = add_new_features_statistics_to_train_statistics(X_train, train_statistics, train_outiler_cols)

    # Apply outlier indicators on validation and test

    # get train outlier columns
    train_outiler_cols = get_train_features_with_suffix(train_statistics, the_suffix=outlier_col_suffix)
    # Add outlier indicators to val and test in those specific features
    add_outlier_indicators_on_features_fn = partial(add_outlier_indicators_on_features,
                                                    the_train_statistics=train_statistics,
                                                    X_train_numeric_features=train_outiler_cols,
                                                    outlier_col_suffix=outlier_col_suffix)
    X_val, _ = add_outlier_indicators_on_features_fn(X_val)
    X_test, _ = add_outlier_indicators_on_features_fn(X_test)

    # Validate outliers detection: Test if train outlier statistics are different from val outlier statistics
    remove_suffix = False
    train_outlier_cols = get_train_features_with_suffix(train_statistics, the_suffix=outlier_col_suffix,
                                                        remove_suffix=remove_suffix)
    remove_suffix = True
    train_orig_outlier_cols = get_train_features_with_suffix(train_statistics, the_suffix=outlier_col_suffix,
                                                             remove_suffix=remove_suffix)
    X_train_outliers = X_train.loc[(X_train[train_outlier_cols] == 1).any(axis=1), train_orig_outlier_cols]
    X_val_outliers = X_val.loc[(X_val[train_outlier_cols] == 1).any(axis=1), train_orig_outlier_cols]
    print(f"\n# The train outliers:\n {X_train_outliers}")
    alpha = 0.01  # significance level
    dfs_dict_to_test = {'X_val_outliers': X_val_outliers}
    train_val_outlier_means_test = test_if_features_statistically_different(X_train_outliers, dfs_dict_to_test,
                                                                            alpha=alpha)
    print('\n# Test if train and validation outliers means are statisically not different:\n',
          train_val_outlier_means_test)

    # Impute outliers features

    train_statistics = add_winsorization_values_to_train_statistics(X_train, train_statistics)

    X_train = winsorize_outliers(X_train, train_statistics)
    X_val = winsorize_outliers(X_val, train_statistics)
    X_test = winsorize_outliers(X_test, train_statistics)

    ## Engineer new features

    X_train, new_features_list = engineer_new_features(X_train)
    X_val, _ = engineer_new_features(X_val)
    X_test, _ = engineer_new_features(X_test)

    train_statistics = add_new_features_statistics_to_train_statistics(X_train, train_statistics, new_features_list)
    train_statistics = add_binary_property_to_train_statistics(train_statistics, 'is_engineered', new_features_list)
    train_statistics = add_binary_property_to_train_statistics(train_statistics, 'is_engineered',
                                                               [feature + "_is_outlier" for feature in
                                                                train_outiler_cols])  # outlier features are also engineered

    n_new_features = train_statistics['is_engineered'].sum()
    print(f"\n# Including outliers, we have successfully created {n_new_features} new engineered features!")

    ## Drop highly correlated features

    X_train, correlated_dropped_features = drop_high_correlation_features(X_train, train_statistics, method='pearson',
                                                                          high_correlation_threshold=0.9,
                                                                          add_print=add_print)
    train_statistics = add_binary_property_to_train_statistics(train_statistics, 'is_correlated_to_drop',
                                                               correlated_dropped_features)
    correlated_dropped_features = get_train_feautres_with_property(train_statistics, 'is_correlated_to_drop')
    X_val = X_val.drop(columns=correlated_dropped_features)
    X_test = X_test.drop(columns=correlated_dropped_features)

    ## Normalize dataset

    # Update train_statistics with current X_train mean and std
    train_statistics['mean'] = X_train.mean().T
    train_statistics['std'] = X_train.std().T
    # standardize train, validation and test
    X_train = standardize_df_using_train_statistics(X_train, train_statistics, add_print=add_print)
    X_val = standardize_df_using_train_statistics(X_val, train_statistics, add_print=add_print)
    X_test = standardize_df_using_train_statistics(X_test, train_statistics, add_print=add_print)

    ## Balance Target

    # Update train statistics with the target distribution and mark
    train_statistics = add_target_to_train_statistics(y_train, train_statistics, target)

    # Concatenate the feature with the target before upsampling
    train = pd.concat([X_train, y_train], axis=1)

    upsample_target_minority_fn = partial(upsample_target_minority, the_train_statistics=train_statistics,
                                          random_state=42)
    X_train, y_train = upsample_target_minority_fn(train)
    # print(f"\n# After upsampling, the current train target distribution is:\n{y_train.value_counts(normalize=False, dropna=False)}")

    ## final preprocessing

    # fix columns names
    X_train = replace_columns_spaces_with_underscores(X_train)
    X_val = replace_columns_spaces_with_underscores(X_val)
    X_test = replace_columns_spaces_with_underscores(X_test)
    train_statistics = replace_columns_spaces_with_underscores(train_statistics.T).T

    ## Model Selection
    algos = [KNeighborsClassifier(), SVC(probability=True), LogisticRegression(), DecisionTreeClassifier(),
                         RandomForestClassifier(), GradientBoostingClassifier(), MLPClassifier()]
    # features_to_drop_from_train_statistics_in_first='is_engineered'
    features_to_drop_from_train_statistics_in_first = None
    train_eval_plot_classification_algos_two_datasets(X_train, y_train, X_val, y_val, train_statistics,
                                                      the_algos=algos,
                                                      features_to_drop_from_train_statistics_in_first=features_to_drop_from_train_statistics_in_first,
                                                      export_metrics_to_csv=True, to_plot=True,
                                                      filename=METRICS_FILENAME)

    ## Feature Selection

    # feed backward features selection - skipped

    # model = RandomForestClassifier(random_state=0)
    # metric_name = 'AUC'
    # metric_fn = get_AUC_score
    # n_features = None
    # best_model, best_feed_backward_set_of_features = find_best_model_greedy_feed_backward(
    #     the_X_train=X_train, the_y_train=y_train, the_X_val=X_val, the_y_val=y_val, the_model=model, the_metric_fn=metric_fn, the_metric_name=metric_name, n_features=n_features)
    #
    # print("# The best feed backward set of features are:", best_feed_backward_set_of_features)

    # feed forward features selection
    # assign features we've already found are best, instead of rerunning a feed forward search
    best_feed_forward_set_of_features = [
        'alcohol', 'total_acidity/free_sulfur_dioxide', 'sulphates',
        'density', 'free_sulfur_dioxide/total_sulfur_dioxide',
        'free_sulfur_dioxide', 'pH_is_outlier', 'density_is_outlier',
        'chlorides', 'pH', 'total_sulfur_dioxide', 'fixed_acidity',
        'residual_sugar_is_outlier', 'residual_sugar',
        'volatile_acidity_is_outlier', 'citric_acid',
        'chlorides_is_outlier', 'free_sulfur_dioxide_is_outlier',
        'alcohol_is_outlier', 'type_white', 'citric_acid_is_outlier',
        'sulphates_is_outlier', 'fixed_acidity_is_outlier', 'volatile_acidity']

    if not best_feed_forward_set_of_features:
        model = RandomForestClassifier(random_state=0)
        metric_name = 'AUC'
        metric_fn = get_AUC_score
        n_features = None
        best_model, best_feed_forward_set_of_features = find_best_model_greedy_feed_forward(
            the_X_train=X_train, the_y_train=y_train, the_X_val=X_val, the_y_val=y_val, the_model=model,
            the_metric_fn=metric_fn, the_metric_name=metric_name, n_features=n_features)

    print("# The best feed forward set of features are:", best_feed_forward_set_of_features)

    train_statistics = add_binary_property_to_train_statistics(
        train_statistics, 'is_feature_selected',
        best_feed_forward_set_of_features)

    selected_features = get_train_feautres_with_property(train_statistics, 'is_feature_selected')
    print(f"# The model with {len(selected_features)} selected features: {selected_features}")

    X_train = X_train[selected_features]
    X_val = X_val[selected_features]
    plot_time = 'unique'
    train_evaluate_plot_report_sklearn_classification_model_engineered_features = partial(
        train_evaluate_plot_report_sklearn_classification_model, the_X_train=X_train, the_y_train=y_train,
        the_X_val=X_val, the_y_val=y_val, export_metrics_to_csv=True, plot_time=plot_time, to_plot=False,
        filename=METRICS_FILENAME)

    model = RandomForestClassifier(random_state=0)
    model_name_engineered = str(model) + "_selected_features"

    model_metrics, _ = train_evaluate_plot_report_sklearn_classification_model_engineered_features(the_model=model,
                                                                                                   the_model_name=model_name_engineered,
                                                                                                   axs=None)

    # important features selection
    feature_importance = pd.DataFrame.from_dict(model_metrics.iloc[-1]['feature_importance'], orient='index')
    important_features = get_important_features(feature_importance)

    # important_features = ['fixed_acidity', 'volatile_acidity', 'citric_acid', 'residual_sugar',
    #                       'chlorides', 'free_sulfur_dioxide', 'total_sulfur_dioxide', 'density', 'pH',
    #                       'sulphates', 'alcohol', 'type_white', 'total_acidity/free_sulfur_dioxide',
    #                       'free_sulfur_dioxide/total_sulfur_dioxide']

    train_statistics = add_binary_property_to_train_statistics(
        train_statistics, 'is_important',
        important_features)

    print(f"# The current model most important features: {important_features}")

    # Our most important features
    important_features = get_train_feautres_with_property(train_statistics, 'is_important')
    X_train = X_train[important_features]
    X_val = X_val[important_features]
    X_test = X_test[important_features]

    ## Model Optimization

    # optimize model complexity
    plt.rcParams.update({'font.size': 11})

    hyper_parameter = 'max_depth'
    range_values = np.arange(1, int(len(important_features) * 1.5))
    metric_fn = get_AUC_score
    metric_name = 'AUC'
    OPTIMIZE_MODEL = False
    if OPTIMIZE_MODEL:
        opt_res, val_optimized_value, train_optimized_value = optimize_model_complexity_early_stopping(model, X_train,
                                                                                                       y_train, X_val,
                                                                                                       y_val,
                                                                                                       hyper_parameter,
                                                                                                       range_values,
                                                                                                       metric_fn,
                                                                                                       metric_name)
    else:
        val_optimized_value = 14
        train_optimized_value = 8

    print(f"# Train optimized {hyper_parameter} model:")
    model = RandomForestClassifier(random_state=0)
    hyper_parameter = 'max_depth'
    model.set_params(**{hyper_parameter: val_optimized_value})
    model_name = str(model)

    model_metrics, _ = train_evaluate_plot_report_sklearn_classification_model_engineered_features(
        the_model=model, the_model_name=model_name, axs=None)

    print(f"# Validation optimized {hyper_parameter} model:")
    model = RandomForestClassifier(random_state=0)
    hyper_parameter = 'max_depth'
    model.set_params(**{hyper_parameter: train_optimized_value})
    model_name = str(model)

    model_metrics, _ = train_evaluate_plot_report_sklearn_classification_model_engineered_features(
        the_model=model, the_model_name=model_name, axs=None)

    print(f"# Train and Validation mean optimized {hyper_parameter} model:")
    # Load model function, with best model
    plot_time = 'unique'
    model = RandomForestClassifier(random_state=0)
    hyper_parameter = 'max_depth'
    val_optimized_value = 14
    train_optimized_value = 8
    mean_optimized_value = int((val_optimized_value + train_optimized_value) / 2)
    model.set_params(**{hyper_parameter: mean_optimized_value})  # # Our Best Model, max_depth=11
    train_evaluate_plot_report_sklearn_classification_model_engineered_features = partial(
        train_evaluate_plot_report_sklearn_classification_model,
        the_X_train=X_train,
        the_y_train=y_train,
        export_metrics_to_csv=True,
        plot_time=plot_time,
        the_model=model,
        filename=METRICS_FILENAME)

    model_metrics, _ = train_evaluate_plot_report_sklearn_classification_model_engineered_features(
        the_model_name=model_name,
        the_X_val=X_val,
        the_y_val=y_val,
        filename=METRICS_FILENAME)

    # Threshold moving

    best_threshold, best_score = move_threshold(model, X_train, y_train, X_val, y_val)
    # no substinal improvment - skip threshold moving

    ## Model Validation on test

    # Model Name
    model_name = 'prod_val_' + str(model) + '_best_' + str(len(important_features)) + '_features'

    model_metrics, _ = train_evaluate_plot_report_sklearn_classification_model_engineered_features(
        the_model_name=model_name,
        the_X_val=X_test,
        the_y_val=y_test,
        filename=METRICS_FILENAME)

    print(model_metrics)
    if KEEP_WORKING_DIR:
        os.chdir(orig_dir)


if __name__ == "__main__":
    main()
