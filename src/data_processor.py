"""
This module contains the various procedures for processing data.
"""

import argparse
import numpy as np
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)

def load_data(data_path):
    """
    Read dataset from given directory.
        Parameters:
            data_path (str): directory containing dataset in csv
        Returns:
            df: dataframe containing the input data
    """
    df = pd.read_csv(data_path)
    return df

def save_data(data_path, df):
    """
    Save data to directory.
        Parameters:
            data_path (str): Directory for saving dataset
            df: Dataframe containing data to save
        Returns:
            None: No returns required
    """
    df.to_csv(data_path.replace('.csv','_processed.csv'), index=False)
    return None

def log_txf(df, cols: list):
    """
    Perform log transformation on specified columns in dataset.
        Parameters:
            df: input dataframe
            cols (list): columns that need log transformation
        Returns:
            df: resultant dataframe containing newly transformed columns
    """
    for col in cols:
        df[col] = df[col].clip(lower=0)
        df['log_'+col] = np.log(df[col]+1)
    return df

def remap_dependents(x):
    """
    Convert no_of_dependents into categorical variable.
        Parameters:
            x (int): Input category
        Returns:
            New category in (str)
    """
    if x == 0:
        return 'no_dep'
    if x == 1:
        return '1_dep'
    if x > 1 and x <= 3:
        return '2_to_3_dep'
    return 'more_than_3_dep'

def preprocess(df):
    """
    Orchestrate data pre-processing procedures.
        Parameters:
            df: Input dataframe to be pre-processed
        Returns:
            df: Resultant dataframe after pre-processing
    """
    df = log_txf(df, ['residential_assets_value','loan_amount'])
    df['dep_cat'] = df['no_of_dependents'].map(remap_dependents)
    return df

def run(data_path):
    """
    Main script to read and pre-process data.
        Parameters:
            data_path (str): Directory containing dataset in csv
        Returns:
            df: Dataframe containing the final pre-processed data
    """
    logging.info('Load data..')
    df = load_data(data_path)
    logging.info('Processing data...')
    df = preprocess(df)
    logging.info('Save data...')
    save_data(data_path, df)
    logging.info('Completed')
    return df

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--data_path", type=str)
    args = argparser.parse_args()
    run(args.data_path)