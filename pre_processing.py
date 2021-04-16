'''
April 13, 2021
Title: Implementation of regression techniques on time-series data to
generate future predictions
Module: Pre-processing
'''

#imports
import pandas as pd

'''
Read the csv dumped by input_processing module
The input is read in fixed sized chunks based on the target label for prediction
'''
def read_file(path, label, value):
    chunk_list=[]
    for chunk in pd.read_csv(path, chunksize=10000):
        chunk_list.append(chunk[chunk[label] == value])
        
    df = pd.concat(chunk_list)
    df.drop(df.columns[[0]], axis=1)
    return df

'''
Remove the faulty records
'''
def clean_dataframe(df):
    df['timestamp'].str.len() != 28) | (df['group'].str.len() < 6) | (df['group'].str.len() > 7)].index)
    return df

'''
Format the timestamp in datetime dtype to enable time series generation
'''
def format_datetime(df):
    df['date'] = pd.to_datetime(df['timestamp']).dt.strftime('%Y%m%d%H%M%S')
    return df
