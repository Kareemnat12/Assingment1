import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
from fontTools.misc.cython import returns


# ***************************************************************
#                     Load Data Function
# ***************************************************************

def load_data(filepath: str):
    return pd.read_excel(filepath)
    #I have question here, should i make it input output by user?
    #or just get the name of the file by a string variable ?
# ***************************************************************


# ***************************************************************
#               Group and Aggregate Data
# ***************************************************************
def group_and_aggregate_data(df: pd.DataFrame, group_by_column: str, agg_func):
    return df.groupby(group_by_column).agg(agg_func)
#im not sure if the agg_func is a string or not because it gives me a warning if it was for instanc np.mean
#it prefers using string
## ***************************************************************

# ***************************************************************
#               Remove Sparse Columns
# ***************************************************************
def remove_sparse_columns(df: pd.DataFrame, threshold: int):
        always_keep = ['city_name', 'ballot_code']
        numeric_df = df.select_dtypes(include='number')
        filtered_columns = [col for col in numeric_df.columns
                                if numeric_df[col].sum() >= threshold]
        filtered_columns = always_keep + filtered_columns
        filtered_columns = list(set(filtered_columns))
        ordered_columns = [col for col in df.columns
                            if col in filtered_columns]
        return df[ordered_columns]

# ***************************************************************


# ***************************************************************
#               remove sparse columns 
# ***************************************************************
def remove_sparse_columns(df: pd.DataFrame, threshold: int):
        # Always include these columns
        always_keep = ['city_name', 'ballot_code']

        # Add numeric columns based on the sum threshold
        numeric_df = df.select_dtypes(include='number')
        filtered_columns = [col for col in numeric_df.columns
                                if numeric_df[col].sum() >= threshold]

        # Combine the always-keep columns with the filtered numeric columns
        filtered_columns = always_keep + filtered_columns

        # Remove duplicates by converting to a set and back to a list
        filtered_columns = list(set(filtered_columns))

        # Preserve the original order of the columns
        ordered_columns = [col for col in df.columns
                            if col in filtered_columns]

        return df[ordered_columns]

# this for test fuctions
# print("################test functions ###########")
# print("#####################this is the load ###################")
# df = load_data("knesset_25.xlsx")
# print(df.head())
#
# print("#####################this is the group ###################")
# af=group_and_aggregate_data(df,"city_name",'mean')
# print(af.head())
#
## print("#####################this is the remove sparse ###################")
#filtered_df = remove_sparse_columns(df,1000)
# print("Filtered DataFrame:")
# print(filtered_df)
