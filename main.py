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




# this for test fuctions
# print("################test functions ###########")
# print("#####################this is the load ###################")
# df = load_data("knesset_25.xlsx")
# print(df.head())
#
# print("#####################this is the group ###################")
# af=group_and_aggregate_data(df,"city_name",'mean')
# print(af.head())