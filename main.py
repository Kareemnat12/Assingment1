import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
'''is it allowed to import os ? '''
import os
import plotly.express as px
from fontTools.misc.cython import returns
import streamlit as st



# ************************ Load Data Function ********************************

def load_data(filepath: str):
    # To Handle if the file is not found
    try:
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"The file {filepath} does not exist.")

        # Check the extention if it excel or csv
        if filepath.endswith('.csv'):
            data = pd.read_csv(filepath)
        elif filepath.endswith(('.xls', '.xlsx')):
            data = pd.read_excel(filepath)
        else:
            raise ValueError("Unsupported file format. Please provide a CSV or Excel file.")

        # Optional: Check if the DataFrame is empty
        if data.empty:
            raise ValueError("The loaded file is empty.")

        return data
    #print the error that occured in file reading or loading
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except ValueError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")






# ********************* Group and Aggregate Data *****************************
def group_and_aggregate_data(df: pd.DataFrame, group_by_column: str, agg_func) -> pd.DataFrame:
    # Performing the functionality
    result = df.drop(columns='ballot_code').groupby(group_by_column).agg(agg_func)
    return result
## we should ask about the agg fun wheter ists texst or function




# ********************** Remove Sparse Columns********************************
def remove_sparse_columns(df: pd.DataFrame, threshold: int):
        always_keep = ['city_name', 'ballot_code'] # ok i think here there is no need to put ballot code
        numeric_df = df.select_dtypes(include='number')
        filtered_columns = [col for col in numeric_df.columns
                                if numeric_df[col].sum() >= threshold] # i think and dont know but there is way much easier chat gpt:     filtered_columns = numeric_df.columns[numeric_df.sum() >= threshold].tolist()---- see below

        filtered_columns = always_keep + filtered_columns
        filtered_columns = list(set(filtered_columns))
        ordered_columns = [col for col in df.columns
                            if col in filtered_columns]
        return df[ordered_columns]






# ***************************************************************
#           dimensionality reduction
# **************************************************************

def dimensionality_reduction(df: pd.DataFrame, num_components: int, meta_columns: list[str]) -> pd.DataFrame:

    metadata = df[meta_columns]
    data = df.drop(columns=meta_columns)
    centered_data = data - data.mean()
    covariance_matrix = np.cov(centered_data, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]
    principal_components = eigenvectors[:, :num_components]
    #to make the custom function match the pca function
    for i in range(principal_components.shape[1]):
        if np.sum(principal_components[:, i]) < 0:
            principal_components[:, i] *= -1
    reduced_data = np.dot(centered_data, principal_components)
    reduced_df = pd.DataFrame(reduced_data, columns=[f"PC{i + 1}" for i in range(num_components)])
    result = pd.concat([metadata.reset_index(drop=True), reduced_df], axis=1)
    return result

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


# ***************************************************************
#          Analyze and Visualize Function
# ***************************************************************
def analyze_and_visualize(filepath: str):
    """
    Perform the full analysis and visualization process.
    """
    # Load the data
    df = load_data(filepath)
    print("Loaded Columns:", df.columns.tolist())  # Debugging output

    # Remove sparse columns
    df.columns = df.columns.str.strip()  # Ensure column names are clean
    filtered_df = remove_sparse_columns(df, threshold=1000)
    print("Filtered Columns:", filtered_df.columns.tolist())  # Debugging output

    # Dimensionality reduction for cities
    meta_columns = ["city_name"]
    for col in meta_columns:
        if col not in filtered_df.columns:
            raise KeyError(f"Column '{col}' not found in the DataFrame!")
    reduced_cities_df = dimensionality_reduction(filtered_df, num_components=2, meta_columns=meta_columns)
    print("Reduced Cities DataFrame:", reduced_cities_df.head())

    # Visualize city data
    fig_cities = px.scatter(reduced_cities_df, x="PC1", y="PC2", hover_data=meta_columns, title="Dimensionality Reduction for Cities")
    fig_cities.show()

    # Transpose for parties
    transposed_df = filtered_df.set_index("city_name").transpose()
    transposed_df.reset_index(inplace=True)
    transposed_df.rename(columns={"index": "party_name"}, inplace=True)
    print("Transposed DataFrame:", transposed_df.head())

    # Remove sparse rows for parties
    filtered_parties_df = remove_sparse_columns(transposed_df, threshold=1000)
    reduced_parties_df = dimensionality_reduction(filtered_parties_df, num_components=2, meta_columns=["party_name"])
    print("Reduced Parties DataFrame:", reduced_parties_df.head())

    # Visualize party data
    fig_parties = px.scatter(reduced_parties_df, x="PC1", y="PC2", hover_data=["party_name"], title="Dimensionality Reduction for Parties")
    fig_parties.show()



# ***************************************************************
#                          Main Execution
# ***************************************************************
def main():
    # Prompt the user for a file path or use a default file


    analyze_and_visualize('knesset_25.xlsx')


# Ensure the script runs only when executed directly
if __name__ == "__main__":
    main()
