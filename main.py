#####   Library importing

import numpy as np
import pandas as pd
import os
import plotly.express as px



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

    return df.drop(columns='ballot_code').groupby(group_by_column).agg(agg_func).reset_index()


# ********************** Remove Sparse Columns********************************

def remove_sparse_columns(df: pd.DataFrame, threshold: int) -> pd.DataFrame:
    text_df = df.select_dtypes(include=['object'])
    numeric_df = df.select_dtypes(include=['number'])
    if "ballot_code" in df.columns:
        text_df["ballot_code"] = df["ballot_code"]
        numeric_df.drop("ballot_code", axis=1, inplace=True)

    column_sums = numeric_df.sum()
    filtered_columns = column_sums[column_sums >= threshold].index
    numeric_df = numeric_df[filtered_columns]

    final_df = pd.concat([text_df, numeric_df], axis=1)
    return final_df



# ***************************************************************
#           dimensionality reduction
# **************************************************************


def dimensionality_reduction(df: pd.DataFrame, num_components: int, meta_columns: list[str]) -> pd.DataFrame:

    metadata = df[meta_columns]
    data = df.drop(columns=meta_columns)

    # Standardize the data
    centered_data = data - data.mean()
    scaled_data = centered_data / data.std()

    # Compute the covariance matrix
    covariance_matrix = np.cov(scaled_data, rowvar=False)

    # Eigen decomposition: eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, sorted_indices]

    principal_components = eigenvectors[:, :num_components]

    # Flip signs if necessary to match sklearn behavior
    for i in range(principal_components.shape[1]):
        if np.sum(principal_components[:, i]) < 0:
            principal_components[:, i] *= -1

    # Project the data onto the principal components
    reduced_data = np.dot(scaled_data, principal_components)

    # Create a DataFrame for the reduced data (principal components)
    reduced_df = pd.DataFrame(reduced_data, columns=[f"PC{i + 1}" for i in range(num_components)])

    # Concatenate the metadata with the reduced DataFrame
    result = pd.concat([metadata.reset_index(drop=True), reduced_df], axis=1)

    return result


# ***************************************************************
#          Analyze and Visualize Function
# ***************************************************************


def CompareCitiesVisual(filepath):
    visual_df = load_data(filepath)
    visual_df = group_and_aggregate_data(visual_df, 'city_name', 'sum')
    Filtered_visual_df = remove_sparse_columns(visual_df, 1000)
    Reduced_visual_df = dimensionality_reduction(Filtered_visual_df, 2, ["city_name"])
    fig = px.scatter(Reduced_visual_df, x='PC1', y='PC2', color='city_name', title="PCA of Vote Data by City")
    fig.show()


def ComparePartiesVisual(filepath):
    visual2_df = load_data(filepath)
    visual2_df = group_and_aggregate_data(visual2_df, 'city_name', 'sum')

    visual2_df_transposed = visual2_df.set_index("city_name").T
    visual2_df_transposed = visual2_df_transposed.reset_index()

    visual2_df_transposed.columns.name = None  # Remove the name for the columns if needed
    visual2_df_transposed = visual2_df_transposed.rename(columns={'index': 'city_name'})

    b3 = remove_sparse_columns(visual2_df_transposed, 1000)
    b3 = dimensionality_reduction(visual2_df_transposed, 2, ['city_name'])
    fig = px.scatter(b3, x='PC1', y='PC2', color='city_name', title="PCA of Vote Data by City")
    fig.show()

