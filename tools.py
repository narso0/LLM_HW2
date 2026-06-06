import os
import pandas as pd
import sys
from io import StringIO

#agent 1: data cleaner tools

def inspect_metadata(df):
    #grab basic info like shape and null counts
    buffer = StringIO()
    df.info(buf=buffer)
    return buffer.getvalue()

def get_column_stats(df, col):
    #get distribution stats or unique values for a specific column
    if col not in df.columns:
        return f"Error: Column '{col}' not found."
    return str(df[col].describe(include='all').to_dict())

def impute_missing(df, col, strategy='median'):
    #fill missing values so the model doesn't complain
    if col not in df.columns:
        return df
    if strategy == 'median':
        df[col] = df[col].fillna(df[col].median())
    elif strategy == 'mean':
        df[col] = df[col].fillna(df[col].mean())
    elif strategy == 'mode':
        df[col] = df[col].fillna(df[col].mode()[0])
    return df

def drop_column(df, col):
    #toss out a column we don't need
    if col in df.columns:
        df = df.drop(columns=[col])
    return df

#agent 2: feature engineer tools

def create_interaction(df, expression):
    #make a new feature by doing some math on existing ones
    try:
        col_name = expression.replace(" ", "_").replace("/", "_per_").replace("*", "_times_")
        df[col_name] = df.eval(expression)
        return df, f"Successfully created feature: {col_name}"
    except Exception as e:
        return df, f"Error creating feature: {e}"

def encode_categorical(df, col):
    #turn text categories into numbers (one-hot encoding)
    if col in df.columns:
        df = pd.get_dummies(df, columns=[col], drop_first=True)
    return df

def correlation_analysis(df, target):
    #see which features actually correlate with our target variable
    if target not in df.columns:
        return "Target column not found."
    corr = df.select_dtypes(include=['number']).corr()[target].sort_values(ascending=False)
    return corr.to_string()
#agent 3: model trainer tools

def execute_python_code(code_string):
    #dangerously run whatever code the llm spits out and catch the printed output
    old_stdout = sys.stdout
    redirected_output = sys.stdout = StringIO()

    try:
        exec(code_string, globals())
        sys.stdout = old_stdout
        return redirected_output.getvalue(), None
    except Exception as e:
        sys.stdout = old_stdout
        return "", str(e)