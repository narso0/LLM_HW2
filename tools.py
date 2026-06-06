import os
import pandas as pd
import sys
from io import StringIO


#agent 1: data cleaner tools

def inspect_metadata(df):
    #grab basic info like shape, null counts, and cardinality
    report = {
        "shape": df.shape,
        "columns": {}
    }

    for col in df.columns:
        report["columns"][col] = {
            "dtype": str(df[col].dtype),
            "null_count": int(df[col].isnull().sum()),
            "null_pct": round((df[col].isnull().mean() * 100), 2),
            "unique_count": int(df[col].nunique()),
            "unique_ratio": round(df[col].nunique() / len(df), 4)
        }

    return str(report)


def get_column_stats(df, col):
    #get distribution stats or unique values for a specific column
    if col not in df.columns:
        return f"Error: Column '{col}' not found."

    stats = {
        "dtype": str(df[col].dtype),
        "missing": int(df[col].isnull().sum()),
        "unique_count": int(df[col].nunique())
    }

    if pd.api.types.is_numeric_dtype(df[col]):
        stats.update({
            "mean": float(df[col].mean()) if not df[col].isnull().all() else None,
            "median": float(df[col].median()) if not df[col].isnull().all() else None,
            "std": float(df[col].std()) if not df[col].isnull().all() else None,
            "min": float(df[col].min()) if not df[col].isnull().all() else None,
            "max": float(df[col].max()) if not df[col].isnull().all() else None
        })
    else:
        stats["top_values"] = df[col].value_counts().head(10).to_dict()

    return str(stats)


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
    #toss out a column we don't need (like PassengerId or Ticket)
    if col in df.columns:
        df = df.drop(columns=[col])
    return df


#agent 2: feature engineer tools

def create_interaction(df, expression):
    #make a new feature by doing some math on existing ones
    try:
        col_name = expression.replace(" ", "_").replace("/", "_per_").replace("*", "_times_").replace("+", "_plus_")
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
    numeric_df = df.select_dtypes(include=['number'])
    if target not in numeric_df.columns:
        return "Target is not numeric."
    corr = numeric_df.corr()[target].sort_values(ascending=False)
    return corr.to_string()


def select_top_features(df, target_col, k=5):
    #drops useless columns and keeps only the top k most predictive numeric features
    numeric_df = df.select_dtypes(include=['number'])

    if target_col not in numeric_df.columns:
        return df, f"Error: Target '{target_col}' must be numeric."

    correlations = numeric_df.corr()[target_col].abs().sort_values(ascending=False)

    top_cols = correlations.head(int(k) + 1).index.tolist()

    categorical_cols = df.select_dtypes(exclude=['number']).columns.tolist()

    final_cols = list(set(top_cols + categorical_cols))

    if target_col not in final_cols:
        final_cols.append(target_col)

    df = df[final_cols]

    return df, f"Selected top {k} numeric features based on correlation with {target_col}."

#agent 3: model trainer tools

def execute_python_code(code_string):
    #safely run whatever code the llm spits out and catch the printed output
    old_stdout = sys.stdout
    redirected_output = sys.stdout = StringIO()

    try:
        #run the code
        sandbox = {}
        exec(code_string, globals(), sandbox)

        sys.stdout = old_stdout
        return redirected_output.getvalue(), None

    except Exception as e:
        sys.stdout = old_stdout
        return "", str(e)