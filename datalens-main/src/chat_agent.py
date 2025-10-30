# chat_agent.py — Gemini 2.5 Flash data-aware chat agent
import re
import pandas as pd
from vertexai import init
from vertexai.preview.generative_models import GenerativeModel

# Initialize Vertex AI (using Workbench credentials)
PROJECT_ID = "wfargo-cdo25cbh-915"       # 🔹 replace with your project ID
LOCATION = "us-central1"
init(project=PROJECT_ID, location=LOCATION)

model = GenerativeModel("gemini-2.5-flash")

def extract_column_name(question: str, df: pd.DataFrame):
    """Try to detect which column the user refers to."""
    lower_cols = {c.lower(): c for c in df.columns}
    for word in re.findall(r"[A-Za-z0-9_]+", question.lower()):
        if word in lower_cols:
            return lower_cols[word]
    return None

def get_column_stats(df: pd.DataFrame, column: str):
    """Return basic numeric or categorical stats for a column."""
    s = df[column].dropna()
    stats = {}
    if pd.api.types.is_numeric_dtype(s):
        stats["min"] = float(s.min())
        stats["max"] = float(s.max())
        stats["mean"] = float(s.mean())
        stats["median"] = float(s.median())
        stats["std"] = float(s.std())
    else:
        stats["unique_values"] = int(s.nunique())
        sample_vals = s.unique()[:10]
        stats["sample_values"] = [str(v) for v in sample_vals]
    stats["nulls"] = int(df[column].isna().sum())
    stats["dtype"] = str(df[column].dtype)
    return stats

def answer_question_about_df(question: str, df: pd.DataFrame, profile=None):
    """
    Answers a question about a dataset using Gemini 2.5 Flash.
    It detects columns, computes basic stats, and sends structured context to Gemini.
    """
    if df is None:
        return "⚠️ Please upload a dataset first."

    # 1️⃣ Try to detect column reference
    col = extract_column_name(question, df)
    column_context = ""
    if col:
        try:
            col_stats = get_column_stats(df, col)
            column_context = f"Column '{col}' stats: {col_stats}\n"
        except Exception as e:
            column_context = f"Error extracting stats for '{col}': {e}\n"
    else:
        column_context = "No specific column detected.\n"

    # 2️⃣ Build dataset summary
    dataset_overview = f"Dataset shape: {df.shape}, Columns: {list(df.columns)}"
    if profile and "columns_overview" in profile:
        dataset_overview += "\nProfiling info available.\n"

    # 3️⃣ Construct Gemini prompt
    prompt = f"""
You are a data analyst AI assistant using Gemini 2.5 Flash.
Analyze this dataset context and question.

Dataset:
{dataset_overview}

{column_context}

User question:
{question}

Please provide:
- Clear, factual answer based on the dataset stats provided.
- If the question asks for min, max, average, etc., compute from column_stats.
- Include numeric values if available.
- Add short interpretations or recommendations.
"""

    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"⚠️ Gemini error: {e}"
