# profiler.py
import pandas as pd
import streamlit as st
import numpy as np
import re
from sklearn.ensemble import IsolationForest

st.set_page_config(page_title="Data Profiling Assistant", layout="wide")
st.title("📊 Data Profiling Assistant (Gemini 2.5 Flash)")

def detect_pattern(series, sample_n=500):
    s = series.dropna().astype(str)
    if s.empty:
        return "empty"
    sample = s.sample(min(len(s), sample_n)).astype(str)
    patterns = set()
    for v in sample:
        if re.fullmatch(r'\d+', v):
            patterns.add("numeric")
        if re.fullmatch(r'[A-Za-z ]+', v):
            patterns.add("alpha")
        if re.search(r'@', v):
            patterns.add("email-like")
        if re.search(r'\d{2,4}[-/]\d{1,2}[-/]\d{1,2}', v):
            patterns.add("date-like")
    return ", ".join(sorted(patterns)) if patterns else "mixed"

def compute_outliers(series):
    s = pd.to_numeric(series, errors='coerce').dropna()
    if s.empty:
        return {'method':'none', 'count':0}
    q1 = s.quantile(0.25)
    q3 = s.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    out = s[(s < lower) | (s > upper)]
    return {'method':'IQR', 'count': int(out.count()), 'lower':float(lower), 'upper':float(upper)}

def isolation_anomaly_scores(df, n_estimators=100):
    numeric = df.select_dtypes(include=[np.number]).fillna(0)
    if numeric.shape[1] == 0:
        return None
    iso = IsolationForest(n_estimators=n_estimators, contamination=0.01, random_state=42)
    iso.fit(numeric)
    scores = iso.decision_function(numeric)
    return pd.Series(scores, index=df.index)

def profile_dataframe(df: pd.DataFrame):
    overview = {}
    top_insights = []
    for col in df.columns:
        s = df[col]
        dtype = str(s.dtype)
        cnt = len(s)
        nulls = int(s.isna().sum())
        distinct = int(s.nunique(dropna=True))
        sample_nonnull = s.dropna().head(5).astype(str).tolist()
        patterns = detect_pattern(s)
        col_stats = {'dtype':dtype, 'count':cnt, 'nulls':nulls, 'distinct':distinct,
                     'sample_values': sample_nonnull, 'pattern':patterns}
        if pd.api.types.is_numeric_dtype(s):
            s_num = pd.to_numeric(s, errors='coerce')
            col_stats.update({
                'min': float(s_num.min()) if s_num.count()>0 else None,
                'max': float(s_num.max()) if s_num.count()>0 else None,
                'mean': float(s_num.mean()) if s_num.count()>0 else None,
                'median': float(s_num.median()) if s_num.count()>0 else None,
                'std': float(s_num.std()) if s_num.count()>0 else None,
                'outliers': compute_outliers(s)
            })
            if col_stats['outliers']['count'] > 0:
                top_insights.append(f"Column '{col}' has {col_stats['outliers']['count']} potential outliers (IQR).")
        else:
            if nulls > 0:
                top_insights.append(f"Column '{col}' has {nulls} nulls.")
            if distinct / max(1,cnt) < 0.02:
                top_insights.append(f"Column '{col}' appears low-cardinality ({distinct} unique). may be categorical.")
        overview[col] = col_stats

    numeric_df = df.select_dtypes(include=[np.number])
    correlations = None
    if numeric_df.shape[1] > 1:
        correlations = numeric_df.corr().round(3)

    anomalies = isolation_anomaly_scores(df)

    return {
        'columns_overview': overview,
        'correlations': correlations,
        'anomaly_scores': anomalies,
        'top_insights': top_insights
    }
