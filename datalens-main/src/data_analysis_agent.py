from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import plotly.express as px
import plotly.graph_objects as go
from langchain.llms import VertexAI
from langchain.agents import Tool
import json
import vertexai
from vertexai.language_models import TextGenerationModel
from google.cloud import aiplatform

class DataAnalysisAgent:
    def __init__(self, project_id: str, credentials: Any = None, location: str = "us-central1"):
        """Initialize the Data Analysis Agent with Google Cloud project details and credentials."""
        # Initialize Vertex AI with workload identity credentials
        vertexai.init(
            project=project_id,
            location=location,
            credentials=credentials
        )
        
        # Initialize AI Platform with the same credentials
        aiplatform.init(
            project=project_id,
            location=location,
            credentials=credentials
        )
        
        self.llm = TextGenerationModel.from_pretrained("text-bison@001")
        self.data = None
        self.analysis_results = {}
        self.column_profiles = {}
        
    def load_data(self, file_path: str) -> Dict[str, Any]:
        """Load data from uploaded file."""
        try:
            self.data = pd.read_csv(file_path)
            self._generate_column_profiles()
            return {
                "status": "success",
                "message": "Data loaded successfully",
                "shape": self.data.shape
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error loading data: {str(e)}"
            }

    def _generate_column_profiles(self):
        """Generate profiles for each column in the dataset."""
        for column in self.data.columns:
            profile = self._analyze_column(column)
            self.column_profiles[column] = profile

    def _analyze_column(self, column: str) -> Dict[str, Any]:
        """Analyze a single column and generate its profile."""
        series = self.data[column]
        dtype = str(series.dtype)
        
        profile = {
            "name": column,
            "dtype": dtype,
            "null_count": series.isnull().sum(),
            "distinct_count": series.nunique(),
            "is_unique": series.is_unique,
        }

        if np.issubdtype(series.dtype, np.number):
            profile.update({
                "min": float(series.min()),
                "max": float(series.max()),
                "mean": float(series.mean()),
                "median": float(series.median()),
                "std": float(series.std()),
                "skew": float(series.skew())
            })
            # Detect outliers using IsolationForest
            if len(series.dropna()) > 10:
                outliers = self._detect_outliers(series)
                profile["outlier_count"] = int(sum(outliers == -1))
        
        elif series.dtype == 'object' or series.dtype == 'category':
            value_counts = series.value_counts()
            profile.update({
                "top_values": value_counts.head(5).to_dict(),
                "contains_numbers": any(str(x).isnumeric() for x in series.dropna()),
                "contains_special_chars": series.str.contains(r'[^a-zA-Z0-9\s]').any()
            })

        return profile

    def _detect_outliers(self, series: pd.Series) -> np.ndarray:
        """Detect outliers using Isolation Forest."""
        data = series.dropna().values.reshape(-1, 1)
        iso_forest = IsolationForest(contamination=0.1, random_state=42)
        return iso_forest.fit_predict(data)

    def generate_basic_stats(self) -> Dict[str, Any]:
        """Generate basic statistical analysis of the dataset."""
        if self.data is None:
            return {"error": "No data loaded"}

        return {
            "dataset_info": {
                "rows": len(self.data),
                "columns": len(self.data.columns),
                "total_cells": self.data.size,
                "missing_cells": self.data.isnull().sum().sum(),
                "memory_usage": self.data.memory_usage().sum()
            },
            "column_profiles": self.column_profiles
        }

    def analyze_patterns(self) -> Dict[str, Any]:
        """Analyze patterns in the data."""
        if self.data is None:
            return {"error": "No data loaded"}

        numeric_columns = self.data.select_dtypes(include=[np.number]).columns
        
        patterns = {
            "correlations": {},
            "trends": {},
            "distributions": {}
        }

        # Calculate correlations
        if len(numeric_columns) > 1:
            corr_matrix = self.data[numeric_columns].corr()
            patterns["correlations"] = corr_matrix.to_dict()

        # Analyze distributions
        for col in numeric_columns:
            patterns["distributions"][col] = {
                "skewness": float(self.data[col].skew()),
                "kurtosis": float(self.data[col].kurtosis()),
                "is_normal": self._test_normality(self.data[col])
            }

        return patterns

    def _test_normality(self, series: pd.Series) -> bool:
        """Simple test for normal distribution using skewness and kurtosis."""
        skew = abs(series.skew())
        kurt = abs(series.kurtosis())
        return skew < 0.5 and kurt < 0.5

    def generate_business_logic(self, field_name: str) -> Dict[str, Any]:
        """Generate Plain Business Logic (PBL) rules for a specific field."""
        if field_name not in self.column_profiles:
            return {"error": f"Field {field_name} not found in dataset"}

        profile = self.column_profiles[field_name]
        rules = {
            "field_name": field_name,
            "data_type": profile["dtype"],
            "validations": []
        }

        # Generate type-specific rules
        if np.issubdtype(self.data[field_name].dtype, np.number):
            rules["validations"].extend([
                f"Data type must be numeric",
                f"Value range: {profile['min']} to {profile['max']}",
                f"Null values: {'not allowed' if profile['null_count'] == 0 else 'allowed'}",
                f"Decimals: {'allowed' if 'float' in str(profile['dtype']) else 'not allowed'}"
            ])
            if profile.get('outlier_count', 0) > 0:
                rules["validations"].append(f"Outlier detection required (found {profile['outlier_count']} potential outliers)")

        elif profile["dtype"] == 'object':
            max_length = self.data[field_name].str.len().max()
            rules["validations"].extend([
                f"Maximum length: {max_length} characters",
                f"Special characters: {'present' if profile['contains_special_chars'] else 'not allowed'}",
                f"Numeric characters: {'present' if profile['contains_numbers'] else 'not allowed'}",
                f"Distinct values: {profile['distinct_count']}"
            ])

        return rules

    def answer_question(self, question: str) -> str:
        """Answer questions about the data using the Google PaLM model."""
        if self.data is None:
            return "No data has been loaded yet."

        # Create context from analysis results
        context = {
            "dataset_info": f"Dataset with {len(self.data)} rows and {len(self.data.columns)} columns",
            "column_info": json.dumps(self.column_profiles, default=str),
            "question": question
        }

        # Generate prompt for the LLM
        prompt = f"""
        You are a data analysis expert. Based on the following dataset information:
        {context['dataset_info']}
        
        Column profiles:
        {context['column_info']}
        
        Please answer this question:
        {question}
        
        Provide a clear and concise answer based only on the data available.
        Use bullet points where appropriate for better readability.
        """

        try:
            parameters = {
                "temperature": 0.3,
                "max_output_tokens": 1024,
                "top_p": 0.8,
                "top_k": 40
            }
            response = self.llm.predict(prompt, **parameters)
            return response.text
        except Exception as e:
            return f"Error generating response: {str(e)}"

    def generate_visualizations(self) -> Dict[str, Any]:
        """Generate relevant visualizations for the dataset."""
        if self.data is None:
            return {"error": "No data loaded"}

        visualizations = {}
        numeric_columns = self.data.select_dtypes(include=[np.number]).columns

        # Distribution plots for numeric columns
        for col in numeric_columns:
            fig = px.histogram(self.data, x=col, title=f'Distribution of {col}')
            visualizations[f"{col}_distribution"] = fig

        # Correlation heatmap for numeric columns
        if len(numeric_columns) > 1:
            corr_matrix = self.data[numeric_columns].corr()
            fig = px.imshow(corr_matrix, title='Correlation Heatmap')
            visualizations["correlation_heatmap"] = fig

        # Box plots for outlier detection
        for col in numeric_columns:
            fig = px.box(self.data, y=col, title=f'Box Plot of {col}')
            visualizations[f"{col}_boxplot"] = fig

        return visualizations