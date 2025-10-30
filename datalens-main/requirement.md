GitHub Copilot: # AI Agent for Data Analysis - Requirements Document

## 1. Project Overview
Development of an interactive AI agent using Python to perform automated exploratory data analysis (EDA) on uploaded data files, provide intelligent insights, and generate business logic rules. The system will feature a user-friendly interface for data upload and analysis, along with interactive Q&A capabilities.

## 2. System Goals

### 2.1 Data Analysis Interface
- Provide a user-friendly web interface for data upload
- Support multiple file formats (CSV, Excel, etc.)
- Perform comprehensive exploratory data analysis
- Display interactive visualizations and statistics
- Enable real-time analysis updates

### 2.2 Automated Analysis Capabilities
- Generate comprehensive statistical analysis
- Identify patterns and relationships
- Detect outliers and anomalies
- Provide automated data quality assessment
- Generate key insights and summaries

### 2.3 Interactive AI Assistant
- Enable natural language queries about the data
- Provide context-aware responses
- Support drill-down analysis requests
- Offer data-driven recommendations
- Maintain conversation history for context

### 2.4 Business Logic Generation
- Automatically generate Plain Business Logic (PBL) rules
- Provide field-specific validation rules
- Define data quality constraints
- Generate technical specifications
- Support custom rule modifications

## 3. Technical Requirements

### 3.1 Environment Setup
- Python 3.8+
- Required Python packages:
  - pandas
  - numpy
  - scikit-learn
  - matplotlib
  - seaborn
  - langchain
  - streamlit (for web interface)
  - jupyter (for development)
  - openai (for AI assistance)
  - plotly (for interactive visualizations)

### 3.2 Core Functionality

#### 3.2.1 Data Upload Interface
- Web-based file upload interface
- Support for multiple file formats
- File validation and preview
- Data sampling for large files
- Progress tracking for uploads

#### 3.2.2 Basic Statistical Analysis
- Calculate for each numerical column:
  - Minimum value
  - Maximum value
  - Mean
  - Median
  - Standard deviation
  - Quartiles
- Generate frequency distributions for categorical columns

#### 2.2.3 Data Quality Analysis
- Identify data types for each column
- Count distinct values
- Calculate null/missing values percentage
- Detect duplicate records
- Identify potential data entry errors

#### 2.2.4 Pattern Analysis
- Correlation analysis between numerical columns
- Chi-square tests for categorical variables
- Time series pattern detection (if applicable)
- Seasonal decomposition for temporal data

#### 2.2.5 Outlier Detection
- Z-score analysis
- IQR method
- Isolation Forest for complex outlier detection
- Visual box plots for outlier representation

#### 2.2.6 Anomaly Detection
- Statistical process control
- Machine learning based anomaly detection
- Pattern deviation analysis

## 3. Output Requirements

### 4.1 Automated Insights Generation
- Key findings summary
- Trend identification
- Correlation highlights
- Data quality issues
- Business recommendations
- Natural language explanations

### 4.2 Interactive Query System
- Natural language query processing
- Context-aware responses
- Data-driven explanations
- Dynamic visualization generation
- Follow-up question handling

### 4.3 Business Logic Generator
- Field-specific PBL rules generation
- Common validation rules:
  - Data type constraints
  - Length restrictions
  - Null/Not null rules
  - Character set restrictions
  - Range validations
  - Format patterns
  - Relationship rules
- Custom rule definition interface
- Rule export functionality

### 4.4 Visualizations
- Histograms
- Box plots
- Correlation matrices
- Scatter plots
- Time series plots (if applicable)

## 5. Sample Implementation Structure

```python
from langchain.agents import Agent, Tool
from langchain.llms import OpenAI
import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, List

class InteractiveDataAnalysisAgent:
    def __init__(self):
        self.llm = OpenAI()
        self.data = None
        self.insights = {}
        self.tools = self._setup_tools()

    def _setup_tools(self) -> List[Tool]:
        return [
            Tool(
                name="upload_data",
                func=self._handle_file_upload,
                description="Handles file upload and validation"
            ),
            Tool(
                name="analyze_data",
                func=self._perform_analysis,
                description="Performs comprehensive data analysis"
            ),
            Tool(
                name="generate_insights",
                func=self._generate_insights,
                description="Generates key insights from analysis"
            ),
            Tool(
                name="answer_query",
                func=self._answer_data_query,
                description="Answers questions about the data"
            ),
            Tool(
                name="generate_pbl",
                func=self._generate_business_logic,
                description="Generates Plain Business Logic rules"
            )
        ]

    def _handle_file_upload(self, file):
        try:
            self.data = pd.read_csv(file)  # Add support for other formats
            return {"status": "success", "message": "File uploaded successfully"}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def _perform_analysis(self) -> Dict:
        analysis = {
            'basic_stats': self._calculate_basic_stats(),
            'pattern_analysis': self._analyze_patterns(),
            'relationships': self._analyze_relationships(),
            'quality_metrics': self._assess_data_quality()
        }
        return analysis

    def _generate_business_logic(self, field_name: str) -> Dict:
        field_type = str(self.data[field_name].dtype)
        unique_values = self.data[field_name].nunique()
        
        rules = {
            "field_name": field_name,
            "data_type": field_type,
            "validations": []
        }
        
        # Generate type-specific rules
        if field_type in ['int64', 'float64']:
            rules["validations"].extend([
                f"1.a) Only numeric values allowed",
                f"1.b) Value range: {self.data[field_name].min()} to {self.data[field_name].max()}",
                f"1.c) Null values: {'not allowed' if self.data[field_name].isnull().sum() == 0 else 'present'}"
            ])
        elif field_type == 'object':
            rules["validations"].extend([
                f"1.a) Text field with {unique_values} distinct values",
                f"1.b) Maximum length: {self.data[field_name].str.len().max()}",
                f"1.c) Special characters: {'present' if self.data[field_name].str.contains('[^A-Za-z0-9]').any() else 'not allowed'}"
            ])
        
        return rules

    def answer_question(self, question: str) -> str:
        context = {
            "data": self.data,
            "analysis": self._perform_analysis(),
            "question": question
        }
        return self._generate_response(context)
````

## 6. Implementation Timeline
1. Environment Setup & Web Interface (3 days)
2. Data Upload & Basic Analysis (4 days)
3. Interactive Query System (5 days)
4. Business Logic Generator (4 days)
5. Automated Insights Engine (4 days)
6. Testing & Integration (5 days)

## 7. Testing Requirements
- Unit tests for each analysis component
- Integration tests for the complete workflow
- Performance testing with large datasets
- Error handling and edge cases

## 8. Documentation Requirements
- Setup instructions and dependencies
- User interface guide
- API documentation
- Query examples and templates
- Business logic rules documentation
- Troubleshooting guide

## 9. Future Enhancements
- Support for more complex file formats
- Advanced AI model integration
- Custom business logic templates
- API endpoint for remote access
- Automated report generation
- Collaborative analysis features
- Integration with BI tools
- Export capabilities for insights and rules

This requirement document provides a foundation for building a comprehensive AI agent for data analysis. The implementation should be modular and extensible to accommodate future enhancements and modifications.