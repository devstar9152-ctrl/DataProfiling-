import streamlit as st
import pandas as pd
import plotly.express as px
from data_analysis_agent import DataAnalysisAgent
import os
from dotenv import load_dotenv
from google.cloud import aiplatform
from google.auth import identity_pool
import google.auth

# Load environment variables
load_dotenv()

def initialize_google_auth():
    """Initialize Google Cloud authentication using Workload Identity Federation."""
    try:
        # Attempt to get credentials using Workload Identity Federation
        credentials, project_id = google.auth.default()
        
        # Ensure credentials can be refreshed
        if not credentials.valid:
            credentials.refresh(google.auth.transport.requests.Request())
            
        return project_id, credentials
    except Exception as e:
        st.error(f"Error initializing Google Cloud authentication: {str(e)}")
        st.error("Please ensure Workload Identity Federation is properly configured")
        st.stop()

# Initialize session state
if 'agent' not in st.session_state:
    project_id, credentials = initialize_google_auth()
    if not project_id:
        st.error("Failed to determine Google Cloud project ID")
        st.stop()
    st.session_state.agent = DataAnalysisAgent(
        project_id=project_id,
        credentials=credentials
    )

def main():
    st.title("DataLens: Interactive Data Analysis Assistant")
    st.sidebar.header("Navigation")
    
    # Navigation options
    page = st.sidebar.radio(
        "Choose a section:",
        ["Data Upload", "Basic Analysis", "Pattern Analysis", "Business Logic", "Ask Questions"]
    )
    
    if page == "Data Upload":
        show_data_upload()
    elif page == "Basic Analysis":
        show_basic_analysis()
    elif page == "Pattern Analysis":
        show_pattern_analysis()
    elif page == "Business Logic":
        show_business_logic()
    elif page == "Ask Questions":
        show_qa_interface()

def show_data_upload():
    st.header("Data Upload")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            # Save the uploaded file temporarily
            with open("temp_upload.csv", "wb") as f:
                f.write(uploaded_file.getvalue())
            
            # Load the data using the agent
            result = st.session_state.agent.load_data("temp_upload.csv")
            
            if result["status"] == "success":
                st.success(f"Data loaded successfully! Shape: {result['shape']}")
                st.subheader("Preview of the data:")
                st.dataframe(st.session_state.agent.data.head())
            else:
                st.error(result["message"])
                
            # Clean up temporary file
            os.remove("temp_upload.csv")
            
        except Exception as e:
            st.error(f"Error: {str(e)}")

def show_basic_analysis():
    if st.session_state.agent.data is None:
        st.warning("Please upload data first!")
        return
    
    st.header("Basic Statistical Analysis")
    
    # Get basic stats
    stats = st.session_state.agent.generate_basic_stats()
    
    # Display dataset info
    st.subheader("Dataset Overview")
    dataset_info = stats["dataset_info"]
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Rows", dataset_info["rows"])
    col2.metric("Total Columns", dataset_info["columns"])
    col3.metric("Missing Cells", dataset_info["missing_cells"])
    
    # Display column profiles
    st.subheader("Column Profiles")
    for col_name, profile in stats["column_profiles"].items():
        with st.expander(f"{col_name} ({profile['dtype']})"):
            st.write(profile)

def show_pattern_analysis():
    if st.session_state.agent.data is None:
        st.warning("Please upload data first!")
        return
    
    st.header("Pattern Analysis")
    
    # Get pattern analysis
    patterns = st.session_state.agent.analyze_patterns()
    
    if "error" not in patterns:
        # Display correlations
        if patterns["correlations"]:
            st.subheader("Correlation Analysis")
            st.write(pd.DataFrame(patterns["correlations"]))
        
        # Display distributions
        st.subheader("Distribution Analysis")
        for col, dist_info in patterns["distributions"].items():
            st.write(f"**{col}**")
            st.write(dist_info)
        
        # Show visualizations
        st.subheader("Visualizations")
        visualizations = st.session_state.agent.generate_visualizations()
        for viz_name, fig in visualizations.items():
            st.plotly_chart(fig)
    else:
        st.error(patterns["error"])

def show_business_logic():
    if st.session_state.agent.data is None:
        st.warning("Please upload data first!")
        return
    
    st.header("Business Logic Generator")
    
    # Field selection
    selected_field = st.selectbox(
        "Select a field to generate business logic:",
        st.session_state.agent.data.columns
    )
    
    if st.button("Generate Business Logic"):
        rules = st.session_state.agent.generate_business_logic(selected_field)
        if "error" not in rules:
            st.subheader(f"Business Logic for {selected_field}")
            st.write("**Data Type:**", rules["data_type"])
            st.write("**Validation Rules:**")
            for rule in rules["validations"]:
                st.write(f"- {rule}")
        else:
            st.error(rules["error"])

def show_qa_interface():
    if st.session_state.agent.data is None:
        st.warning("Please upload data first!")
        return
    
    st.header("Ask Questions About Your Data")
    
    question = st.text_input("What would you like to know about your data?")
    
    if question:
        with st.spinner("Analyzing your question..."):
            answer = st.session_state.agent.answer_question(question)
            st.write("**Answer:**")
            st.write(answer)

if __name__ == "__main__":
    main()