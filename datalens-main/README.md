# DataLens Application Setup Guide

## Prerequisites

1. Python 3.8 or higher
2. pip (Python package installer)
3. OpenAI API key

## Installation Steps

1. Clone the repository or download the source code

2. Create a virtual environment:
```bash
python -m venv venv
```

3. Activate the virtual environment:
- Windows:
```bash
.\venv\Scripts\activate
```
- Unix/MacOS:
```bash
source venv/bin/activate
```

4. Install required packages:
```bash
pip install -r requirements.txt
```

# Set up Google Cloud Workload Identity Federation:

1. Create a Google Cloud Project and enable the Vertex AI API

2. Configure Workload Identity Federation:

   a. Create a Workload Identity Pool:
   ```bash
   gcloud iam workload-identity-pools create "datalens-pool" \
     --location="global" \
     --display-name="DataLens Pool"
   ```

   b. Create a Workload Identity Provider (example for GitHub Actions):
   ```bash
   gcloud iam workload-identity-pools providers create-oidc "github-provider" \
     --location="global" \
     --workload-identity-pool="datalens-pool" \
     --display-name="GitHub provider" \
     --attribute-mapping="google.subject=assertion.sub,attribute.actor=assertion.actor,attribute.repository=assertion.repository" \
     --issuer-uri="https://token.actions.githubusercontent.com"
   ```

   c. Grant IAM permissions:
   ```bash
   gcloud projects add-iam-policy-binding PROJECT_ID \
     --member="principalSet://iam.googleapis.com/projects/PROJECT_NUMBER/locations/global/workloadIdentityPools/datalens-pool/*" \
     --role="roles/aiplatform.user"
   ```

3. Configure Environment:

   Create a `.env` file in the root directory:
   ```
   GOOGLE_CLOUD_PROJECT=your_project_id_here
   WORKLOAD_IDENTITY_PROVIDER=projects/PROJECT_NUMBER/locations/global/workloadIdentityPools/datalens-pool/providers/github-provider
   SERVICE_ACCOUNT=datalens-sa@PROJECT_ID.iam.gserviceaccount.com
   ```

4. Configure Application Default Credentials:
   ```bash
   gcloud auth application-default login
   ```

Note: The exact configuration steps may vary depending on your identity provider (GitHub, Azure AD, etc.).

## Running the Application

1. Make sure your virtual environment is activated

2. Start the Streamlit application:
```bash
streamlit run src/app.py
```

3. Open your web browser and navigate to the URL shown in the terminal (typically http://localhost:8501)

## Usage Instructions

1. **Data Upload**
   - Navigate to the "Data Upload" section
   - Upload your CSV file using the file uploader
   - Preview your data in the table view

2. **Basic Analysis**
   - View basic statistical information about your dataset
   - Examine column profiles and data quality metrics
   - Visualize data distributions

3. **Pattern Analysis**
   - Explore correlations between variables
   - View distribution patterns
   - Identify trends and relationships

4. **Business Logic**
   - Select a field to generate business logic rules
   - View validation rules and constraints
   - Export rules for implementation

5. **Ask Questions**
   - Type natural language questions about your data
   - Get AI-powered insights and answers
   - Explore your data interactively

## Troubleshooting

1. If you encounter OpenAI API errors:
   - Verify your API key in the .env file
   - Check your OpenAI account status and quota

2. For data loading issues:
   - Ensure your CSV file is properly formatted
   - Check for encoding issues
   - Verify file permissions

3. If visualizations don't render:
   - Check your internet connection (Plotly requires internet)
   - Try refreshing the page
   - Clear your browser cache

## Support

For additional support or to report issues, please refer to the project's issue tracker.