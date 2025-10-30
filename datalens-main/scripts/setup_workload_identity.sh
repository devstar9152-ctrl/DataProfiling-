#!/bin/bash

# This script helps configure Workload Identity Federation for the DataLens application
# It should be run by a user with appropriate Google Cloud permissions

# Exit on error
set -e

# Load environment variables
source .env

# Validate required environment variables
if [[ -z "$GOOGLE_CLOUD_PROJECT" ]]; then
    echo "Error: GOOGLE_CLOUD_PROJECT is not set in .env file"
    exit 1
fi

echo "Configuring Workload Identity Federation for project: $GOOGLE_CLOUD_PROJECT"

# Get the project number
PROJECT_NUMBER=$(gcloud projects describe $GOOGLE_CLOUD_PROJECT --format='value(projectNumber)')

# Create Workload Identity Pool
echo "Creating Workload Identity Pool..."
gcloud iam workload-identity-pools create "datalens-pool" \
    --location="global" \
    --display-name="DataLens Pool" \
    --project=$GOOGLE_CLOUD_PROJECT

# Get the Workload Identity Pool ID
POOL_ID=$(gcloud iam workload-identity-pools describe "datalens-pool" \
    --location="global" \
    --project=$GOOGLE_CLOUD_PROJECT \
    --format='value(name)')

# Create Service Account
echo "Creating Service Account..."
gcloud iam service-accounts create "datalens-sa" \
    --display-name="DataLens Service Account" \
    --project=$GOOGLE_CLOUD_PROJECT

# Grant necessary permissions
echo "Granting IAM permissions..."
gcloud projects add-iam-policy-binding $GOOGLE_CLOUD_PROJECT \
    --member="serviceAccount:datalens-sa@$GOOGLE_CLOUD_PROJECT.iam.gserviceaccount.com" \
    --role="roles/aiplatform.user"

# Update .env file with new values
echo "Updating .env file with new configuration..."
echo "WORKLOAD_IDENTITY_PROVIDER=projects/$PROJECT_NUMBER/locations/global/workloadIdentityPools/datalens-pool/providers/github-provider" >> .env
echo "SERVICE_ACCOUNT=datalens-sa@$GOOGLE_CLOUD_PROJECT.iam.gserviceaccount.com" >> .env

echo "Configuration complete!"
echo "Please ensure you configure your identity provider (GitHub, Azure AD, etc.) with the following values:"
echo "Workload Identity Pool ID: $POOL_ID"
echo "Service Account: datalens-sa@$GOOGLE_CLOUD_PROJECT.iam.gserviceaccount.com"