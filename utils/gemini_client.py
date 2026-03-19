"""
Shared Gemini client utilities for services that use Google Gemini AI
"""
import json
import logging
import os

from google import genai
from google.genai import types
from google.oauth2 import service_account

logger = logging.getLogger(__name__)


def initialize_gemini_client() -> genai.Client:
    """Initialize Gemini client with service account credentials.

    Returns:
        Configured genai.Client instance

    Raises:
        ValueError: If required environment variables are not set
    """
    credentials_file_path = os.getenv("CREDENTIALS_FILE_PATH")
    project_id = os.getenv("PROJECT_ID")
    location = os.getenv("LOCATION", "us-central1")

    if not credentials_file_path or not project_id:
        raise ValueError("PROJECT_ID and CREDENTIALS_FILE_PATH must be set in environment variables")

    with open(credentials_file_path, 'r') as f:
        service_account_info = json.load(f)

    credentials = service_account.Credentials.from_service_account_info(
        service_account_info,
        scopes=['https://www.googleapis.com/auth/cloud-platform']
    )

    return genai.Client(
        vertexai=True,
        project=project_id,
        location=location,
        credentials=credentials
    )


def prepare_image_part(file_content: bytes, mime_type: str) -> types.Part:
    """Prepare image part for Gemini API calls."""
    return types.Part(
        inline_data=types.Blob(
            mime_type=mime_type,
            data=file_content
        )
    )


def extract_usage_metadata(response) -> dict:
    """Extract usage metadata from a Gemini response."""
    if hasattr(response, 'usage_metadata'):
        return {
            "prompt_token_count": response.usage_metadata.prompt_token_count,
            "candidates_token_count": response.usage_metadata.candidates_token_count,
            "total_token_count": response.usage_metadata.total_token_count
        }
    return {}
