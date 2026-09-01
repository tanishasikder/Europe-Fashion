from fastapi.responses import RedirectResponse
from supabase import create_client, Client
import os
from fastapi import Body, FastAPI, File, HTTPException, UploadFile
import torch
from PIL import Image
from src.schemas.input import ClothingRequest
from typing import Optional, List
from datetime import datetime, timedelta
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from postgrest.exceptions import APIError
import httpx
import logging

SUPABASE_URL = os.environ.get('SUPABASE_URL')
BUCKET_NAME = os.environ.get('BUCKET_NAME')
SUPABASE_KEY = os.environ.get('SUPABASE_KEY')

supabase: Client = create_client(
    SUPABASE_URL, 
    SUPABASE_KEY
)

SUPABASE_BUCKET = supabase.storage.from_(BUCKET_NAME)