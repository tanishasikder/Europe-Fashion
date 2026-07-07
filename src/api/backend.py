'''Ties everything together in fastapi'''

from fastapi.responses import RedirectResponse
from supabase import create_client, Client
from dotenv import load_dotenv
import os
from fastapi import Body, FastAPI, File, HTTPException, UploadFile
import torch
from PIL import Image
from schemas.input import ClothingRequest
from typing import Optional, List
from datetime import datetime, timedelta
from schemas.input import upload, root
load_dotenv()

SUPABASE_URL = os.environ.get('SUPABASE_URL')
BUCKET_NAME = os.environ.get('BUCKET_NAME')
SUPABASE_KEY = os.environ.get('SUPABASE_KEY')

supabase: Client = create_client(
    SUPABASE_URL, 
    SUPABASE_KEY
)

SUPABASE_BUCKET = supabase.storage.from_(BUCKET_NAME)

root()

