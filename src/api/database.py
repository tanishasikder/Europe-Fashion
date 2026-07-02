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

load_dotenv()

SUPABASE_URL = os.environ.get('SUPABASE_URL')
SUPABASE_BUCKET = os.environ.get('SUPABASE_URL')
SUPABASE_KEY = os.environ.get('SUPABASE_KEY')

supabase: Client = create_client(
    SUPABASE_URL, 
    SUPABASE_KEY
)

# Image is uploaded somewhere else the raw bytes are just passed here
def store_image(matrix: List[ClothingRequest],
                image: bytes = None):
    
    image_url = None
    if image and image.filename != "":
        image_filename = f"{matrix.color}_{matrix.category}_{image.filename}"
        response = supabase.storage.from_(SUPABASE_BUCKET).upload(image_filename, image)
        if response.status_code == 200:
            image_url = f"{SUPABASE_URL}/storage/v1/object/public/{SUPABASE_BUCKET}/{image_filename}"

    supabase.table('clothes').insert({
        'color': matrix.color,
        'category': matrix.category,
        'size': matrix.size,
        'price': matrix.price,
        'image_url': image_url
    }).execute()

    return RedirectResponse("/", status_code=303)

def remove_expired_images():
    # Calculating the threshold of expiration
    time = datetime.now() - timedelta(hours=2)
    # Turning time into the format Supabase wants
    exp = time.isoformat()

    response = (
        supabase.table('clothes')
        .delete()
        .select('created_at')
        .lt(exp)
        .execute()
    )

    return response