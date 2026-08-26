from fastapi import APIRouter, Request, Depends, UploadFile, File
from fastapi.responses import HTMLResponse, RedirectResponse
from src.services.db_service.database import supabase, SUPABASE_BUCKET, SUPABASE_URL
from models import image_extraction
from fastapi.templating import Jinja2Templates
import os

def add_clothes(
    file_content : bytes,
    file_name : str,
    clothes : image_extraction
):
    image_url = None
    if file_content and file_name != "":
        image_filename = f"{clothes.color}_{clothes.category}_{file_name}"
        response = supabase.storage.from_(SUPABASE_BUCKET).upload(image_filename, file_content)
        if response.status_code == 200:
            image_url = f"{SUPABASE_URL}/storage/v1/object/public/{SUPABASE_BUCKET}/{image_filename}"

    supabase.table('clothes').insert({
        'color': clothes.color,
        'category': clothes.category,
        'size': clothes.size,
        'price': clothes.price,
        'image_url': image_url
    }).execute()

    return RedirectResponse("/", status_code=303)

