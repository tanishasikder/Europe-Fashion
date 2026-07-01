from fastapi import APIRouter, Request, Depends, UploadFile, File
from fastapi.responses import HTMLResponse, RedirectResponse
from src.database import supabase, SUPABASE_BUCKET, SUPABASE_URL
from models import image_extracton_model
