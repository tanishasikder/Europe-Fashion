from fastapi import APIRouter, Request, Depends, UploadFile, File
from fastapi.responses import HTMLResponse, RedirectResponse
from src.app.services.db_service.database import supabase, SUPABASE_BUCKET, SUPABASE_URL
from models import image_extraction
from fastapi.templating import Jinja2Templates
import os

TEMPLATE_PATH=os.getenv('TEMPLATE_PATH')

templates = Jinja2Templates(directory=TEMPLATE_PATH)

router = APIRouter()

@router.get("/", response_class=HTMLResponse)
async def read_clothes(request: Request):
    response = supabase.table('clothes').select('*').eq('is_active', True).execute()
    clothes = response.data
    return templates.TemplateResponse('info.html', {'request': request, 'clothes': clothes})

@router.get('/add', response_class=HTMLResponse)
async def add_clothes_form(request: Request):
    return templates.TemplateResponse('add_clothes.html', {'request': request})

@router.post('/add')
async def add_clothes(
    clothes : image_extraction = Depends(image_extraction.as_form),
    image: UploadFile = File(None)
):
    image_url = None
    if image and image.filename != "":
        image_filename = f"{clothes.color}_{clothes.category}_{image.filename}"
        file_content = await image.read()
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

