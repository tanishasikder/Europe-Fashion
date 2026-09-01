'''
Handles sign ups and user authentication for supabase

have a program that allows sign ups then store those in supabase
'''
from fastapi import APIRouter
from src.schemas.db import supabase
router = APIRouter()

@router.get('/user-data')
async def get_user_data():
    # Fetch the current logged-in user session
    response = supabase.auth.get_user()

    # Extract the unique ID (UUID string)
    user_id = response.user.id
    

    