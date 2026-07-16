import pytest
from PIL import Image
import asyncio
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.api.dependencies.depend import get_image_model
from src.core.lifespan import lifespan
from fastapi import FastAPI
app = FastAPI(lifespan=lifespan)

@pytest.mark.asyncio
async def test_image_output():
    color, cloth_type = await get_image_model('tests/test_clothing.jpg')
    assert color.name == 'red'
    assert cloth_type.name == 'dress'


'''
def test_get_user_params():
    assert get_user_params()

def test_initialize_preds():
    assert initialize_preds()

def test_query_rag_system():
    assert query_rag_system()
'''
