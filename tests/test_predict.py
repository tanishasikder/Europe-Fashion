from src.api.predict import get_user_params, image_output, initialize_preds, query_rag_system
import pytest

def test_get_user_params():
    assert get_user_params()

@pytest.mark.asyncio
def test_image_output():
    assert image_output()

def test_initialize_preds():
    assert initialize_preds()

def test_query_rag_system():
    assert query_rag_system()