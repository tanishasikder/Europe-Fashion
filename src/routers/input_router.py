from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi import Request, status
from pydantic import ValidationError
from fastapi import Depends
from src.main import limiter 
from src.core.celery_app import process_img
from src.schemas.jwt import verify_jwt

router = APIRouter(prefix='preds')

# Basic health check to ensure server is functioning
@router.get("/health")
def root():
    return {"status" : "OK"}

@router.post("/upload") # Use with schemas/input function get_upload
@limiter.limit('3/minute') # How much we limit
async def upload(
        request : Request, # Need this or limiter will not work
        payload: dict = Depends(verify_jwt), # Get this from schemas/jwt.py it verifies the user using jwt
        file: UploadFile = File(...)
    ):
    try:
        user_id = payload['sub']
        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail='Authorization is missing',
                headers={'WWW-Authenticate': 'Bearer'}
            )
        contents = await file.read()
        color, cat, attr = process_img.delay(contents) # Process this image
        return color, cat, attr # Returns the prediction of what the image is
    except ValidationError as e:
        raise HTTPException(status_code=422, detail=e.errors())

# Use with services/model_service/predict function query_rag_system
@router.get("/query")
@limiter.limit('3/minute')
async def get_query_rag(request : Request, query: str):
    try:
        return query
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Gets the model predictions for color and clothing type
@router.post('/image_predict') # Somehow combine with the celery function below figure it out
async def get_image_preds(request: Request):
    try:
        image_model = request.app.state.image_model
        return image_model
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


