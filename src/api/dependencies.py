from fastapi import Request

def get_image_model(request: Request):
    return request.app.state.image_model

def get_stats_model(request: Request):
    return request.app.state.stats_model