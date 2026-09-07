import os
from pathlib import Path
import sys 

# Makes python looks at the parent root directories to find the model
parent = Path(__file__).parent
path = parent / "stats_model.joblib"

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '...'))

from src.schemas.input import ColorParams, CategoryParams, AttributeParams
import torch

# Loops through the file names and stores all colors and categories
def get_color_category():
    colors, cats, attrs = [], [], []

    for color in ColorParams:
        colors.append(color.value)

    for cat in CategoryParams:
        cats.append(cat.value)

    for attr in AttributeParams:
        attrs.append(attr.value)

    return colors, cats, attrs

def image_preds(color_pred, cat_pred, attr_pred):
    '''
    Get the english words for the color and category.
    This is used after prediction
    '''
    # We need this to find the true english labels
    colors, cats, attrs = get_color_category()

    color_arg = torch.argmax(color_pred, dim=1)
    cat_arg = torch.argmax(cat_pred, dim=1)
    attr_arg = torch.argmax(attr_pred, dim=1)

    # Find the color and category based on the gotten index
    color, cat, attr = colors[color_arg], cats[cat_arg], attrs[attr_arg]

    return color, cat, attr