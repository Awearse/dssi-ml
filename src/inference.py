"""
This module encapsulates model inference.
"""

from joblib import dump, load
import pandas as pd
import numpy as np
from src.data_processor import preprocess
from src.model_registry import retrieve
from src.config import appconfig

def get_prediction(**kwargs):
    """
    Get prediction for given data.
        Parameters:
            kwargs: Keyworded argument list containing the data for prediction
        Returns:
            Predicted class in str
    """
    clf, features = retrieve(appconfig['Model']['name'])
    pred_df = pd.DataFrame(kwargs, index=[0])
    pred_df = preprocess(pred_df)
    pred = clf.predict(pred_df[features])
    return pred[0]
