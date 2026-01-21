import pytest
from pathlib import Path
import numpy as np
from loguru import logger
from tqdm import tqdm
from time import sleep
import pandas as pd
import iraklis7_linrg.config as config
import iraklis7_linrg.dataset as dataset
import iraklis7_linrg.modeling.train as train
import iraklis7_linrg.modeling.predict as predict
import iraklis7_linrg.features as featuresd


def test_re_model():
    features_path: Path = config.PROCESSED_DATA_DIR / config.DATASET_PROC_FEATURES
    labels_path: Path = config.PROCESSED_DATA_DIR / config.DATASET_PROC_LABELS
    predictions_path: Path = config.PROCESSED_DATA_DIR / config.DATASET_PREDICTIONS

    # Mean absolute error (MAE) needs to over this threshold for the test to pass
    accuracy_threshold = 65

    logger.info("Building environment")
    dataset.main()
    featuresd.main()
    train.main()
    predict.main()

    logger.info("Loading data")
    features = pd.DataFrame(config.read_data(features_path))
    if features is None:
        raise ValueError("read_data failed - data is None")
    labels = config.read_data(labels_path)
    if labels is None:
        raise ValueError("read_data failed - data is None")
    predictions = pd.DataFrame(config.read_data(predictions_path))
    if predictions is None:
        raise ValueError("read_data failed - data is None")

    assert labels.shape == predictions.shape, "Shape of labels and predictions does not match"

    # Iinitialize variables
    mae = 0
    num_predictions = len(predictions)

    # Compare every prediction to its label
    for i in tqdm(range(num_predictions)):
        actual_val = int(labels['Τιμή'].iloc[i])
        predicted_val = int(predictions['ΕκΤιμή'].iloc[i])
        abs_diff = abs(actual_val - predicted_val)
        error = (abs_diff / actual_val) * 100
        accuracy = 100.00 - error
        mae += abs_diff
        #print(f"I: {i} ACT: {actual_val} PRE: {predicted_val} ACC: {accuracy}")
        sleep(0.01)  # to make tqdm visible
    
    # Derive MAE
    mae = mae / num_predictions
    assert(mae > accuracy_threshold), f"MAE below accurcacy threshold of {accuracy_threshold}%"
    logger.info(f"Mean Absolute Error (MAE): {mae}")