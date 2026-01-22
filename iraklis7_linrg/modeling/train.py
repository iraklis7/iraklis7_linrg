from pathlib import Path
from loguru import logger
import typer
from sklearn.linear_model import SGDRegressor
from joblib import dump
import pandas as pd
import iraklis7_linrg.config as config
import iraklis7_linrg.modeling.train as train
from numpy import ravel

app = typer.Typer()

def fit_model(sgdr, features, labels):
    sgdr.fit(features, labels)
    logger.info(sgdr)
    logger.info(f"number of iterations completed: {sgdr.n_iter_}, number of weight updates: {sgdr.t_}")


def save_model(sgdr, output_path):
    try:
        dump(sgdr, output_path)
        logger.debug("Model saved to: " + str(output_path))
    except Exception as e:
        logger.exception("Unable to save model: " + str(e))

@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    features_path: Path = config.PROCESSED_DATA_DIR / config.DATASET_PROC_FEATURES,
    labels_path: Path = config.PROCESSED_DATA_DIR / config.DATASET_PROC_LABELS,
    model_path: Path = config.MODELS_DIR / config.DATASET_MODEL,
    # -----------------------------------------
):
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    logger.info("Training model...")

    features = config.read_data(features_path)
    if features is None:
        raise ValueError("read_data failed - data is None")
    labels = config.read_data(labels_path)
    if labels is None:
        raise ValueError("read_data failed - data is None")

    X_features = ['Εμβαδόν','Όροφος Ρετιρέ', 'Κατάσταση', 'Ασανσέρ από 3ο']
    features = features[X_features]

    # Train model
    logger.info("Fitting training data")
    sgdr = SGDRegressor(max_iter=1000)
    train.fit_model(sgdr, features, ravel(labels))

    # Review parameters
    b_norm = sgdr.intercept_
    w_norm = sgdr.coef_
    logger.info(f"model parameters: w: {w_norm}, b:{b_norm}")
    
    # Save the model
    logger.info("Saving model")
    save_model(sgdr, model_path)

    logger.success("Modeling training complete.")
    # -----------------------------------------


if __name__ == "__main__":
    app()
