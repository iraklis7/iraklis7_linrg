from pathlib import Path
from loguru import logger
from tqdm import tqdm
import typer
import pandas as pd
import joblib
import sklearn.metrics as metrics
import iraklis7_linrg.config as config
import iraklis7_linrg.plots as plots

app = typer.Typer()

def score_model(labels, predictions):
    logger.info(f"MAE: {metrics.mean_absolute_error(labels, predictions)}")
    logger.info(f"MSE: {metrics.mean_squared_error(labels, predictions)}")
    logger.info(f"MSLE: {metrics.mean_squared_log_error(labels, predictions)}")
    logger.info(f"MAPE: {metrics.mean_absolute_percentage_error(labels, predictions)}")
    logger.info(f"MEAE: {metrics.median_absolute_error(labels, predictions)}")
    logger.info(f"MAXE: {metrics.max_error(labels, predictions)}")
    logger.info(f"EVS: {metrics.explained_variance_score(labels, predictions)}")


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    features_path: Path = config.PROCESSED_DATA_DIR / config.DATASET_PROC_FEATURES,
    model_path: Path = config.MODELS_DIR / config.DATASET_MODEL,
    predictions_path: Path = config.PROCESSED_DATA_DIR / config.DATASET_PREDICTIONS,
    labels_path: Path = config.PROCESSED_DATA_DIR / config.DATASET_PROC_LABELS,
    plot_path: Path = config.FIGURES_DIR / config.TRAINING_PLOT,
    # -----------------------------------------
):
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    logger.info("Loading data from: " + str(features_path))
    try:
        features = pd.DataFrame(config.read_data(features_path))
    except Exception as e:
        logger.exception("Unable to load data: " + str(e))

    logger.info("Loading labels from: " + str(labels_path))
    try:
        labels = config.read_data(labels_path)
    except Exception as e:
        logger.exception("Unable to load data: " + str(e))

    logger.info("Loading modeL: " + str(model_path))
    try:
        linreg_model = joblib.load(model_path)
    except Exception as e:
        logger.exception("Unable to load model: " + str(e))
        
    logger.info("Performing inference")
    X_features = ['Εμβαδόν','Όροφος Ρετιρέ', 'Κατάσταση', 'Ασανσέρ από 3ο']
    features = features[X_features]

    # Make predictions
    predictions = linreg_model.predict(features)
    score_model(labels, predictions)

    # Plot labels, predictions, and linear regression lines
    b_norm = linreg_model.intercept_
    w_norm = linreg_model.coef_
    dataN = features.to_numpy()
    plots.gen(list(dataN), dataN, labels, predictions, w_norm, b_norm, plot_path, False)



    logger.info(f"Samnple predictions on training set:\n{predictions[:4]}" )
    logger.info("Writing predictions to " + str(predictions_path))
    try:
        config.write_data(predictions_path, pd.DataFrame(predictions, columns=['ΕκΤιμή']))
    except Exception as e:
        logger.exception("Unable to write predictions: " + str(e))

    logger.success("Inference complete.")
    # -----------------------------------------


if __name__ == "__main__":
    app()
