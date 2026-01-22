import sys
from pathlib import Path
from dotenv import load_dotenv
from loguru import logger
import pandas as pd
from sklearn.preprocessing import StandardScaler


# Load environment variables from .env file if it exists
load_dotenv()

extra_path = "/Users/iraklis/Public/iraklis7_linrg"
if extra_path not in sys.path:
    sys.path.append(extra_path)

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODELS_DIR = PROJ_ROOT / "models"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

DATASET = "listings_data.csv"
DATASET_PROC = DATASET.replace(".csv", "_norm.csv")
DATASET_PROC_FEATURES = DATASET.replace(".csv", "_features.csv")
DATASET_PROC_LABELS = DATASET.replace(".csv", "_labels.csv")
DATASET_MODEL = DATASET.replace(".csv", "_model.joblib")
DATASET_PREDICTIONS = DATASET.replace(".csv", "_predictions.csv")

INITIAL_PLOT = DATASET.replace(".csv", "_initial_plot.png")
HEATMAP_PLOT = DATASET.replace(".csv", "_heatmap_plot.png")
CLEANING_PLOT = DATASET.replace(".csv", "_cleaning_plot.png")
TRAINING_PLOT = DATASET.replace(".csv", "_training_plot.png") 
FEATURES_PLOT = DATASET.replace(".csv", "_features.png") 

scaler = StandardScaler()


def read_data(input_path):
    try:
        return pd.DataFrame(pd.read_csv(input_path))
    except Exception as e:
        logger.exception("Unable to load data: " + str(e))
        raise

def write_data(output_path, data):
    try:
        data.to_csv(output_path, index=False)
    except Exception as e:
        logger.exception("Unable to write data: " + str(e))


# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass
