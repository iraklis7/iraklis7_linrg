from pathlib import Path

from loguru import logger
import typer

import iraklis7_linrg.config as config

app = typer.Typer()


def transform_data(data, sqm_limit):
    result = data.copy()
    # Drop rows where square meters are more then 300
    result = result.query(f"Εμβαδόν < {sqm_limit}")
    # Drop rows where floor is unspecified
    result["Όροφος"].fillna("NULL")
    result = result.query('Όροφος != "NULL"')
    # Set rows where view is unspecifed to 'No View'
    result["Θέα"].fillna("0")
    # Set rows where elevator is unspecified to 'No Elevator'
    result["Ασανσέρ"].fillna("0")
    # Remove thousands from price and convert to numeric
    result["Τιμή"] /= 1000
    result["Αρχική Τιμή"] /= 1000

    return result


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    input_path: Path = config.RAW_DATA_DIR / config.DATASET,
    output_path: Path = config.PROCESSED_DATA_DIR / config.DATASET_PROC,
    # ----------------------------------------------
):
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    logger.info("Processing dataset...")
    data = config.read_data(input_path)
    if data is None:
        raise ValueError("read_data failed - data is None")

    # Inspect data
    logger.debug(data.sample(5))
    logger.debug(data.info())
    logger.debug(data.isna().sum())
    logger.debug(data.describe())
    logger.debug(data.select_dtypes("number"))

    # Tranform data
    logger.info("Transforming data ...")
    data_tr = transform_data(data, 300)

    logger.info("Writing features and labels to file ...")
    config.write_data(output_path, data_tr)

    logger.success("Processing dataset complete.")
    # -----------------------------------------


if __name__ == "__main__":
    app()
