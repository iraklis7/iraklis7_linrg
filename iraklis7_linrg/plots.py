from pathlib import Path

from loguru import logger
from tqdm import tqdm
import matplotlib.pyplot as plt
import typer
import pandas as pd

import iraklis7_linrg.config as config

app = typer.Typer()

def do_hist(data, width, height, show):
    data.hist(figsize=(width,height), edgecolor="black")
    if(show):
        plt.show()

def gen(X_features, X_train, y_train, y_pred, w_norm, b_norm, output_path, show):
    fig,ax=plt.subplots(1, 4, figsize=(12, 3), sharey=True)
    for i in range(len(ax)):
        ax[i].scatter(X_train[:,i],y_train)
        ax[i].set_xlabel(X_features[i])
        if w_norm is not None:
            ax[i].plot(X_train[:,i], X_train[:,i] * w_norm[i] + b_norm, color='red', label = 'model')
        if(y_pred is not None):
            ax[i].scatter(X_train[:,i], y_pred, color='orange', label = 'predict')
    ax[0].set_ylabel("Τιμή (1000'euros)")    
    fig.suptitle("target versus prediction using z-score normalized model")

    plt.savefig(output_path)
    if(show):
        plt.show()


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    features_path: Path = config.PROCESSED_DATA_DIR / config.DATASET_PROC_FEATURES,
    labels_path: Path = config.PROCESSED_DATA_DIR / config.DATASET_PROC_LABELS,
    plot_path: Path = config.FIGURES_DIR / config.FEATURES_PLOT,
    # -----------------------------------------
):
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    logger.info("Generating plot from data...")
    

    features = config.read_data(features_path)
    if features is None:
        raise ValueError("read_data failed - data is None")
    labels = config.read_data(labels_path)
    if labels is None:
        raise ValueError("read_data failed - data is None")
    
    features_sel = features[['Εμβαδόν', 'Όροφος Ρετιρέ', 'Κατάσταση', 'Ασανσέρ από 3ο']]
    gen(list(features), features_sel.to_numpy(), labels, None, None, None, plot_path, False)

    logger.success("Plot generation complete.")
    # -----------------------------------------


if __name__ == "__main__":
    app()
