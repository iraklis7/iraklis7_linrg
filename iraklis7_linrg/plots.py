from pathlib import Path

from loguru import logger
import matplotlib.pyplot as plt
import seaborn as sns
import typer

import iraklis7_linrg.config as config

app = typer.Typer()


def do_heatmap(data, show=False, output_path=None):
    plt.title("Correlation Heatmap")
    sns.heatmap(data.corr(numeric_only=True))
    if show:
        plt.show()
    if output_path is not None:
        try:
            plt.savefig(output_path)
        except Exception as e:
            logger.exception("Unable to save plot: " + str(e))


def do_hist(data, width, height, show=False, output_path=None):
    data.hist(figsize=(width, height), edgecolor="black")
    if show:
        plt.show()
    if output_path is not None:
        try:
            plt.savefig(output_path)
        except Exception as e:
            logger.exception("Unable to save plot: " + str(e))


def gen(X_features, X_train, y_train, y_pred, w_norm, b_norm, show=False, output_path=None):
    fig, ax = plt.subplots(1, 4, figsize=(12, 3), sharey=True)
    for i in range(len(ax)):
        ax[i].scatter(X_train[:, i], y_train)
        ax[i].set_xlabel(X_features[i])
        if w_norm is not None:
            ax[i].plot(
                X_train[:, i], X_train[:, i] * w_norm[i] + b_norm, color="red", label="model"
            )
        if y_pred is not None:
            ax[i].scatter(X_train[:, i], y_pred, color="orange", label="predict")
    ax[0].set_ylabel("Τιμή (1000'euros)")
    fig.suptitle("target versus prediction using z-score normalized model")

    if show:
        plt.show()
    if output_path is not None:
        try:
            plt.savefig(output_path)
        except Exception as e:
            logger.exception("Unable to save plot: " + str(e))


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    norm_path: Path = config.PROCESSED_DATA_DIR / config.DATASET_PROC,
    features_path: Path = config.PROCESSED_DATA_DIR / config.DATASET_PROC_FEATURES,
    labels_path: Path = config.PROCESSED_DATA_DIR / config.DATASET_PROC_LABELS,
    plot_path: Path = config.FIGURES_DIR / config.FEATURES_PLOT,
    heatmap_path: Path = config.FIGURES_DIR / config.HEATMAP_PLOT,
    # -----------------------------------------
):
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    logger.info("Generating plot from data...")

    data_tr = config.read_data(norm_path)
    if data_tr is None:
        raise ValueError("read_data failed - data is None")
    features = config.read_data(features_path)
    if features is None:
        raise ValueError("read_data failed - data is None")
    labels = config.read_data(labels_path)
    if labels is None:
        raise ValueError("read_data failed - data is None")

    features_sel = features[["Εμβαδόν", "Όροφος Ρετιρέ", "Κατάσταση", "Ασανσέρ από 3ο"]]
    gen(
        list(features),
        features_sel.to_numpy(),
        labels,
        None,
        None,
        None,
        show=False,
        output_path=plot_path,
    )

    do_heatmap(data_tr, show=False, output_path=heatmap_path)

    logger.success("Plot generation complete.")
    # -----------------------------------------


if __name__ == "__main__":
    app()
