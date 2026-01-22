from pathlib import Path
from loguru import logger
import typer
import pandas as pd
import iraklis7_linrg.config as config
import iraklis7_linrg.plots as plots

app = typer.Typer()

def is_top_floor(floor, total_floors):
    if(floor is not None and total_floors is not None):
        return int(floor == total_floors and floor > 3)
    else:
        return 0
    
def calc_age(year_built):
    from datetime import datetime
    age = datetime.now().year - year_built
    if age < 10:
        return 1
    elif 10 <= age < 40:
        return 0
    else:
        return -1

def get_condition(year_constructed, year_renovated):
    from datetime import datetime
    age = datetime.now().year - year_constructed

    if(age < 10):
        result = 1
    elif(age>11 and age <30):
        result = 0
    else:
        result = -1
    if(year_renovated is not None):
        ren_age = datetime.now().year - year_renovated
        if(ren_age < 10):
            result = 1
    return result

def elevator_from_third(floor, elevator):
    return int(floor >= 3 and elevator == 1)

def normalize_data(tr_data):
    nparr = tr_data.to_numpy()
    nparr_norm= config.scaler.fit_transform(nparr)
    df_normalized = pd.DataFrame(nparr_norm, columns = tr_data.columns)
    
    return df_normalized


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    input_path: Path = config.PROCESSED_DATA_DIR / config.DATASET_PROC,
    features_path: Path = config.PROCESSED_DATA_DIR / config.DATASET_PROC_FEATURES,
    labels_path: Path = config.PROCESSED_DATA_DIR / config.DATASET_PROC_LABELS,
    plot_path: Path = config.FIGURES_DIR / config.CLEANING_PLOT,
    # -----------------------------------------
):
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    logger.info("Reading transformed dataset...")
    data_tr = config.read_data(input_path)
    if data_tr is None:
        raise ValueError("read_data failed - data is None")

    #Feature Engineering
    logger.info("Feature engineering ...")
    # Return true if floor is the last floor on the building
    data_tr['Όροφος Ρετιρέ'] = data_tr.apply(lambda row: is_top_floor(row['Όροφος'], row['Σύνολο ορόφων']), axis=1) 
    data_tr[['Όροφος Ρετιρέ', 'Όροφος', 'Σύνολο ορόφων']].head()

    # Returns a label based on the age of the property
    data_tr['Ηλικία'] = data_tr['Έτος κατασκευής'].apply(calc_age)
    data_tr[['Ηλικία', 'Έτος κατασκευής']].head()

    # Returns a label based on the condition of the property
    data_tr['Κατάσταση'] = data_tr.apply(lambda row: get_condition(row['Έτος κατασκευής'], row['Έτος ανακαίνισης']), axis=1)
    data_tr[['Κατάσταση', 'Έτος κατασκευής', 'Έτος ανακαίνισης']].head()

    # Return true if elevator exists and floor is third or higher
    data_tr['Ασανσέρ από 3ο'] = data_tr.apply(lambda row: elevator_from_third(row['Όροφος'], row['Ασανσέρ']), axis=1) 
    data_tr[['Ασανσέρ από 3ο', 'Όροφος', 'Ασανσέρ']].head()

    # Normalize data
    logger.info("Normalizing data (z-score)...")
    features = data_tr.drop(['Τιμή', 'Αρχική Τιμή'], axis=1)
    features_norm = normalize_data(features)
    labels = data_tr['Τιμή']
        
    # Write normalized data to file
    logger.info("Writing features and labels to file ...")
    config.write_data(features_path,features_norm)
    config.write_data(labels_path,labels)   

    # Generate plot    
    features_sel = features_norm[['Εμβαδόν', 'Θέα', 'Όροφος Ρετιρέ', 'Κατάσταση', 'Ασανσέρ από 3ο']]
    plots.gen(list(features_sel), features_sel.to_numpy(), labels, None, None, None, show=False, output_path=plot_path)

    logger.success("Features generation complete.")
    # -----------------------------------------


if __name__ == "__main__":
    app()
