# Data

Download VeReMi Extension from https://data.mendeley.com/datasets/k62n4z9gdz and extract into `data/raw/`, keeping the scenario folder structure.

## Dataset note
The Mendeley release is a single file, mixalldata_clean.tar.xz -> data/raw/mix/mixalldata_clean.csv (one row per message, no receiver column, class 0 = genuine, 1-19 = attack type). Build the parquet with: python -m src.data_loader --mixall --all

