import pandas as pd
import pathlib

from omegaconf import OmegaConf

# Parse configs
cfg = OmegaConf.load('./config.yaml')

# Prepare paths
raw_data_path = pathlib.Path(cfg.paths.data.raw)
processed_data_path = pathlib.Path(cfg.paths.data.processed)
raw_filepath = raw_data_path / cfg.files.raw_dataset
processed_filepath = processed_data_path / cfg.files.processed_dataset

# Get full list of columns to select
native_features = cfg.features.native
custom_features = [value for value in cfg.features.custom.values()]
target = list(cfg.target.keys())

all_columns = native_features + custom_features + target

def main():
    # Load the data
    df = pd.read_csv(raw_filepath)

    # Parse time_utc field and extract hour
    df.time_utc = pd.to_datetime(df.time_utc)
    df[cfg.features.custom.datetime_hour] = df.time_utc.dt.hour

    # 6 hour bins for daytime category
    df[cfg.features.custom.datetime_hour_category] = pd.cut(
        df[cfg.features.custom.datetime_hour] ,range(0,25, 6)
        )

    # Select relevant columns
    df = df[all_columns]

    # Store data as csv
    df.to_csv(processed_filepath, index=False)


if __name__ == '__main__':
    main()