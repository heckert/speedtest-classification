import pandas as pd
import pathlib
import logging

from omegaconf import DictConfig

import src.utils.preprocessing


def main(cfg: DictConfig):
    # Prepare paths
    raw_data_dir = pathlib.Path(cfg.paths.data.raw)
    raw_file = raw_data_dir / cfg.files.raw_dataset
    processed_data_dir = pathlib.Path(cfg.paths.data.processed)
    processed_file = processed_data_dir / cfg.files.processed_dataset

    # Pass dtypes for selected fields
    # to avoid mixed dtypes warning
    str_types = {
        'gkz': 'str',
        'implausible': 'str',
        'pinned': 'str',
    }

    df = pd.read_csv(raw_file, dtype=str_types)

    # Transform kbit/s to mbit/s
    df['download_mbit'] = df.download_kbit / 1e3
    df['upload_mbit'] = df.upload_kbit / 1e3

    # Get full list of columns to select
    all_columns = cfg.inputs.numerics + cfg.inputs.categories + \
        cfg.inputs.datetimes + [cfg.target]

    # Select relevant columns
    df = df[all_columns]

    # Filter cases as defined in config.yaml
    if cfg.filters is not None:
        df = src.utils.preprocessing.apply_filters(df=df, filters=cfg.filters)
    else:
        logging.debug('No filters applied')

    logging.debug(
        'Number of rows/cols in processed dataframe: {:,} / {}'
        .format(*df.shape)
    )

    # Store data as csv
    df.to_csv(processed_file, index=False)
