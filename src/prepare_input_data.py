import pandas as pd
import pathlib
import logging

from omegaconf import OmegaConf

import src.utils.preprocessing


def main():
    # Parse configs
    main_dir = pathlib.Path(__file__).parent.parent.resolve()
    cfg = OmegaConf.load(main_dir / 'config.yaml')

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
    numerics = cfg.inputs.numerics
    categories = cfg.inputs.categories
    datetimes = cfg.inputs.datetimes

    all_columns = numerics + categories + \
        datetimes + [cfg.target]

    # Select relevant columns
    df = df[all_columns]

    # Remove cases according to
    # filter defined in config.yaml
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


if __name__ == '__main__':
    main()
