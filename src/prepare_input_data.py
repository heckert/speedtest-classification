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

    # Get full list of columns to select
    all_columns = cfg.inputs + [cfg.target]

    # Select relevant columns
    df = df[all_columns]

    # Select only relevant technologies
    filter_ = src.utils.preprocessing.parse_filters(df, cfg.filters)

    # Log how many cases are filtered
    counts = filter_.value_counts()
    n_not_selected = counts.loc[False]

    logging.debug(
        'Filtered out {negative:,} of {total:,} ({percentage}%)'.format(
            negative=n_not_selected, total=len(df),
            percentage=round(n_not_selected/len(df)*100, 1)
        )
    )

    df = df[filter_]
    logging.debug(
        'Number of rows/cols in processed dataframe: {:,} / {}'
        .format(*df.shape)
    )

    # Store data as csv
    df.to_csv(processed_file, index=False)


if __name__ == '__main__':
    main()
