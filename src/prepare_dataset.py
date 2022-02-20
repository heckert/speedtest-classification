import pandas as pd
import pathlib

from omegaconf import OmegaConf


def get_filters(df: pd.DataFrame, cfg: OmegaConf) -> pd.Series:
    """Get filters from config and create single pandas mask
    """

    trues = pd.Series([True for _ in range(df.shape[0])])
    falses = pd.Series([False for _ in range(df.shape[0])])

    ands = trues.copy()
    for colname, values in cfg.filters.items():
        ors = falses.copy()
        for value in values:
            tmp_cond = df[colname] == value
            ors = ors | tmp_cond

        ands = ands & ors

    return ands


def main():
    # Parse configs
    main_dir = pathlib.Path(__file__).parent.parent.resolve()
    cfg = OmegaConf.load(main_dir / 'config.yaml')

    # Prepare paths
    raw_data_dir = pathlib.Path(cfg.paths.data.raw)
    raw_file = raw_data_dir / cfg.files.raw_dataset
    processed_data_dir = pathlib.Path(cfg.paths.data.processed)
    processed_file = processed_data_dir / cfg.files.processed_dataset

    # Get full list of columns to select
    features = cfg.features.numerics + cfg.features.categories
    target = [cfg.target]

    all_columns = features + target

    # Load the data
    df = pd.read_csv(
        raw_file, 
        dtype={
            'gkz': 'str',
            'implausible': 'str',
            'pinned': 'str',
            }
        )

    # Parse time_utc field and extract hour
    df.time_utc = pd.to_datetime(df.time_utc, errors='coerce')
    df['hour'] = df.time_utc.dt.hour

    # 6 hour bins for daytime category
    df['hour_cat'] = pd.cut(df['hour'],
                            range(0,25, 6))

    # Select relevant columns
    df = df[all_columns]

    # Handle filters defined in config file
    filter_ = get_filters(df, cfg)

    df = df[filter_]

    # Store data as csv
    df.to_csv(processed_file, index=False)


if __name__ == '__main__':
    main()
