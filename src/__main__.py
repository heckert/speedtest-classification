import argparse
import logging
import pathlib
import sys

from omegaconf import OmegaConf

import src.prepare_input_data

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    '--prepareInputData',
    action='store_true',
    help='Select input columns and relevant rows from raw dataset'
)

args = parser.parse_args()

# Parse config
main_dir = pathlib.Path(__file__).parent.parent.resolve()
cfg = OmegaConf.load(main_dir / 'config.yaml')

# Set up logging
logging.basicConfig(stream=sys.stdout,
                    format=cfg.logging.fmt,
                    datefmt=cfg.logging.date_fmt,
                    level=cfg.logging.level)

# Prepare input data
if args.prepareInputData:
    logging.info('Preparing input data...')
    src.prepare_input_data.main()
    logging.info('Processed input data stored at '
                 f'{cfg.paths.data.processed}/{cfg.files.processed_dataset}')
