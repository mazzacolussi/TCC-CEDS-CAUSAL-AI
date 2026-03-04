import pandas as pd
import os
import logging

from app.config.settings import PROCESSED_DATA_DIR, INTERIM_DATA_DIR
from app.utils.transformers import BuildFeatures

import warnings
warnings.filterwarnings('ignore')

def main():
    logger = logging.getLogger(__name__)

    logger.info('Starting feature creation.')
    df = pd.read_parquet(os.path.join(PROCESSED_DATA_DIR, 'processed_dataset.parquet'))
    logger.info(f'Dataset shape: {df.shape}.')

    build_features = BuildFeatures()

    logger.info('Starting processing.')
    df = build_features.transform(df)
    logger.info(f'Success! Processing finished. Processed dataset shape: {df.shape}.')


    logger.info(f"Saving interim dataset.")
    os.makedirs(INTERIM_DATA_DIR, exist_ok=True)
    path_output = os.path.join(INTERIM_DATA_DIR, f'interim_dataset.parquet')
    df.to_parquet(path_output, index=False, engine='pyarrow')
    logger.info("Interim dataset was successfully saved.")


if __name__ == '__main__':

    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()