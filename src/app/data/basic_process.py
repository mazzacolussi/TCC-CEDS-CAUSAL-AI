import os
import pandas as pd
import logging

from app.config.settings import RAW_DATA_DIR, PROCESSED_DATA_DIR
from app.data.builders import (
    build_customers_dataset,
    build_order_items_dataset,
    build_payments_dataset
)

import warnings
warnings.filterwarnings('ignore')

def main():

    logger = logging.getLogger(__name__)
    logger.info("Starting dataset consolidation and processing.")

    datasets = {}

    for file in os.listdir(RAW_DATA_DIR):
        if file.endswith(".csv"):
            file_path = os.path.join(RAW_DATA_DIR, file)
            datasets[file.replace(".csv", "")] = pd.read_csv(file_path)
        elif file.endswith(".parquet"):
            file_path = os.path.join(RAW_DATA_DIR, file)
            datasets[file.replace(".parquet", "")] = pd.read_parquet(file_path)

    customers = build_customers_dataset(datasets, logger)
    order_items = build_order_items_dataset(datasets, logger)
    payments = build_payments_dataset(datasets, logger)

    logger.info("Merging processed tables into final dataset.")

    df = datasets["olist_orders_dataset"].copy()
    before_rows = df.shape[0]

    df = df.merge(customers, on = "customer_id", how = "left")
    df = df.merge(order_items, on = "order_id", how = "left")
    df = df.merge(payments, on = "order_id", how = "left")

    logger.info("Dataset created (rows = %d -> %d, cols = %d).", before_rows, df.shape[0], df.shape[1])

    before_rows = df.shape[0]
    df = df[df["order_status"].isin(["delivered", "canceled", "unavailable"])]

    logger.info("Filtering samples only with 'delivered', 'canceled', or 'unavailable' order status (rows = %d -> %d, cols = %d).", before_rows, df.shape[0], df.shape[1])

    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    output_path = os.path.join(PROCESSED_DATA_DIR, "processed_dataset.parquet")

    logger.info("Saving processed dataset to: %s", output_path)
    df.to_parquet(output_path)

    logger.info("Dataset processing completed successfully.")


if __name__ == '__main__':

    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
