import pandas as pd
import logging
from typing import Dict

def build_customers_dataset(
        datasets: Dict[str, pd.DataFrame], 
        logger: logging.Logger
) -> pd.DataFrame:
    """Create customers dataset enriched with geolocation coordinates by zip prefix."""
    logger.info('Building customers dataset with geolocation aggregation.')

    geo_agg = (
        datasets['olist_geolocation_dataset']
        .groupby('geolocation_zip_code_prefix')
        .agg({'geolocation_lat': 'mean', 'geolocation_lng': 'mean'})
        .reset_index()
    )

    customers = datasets['olist_customers_dataset'].merge(
        geo_agg,
        left_on = 'customer_zip_code_prefix',
        right_on = 'geolocation_zip_code_prefix',
        how = 'left',
    )

    customers.drop(columns=['geolocation_zip_code_prefix'], inplace = True)

    logger.info(
        'Customers dataset ready (rows = %d, cols = %d).', 
        customers.shape[0], customers.shape[1]
    )

    return customers



def build_order_items_dataset(
    datasets: Dict[str, pd.DataFrame], 
    logger: logging.Logger
) -> pd.DataFrame:
    """Create order-level aggregates from order_items joined with product + category translation."""
    logger.info('Building order_items aggregates (order-level).')

    order_items = datasets['olist_order_items_dataset'].merge(
        datasets['olist_products_dataset'],
        how = 'left',
        on = 'product_id',
    )

    order_items = order_items.merge(
        datasets['product_category_name_translation'],
        how = 'left',
        on = 'product_category_name',
    )

    logger.info(
        'Joined order_items with products + category translation (rows = %d).', 
        order_items.shape[0]
    )

    order_items_agg = (
        order_items.groupby('order_id', as_index = False)
        .agg(
            total_price = ('price', 'sum'),
            avg_price = ('price', 'mean'),
            total_freight = ('freight_value', 'sum'),
            avg_freight = ('freight_value', 'mean'),
            n_items_missing_info = ('product_description_lenght', lambda x: x.isna().sum()),
            n_items = ('product_id', 'count'),
            n_item_distinct_categ = ('product_category_name_english', 'count'),
            avg_weight = ('product_weight_g', 'mean'),
            avg_length = ('product_length_cm', 'mean'),
            avg_height = ('product_height_cm', 'mean'),
            avg_width = ('product_width_cm', 'mean'),
        )
    )

    logger.info(
        'Order items aggregates ready (rows = %d, cols = %d).', 
        order_items_agg.shape[0], order_items_agg.shape[1]
    )

    return order_items_agg



def build_payments_dataset(
    datasets: Dict[str, pd.DataFrame], 
    logger: logging.Logger
) -> pd.DataFrame:
    """Create order-level payment aggregates."""
    logger.info('Building payments aggregates (order-level).')

    payments_agg = (
        datasets['olist_order_payments_dataset']
        .groupby('order_id')
        .agg(
            total_payment = ('payment_value', 'sum'),
            avg_payment = ('payment_value', 'mean'),
            max_installments = ('payment_installments', 'max'),
            n_payments_type = ('payment_type', 'count'),
        )
        .reset_index()
    )

    logger.info(
        'Payments aggregates ready (rows = %d, cols = %d).', 
        payments_agg.shape[0], payments_agg.shape[1]
    )

    return payments_agg