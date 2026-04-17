import pandas as pd
from app.config.settings import date_cols
from sklearn.base import BaseEstimator, TransformerMixin


class BuildFeatures(BaseEstimator, TransformerMixin):

    def __init__(self):
        "Classe usada para processar os dados de entrada, criando novas features a partir das variáveis existentes"
        super().__init__()

    def __repr__(self):
        return "Objeto destinado para criar features"
    
    def fit(self, X):
        return self
    
    def transform(self, X):
        X = self.build_features(X)
        return X

    def build_features(self, X):
        def apply_features(X: pd.DataFrame) -> pd.DataFrame:

            # X["installment_value"] = X["total_payment"] / X["max_installments"]

            for col in date_cols:
                X[col] = pd.to_datetime(X[col], errors="coerce")

            # Atraso na entrega
            X["is_delayed"] = (
                (X["order_delivered_customer_date"] - X["order_estimated_delivery_date"]).dt.days > 0
            ).astype(int)

            # Features temporais
            X["purchase_weekday"] = X["order_purchase_timestamp"].dt.weekday
            X["purchase_month"] = X["order_purchase_timestamp"].dt.month

            return X
    
        return apply_features(X)
