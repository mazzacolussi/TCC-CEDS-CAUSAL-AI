import pandas as pd
from app.config.settings import date_cols
from sklearn.base import BaseEstimator, TransformerMixin


class BuildFeatures(BaseEstimator, TransformerMixin):

    def __init__(self):
        "Class used to process input data, creating new features from existing variables."
        super().__init__()

    def __repr__(self):
        return "Object intended for feature creation."
    
    def fit(self, X):
        """
        Fit method required by scikit-learn API.

        Parameters
        ----------
        X : pd.DataFrame
            Input dataset.
        y : optional
            Ignored.

        Returns
        -------
        self
        """
        return self
    
    def transform(self, X):
        """
        Apply feature engineering transformations.

        Parameters
        ----------
        X : pd.DataFrame
            Input dataset.

        Returns
        -------
        pd.DataFrame
            Transformed dataset with engineered features.
        """
        X = self.build_features(X)
        return X

    def build_features(self, X):
        """
        Create new features from raw variables.

        Features created:
        - is_delayed: delivery delay indicator
        - purchase_weekday: weekday of purchase
        - purchase_month: month of purchase

        Parameters
        ----------
        X : pd.DataFrame
            Input dataset.

        Returns
        -------
        pd.DataFrame
            Dataset with additional engineered features.
        """
        def apply_features(X: pd.DataFrame) -> pd.DataFrame:

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
