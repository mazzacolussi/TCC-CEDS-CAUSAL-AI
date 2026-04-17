import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

def preprocess(
        df, 
        treatment, 
        outcome, 
        confundidores
    ):
    df_out = df[confundidores + [treatment, outcome]].dropna().copy()

    le = LabelEncoder()
    df_out['customer_state'] = le.fit_transform(df_out['customer_state'])

    continuous_cols = [c for c in confundidores if c != 'customer_state']
    scaler = StandardScaler()
    df_out[continuous_cols] = scaler.fit_transform(
        df_out[continuous_cols]
    )

    return df_out


def compute_iptw_weights(
        df_model, 
        treatment, 
        confundidores,
        stabilized = True, 
        trim_percentile = 99
    ):

    X = df_model[confundidores].values
    T = df_model[treatment].values

    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X, T)
    
    ps = clf.predict_proba(X)[:, 1]
    ps = np.clip(ps, 0.01, 0.99)
    
    auc = roc_auc_score(T, ps)
    
    p_treated = T.mean()
    
    weights = np.where(
        T == 1, (p_treated / ps),  (1 - p_treated) / (1 - ps)
    )

    if trim_percentile is not None:
        weights = np.clip(weights, None, np.percentile(weights, trim_percentile))
    
    return ps, weights, auc



def compute_smd(
        df_model, 
        treatment, 
        confundidores, 
        weights = None
    ):

    smds = {}
    T = df_model[treatment].values

    for col in confundidores:

        x = df_model[col].values
        
        if weights is not None:
            mean1 = np.average(x[T == 1], weights=weights[T == 1])
            mean0 = np.average(x[T == 0], weights=weights[T == 0])
            var1  = np.average((x[T == 1] - mean1)**2, weights=weights[T == 1])
            var0  = np.average((x[T == 0] - mean0)**2, weights=weights[T == 0])

        else:
            mean1, var1 = x[T == 1].mean(), x[T == 1].var()
            mean0, var0 = x[T == 0].mean(), x[T == 0].var()

        pooled_std = np.sqrt((var1 + var0) / 2)
        smds[col] = abs(mean1 - mean0) / pooled_std if pooled_std > 0 else 0

    return pd.Series(smds)


def ate_iptw(df_model, treatment, outcome, weights):

    T = df_model[treatment].values
    Y = df_model[outcome].values
    
    return (
        np.average(Y[T == 1], weights=weights[T == 1]) -
        np.average(Y[T == 0], weights=weights[T == 0])
    )


def bootstrap_ci(
        df_model, 
        treatment, 
        outcome, 
        confundidores,
        n_bootstrap = 200, 
        alpha = 0.05, 
        trim_percentile = 99
    ):

    ates = []
    n = len(df_model)

    for _ in range(n_bootstrap):

        sample = df_model.sample(n=n, replace=True)
        _, w, _ = compute_iptw_weights(
            sample, 
            treatment, 
            confundidores,
            trim_percentile=trim_percentile
        )
        ates.append(ate_iptw(sample, treatment, outcome, w))
    
    lower = np.percentile(ates, 100 * alpha / 2)
    upper = np.percentile(ates, 100 * (1 - alpha / 2))
    
    return lower, upper, np.array(ates)