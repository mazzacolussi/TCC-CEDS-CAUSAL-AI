import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


def preprocess(
    df: pd.DataFrame,
    confounders: list[str],
    treatment: str,
    outcome: str,
) -> pd.DataFrame:
    """
    Preprocess the dataset for causal modeling.

    Selects the required columns (confounders, treatment, and outcome),
    removes rows with missing values, and standardizes numeric variables
    using StandardScaler.

    Parameters
    ----------
    df : pd.DataFrame
        Original dataset.
    confounders : list[str]
        List of observed confounding variables.
    treatment : str
        Name of the treatment variable.
    outcome : str
        Name of the outcome variable.

    Returns
    -------
    pd.DataFrame
        Cleaned and scaled dataset ready for causal estimation.
    """
    
    cols = confounders + [treatment, outcome]
    df_out = df[cols].dropna().copy()
    cols_num = df_out[confounders].select_dtypes(include='number').columns.tolist()

    if cols_num:
        scaler = StandardScaler()
        df_out[cols_num] = scaler.fit_transform(df_out[cols_num])

    return df_out


def build_gml(confounders, treatment, outcome):

    nodes = list(set(confounders + [treatment, outcome]))

    edges = (
        [(c, treatment) for c in confounders] +
        [(c, outcome) for c in confounders] +
        [(treatment, outcome)]
    )
    """
    Build a DAG representation in GML format.

    Assumes that all confounders affect both treatment and outcome,
    and includes a direct causal edge from treatment to outcome.

    Parameters
    ----------
    confounders : list[str]
        List of observed confounders.
    treatment : str
        Name of the treatment variable.
    outcome : str
        Name of the outcome variable.

    Returns
    -------
    str
        DAG encoded as a GML string.
    """
    lines = ['graph [directed 1']

    for n in nodes:
        lines.append(f'node [id "{n}" label "{n}"]')

    for u, v in edges:
        lines.append(f'edge [source "{u}" target "{v}"]')

    lines.append(']')

    return '\n'.join(lines)


def compute_iptw_weights(
        df: pd.DataFrame, 
        confounders: list[str],
        treatment: str, 
        stabilized: bool = True, 
        trim_percentile: int = 99
    ):
    """
    Estimate propensity scores and compute IPTW weights.

    Fits a logistic regression model for treatment assignment,
    computes stabilized inverse probability weights, and optionally
    trims extreme values.

    Parameters
    ----------
    df : pd.DataFrame
        Modeling dataset.
    confounders : list[str]
        Variables used in the propensity score model.
    treatment : str
        Binary treatment variable.
    stabilized : bool, default=True
        Whether to compute stabilized weights.
    trim_percentile : int, default=99
        Upper percentile used to trim extreme weights.

    Returns
    -------
    tuple
        (fitted_model, propensity_scores, weights, auc_roc)
    """

    X = df[confounders].values
    T = df[treatment].values

    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X, T)
    
    ps = clf.predict_proba(X)[:, 1]
    ps = np.clip(ps, 0.01, 0.99)
    
    auc_score = roc_auc_score(T, ps)
    p_treated = T.mean()
    
    weights = np.where(
        T == 1, (p_treated / ps),  (1 - p_treated) / (1 - ps)
    )

    if trim_percentile is not None:
        weights = np.clip(weights, None, np.percentile(weights, trim_percentile))
    
    return clf, ps, weights, auc_score


def compute_smd(
        df_model, 
        confounders,
        treatment, 
        weights = None
    ):
    """
    Compute Standardized Mean Differences (SMD).

    Measures covariate imbalance between treated and control groups
    before or after weighting.

    Parameters
    ----------
    df_model : pd.DataFrame
        Analysis dataset.
    confounders : list[str]
        List of confounding variables.
    treatment : str
        Treatment variable.
    weights : array-like, optional
        IPTW weights. If None, computes unweighted SMD.

    Returns
    -------
    pd.Series
        SMD value for each confounder.
    """

    smds = {}
    T = df_model[treatment].values

    for col in confounders:

        x = df_model[col].values
        
        if weights is not None:
            mean1 = np.average(x[T == 1], weights=weights[T == 1])
            mean0 = np.average(x[T == 0], weights=weights[T == 0])
            var1 = np.average((x[T == 1] - mean1)**2, weights=weights[T == 1])
            var0 = np.average((x[T == 0] - mean0)**2, weights=weights[T == 0])

        else:
            mean1, var1 = x[T == 1].mean(), x[T == 1].var()
            mean0, var0 = x[T == 0].mean(), x[T == 0].var()

        pooled_std = np.sqrt((var1 + var0) / 2)
        smds[col] = abs(mean1 - mean0) / pooled_std if pooled_std > 0 else 0

    return pd.Series(smds)


def ate_iptw(df, treatment, outcome, weights):
    """
    Estimate the Average Treatment Effect (ATE) using IPTW.

    Computes the weighted mean difference in outcome
    between treated and control groups.

    Parameters
    ----------
    df : pd.DataFrame
        Analysis dataset.
    treatment : str
        Binary treatment variable.
    outcome : str
        Outcome variable.
    weights : array-like
        IPTW weights.

    Returns
    -------
    float
        Estimated ATE.
    """

    T = df[treatment].values
    Y = df[outcome].values
    
    return (
        np.average(Y[T == 1], weights=weights[T == 1]) -
        np.average(Y[T == 0], weights=weights[T == 0])
    )


def bootstrap_ci(
        df, 
        confounders,
        treatment, 
        outcome, 
        n_bootstrap = 200, 
        alpha = 0.05, 
        trim_percentile = 99
    ):
    """
    Compute bootstrap confidence intervals for the ATE.

    Resamples the dataset with replacement, re-estimates IPTW weights,
    and recalculates the ATE in each bootstrap iteration.

    Parameters
    ----------
    df : pd.DataFrame
        Analysis dataset.
    confounders : list[str]
        Adjustment variables.
    treatment : str
        Treatment variable.
    outcome : str
        Outcome variable.
    n_bootstrap : int, default=200
        Number of bootstrap replications.
    alpha : float, default=0.05
        Significance level.
    trim_percentile : int, default=99
        Upper percentile used to trim extreme weights.

    Returns
    -------
    tuple
        (lower_bound, upper_bound, ate_samples)
    """
    ates = []
    n = len(df)

    for _ in range(n_bootstrap):

        sample = df.sample(n=n, replace=True)
        _, _, w, _ = compute_iptw_weights(
            sample, 
            confounders,
            treatment, 
            trim_percentile=trim_percentile
        )
        ates.append(ate_iptw(sample, treatment, outcome, w))
    
    lower = np.percentile(ates, 100 * alpha / 2)
    upper = np.percentile(ates, 100 * (1 - alpha / 2))
    
    return lower, upper, np.array(ates)


def ate_subgrupo(df, treatment, outcome, weights, mask, nome):
    """
    Estimate subgroup-specific weighted ATE.

    Used for heterogeneous treatment effect analysis (approximate CATE).

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset.
    treatment : str
        Treatment variable.
    outcome : str
        Outcome variable.
    weights : array-like
        Global IPTW weights.
    mask : array-like of bool
        Boolean mask defining the subgroup.
    nome : str
        Subgroup name.

    Returns
    -------
    dict
        Dictionary containing subgroup name, estimated ATE,
        and treated/control sample sizes.
    """

    sub = df[mask].copy()

    T = sub[treatment].values
    Y = sub[outcome].values
    W = weights[mask]

    n_tratados = int((T == 1).sum())
    n_controle = int((T == 0).sum())

    if n_tratados == 0 or n_controle == 0:
        return {
            'Subgrupo': nome,
            'ATE (p.p.)': np.nan,
            'N tratados': n_tratados,
            'N controle': n_controle
        }

    ate = (
        np.average(Y[T == 1], weights=W[T == 1]) -
        np.average(Y[T == 0], weights=W[T == 0])
    )

    return {
        'Subgrupo': nome,
        'ATE (p.p.)': round(ate * 100, 2),
        'N tratados': n_tratados,
        'N controle': n_controle
    }