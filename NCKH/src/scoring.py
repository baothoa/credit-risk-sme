import numpy as np
import pandas as pd

from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss
)


# =========================================================
# 1. Calibration
# =========================================================
def calibrate_model(
    base_model,
    X_train,
    y_train,
    method="isotonic",
    n_splits=5,
    random_state=42
):
    """
    Calibrate Probability of Default (PD) using cross-validated calibration.
    """

    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state
    )

    calibrated_model = CalibratedClassifierCV(
        estimator=base_model,
        method=method,
        cv=cv
    )


    calibrated_model.fit(X_train, y_train)

    return calibrated_model


# =========================================================
# 2. Evaluate PD quality
# =========================================================
def eval_pd(model, X_test, y_test):
    """
    Evaluate PD predictions using credit-risk metrics.
    """

    pd_hat = model.predict_proba(X_test)[:, 1]

    metrics = {
        "ROC_AUC": roc_auc_score(y_test, pd_hat),
        "PR_AUC": average_precision_score(y_test, pd_hat),
        "Brier": brier_score_loss(y_test, pd_hat)
    }

    return pd.Series(metrics), pd_hat


# =========================================================
# 3. PD → Credit Score (1–10)
# =========================================================
def pd_to_score(pd_series, bins=10):
    """
    Convert PD to Credit Score (1–10) using quantile-based scoring.
    Lower PD → Higher Score.
    """

    # Quantile cut (higher PD = worse)
    score = pd.qcut(
        pd_series.rank(method="first"),
        q=bins,
        labels=False
    )

    # Convert to 1–10, reverse order
    credit_score = bins - score

    return credit_score.astype(int)
