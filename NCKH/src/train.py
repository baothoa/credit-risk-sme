import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from joblib import dump

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.svm import SVC

from config import FEATURE_COLUMNS, TARGET_COLUMN
from utils import ks_statistic

def build_models(y_train: np.ndarray, random_state: int) -> Dict[str, Any]:
    imputer = SimpleImputer(strategy="median")

    lr = Pipeline([
        ("imp", imputer),
        ("sc", StandardScaler()),
        ("clf", LogisticRegression(max_iter=4000, class_weight="balanced", random_state=random_state))
    ])

    rf = Pipeline([
        ("imp", imputer),
        ("clf", RandomForestClassifier(
            n_estimators=700, min_samples_leaf=3,
            class_weight="balanced",
            random_state=random_state, n_jobs=-1
        ))
    ])

    # Modern boosting
    boost_name, boost = None, None
    try:
        import lightgbm as lgb
        boost_name = "LightGBM"
        boost = Pipeline([
            ("imp", imputer),
            ("clf", lgb.LGBMClassifier(
                n_estimators=1500, learning_rate=0.03,
                num_leaves=31, subsample=0.9, colsample_bytree=0.9,
                class_weight="balanced",
                random_state=random_state
            ))
        ])
    except Exception:
        try:
            from xgboost import XGBClassifier
            boost_name = "XGBoost"
            scale_pos = float(np.sum(y_train == 0) / max(np.sum(y_train == 1), 1))
            boost = Pipeline([
                ("imp", imputer),
                ("clf", XGBClassifier(
                    n_estimators=1500, learning_rate=0.03,
                    max_depth=4, subsample=0.9, colsample_bytree=0.9,
                    reg_lambda=1.0,
                    random_state=random_state, n_jobs=-1,
                    eval_metric="logloss",
                    scale_pos_weight=scale_pos
                ))
            ])
        except Exception:
            boost_name = "HGB_fallback"
            boost = Pipeline([
                ("imp", imputer),
                ("clf", HistGradientBoostingClassifier(
                    learning_rate=0.05, max_depth=4, max_iter=800,
                    random_state=random_state
                ))
            ])

    svm = Pipeline([
        ("imp", imputer),
        ("sc", StandardScaler()),
        ("clf", SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=random_state))
    ])

    return {"LogisticRegression": lr, "RandomForest": rf, boost_name: boost, "SVM_RBF": svm}

def fit_and_eval(models: Dict[str, Any], train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    Xtr, ytr = train_df[FEATURE_COLUMNS], train_df[TARGET_COLUMN].astype(int).values
    Xte, yte = test_df[FEATURE_COLUMNS], test_df[TARGET_COLUMN].astype(int).values

    rows, fitted = [], {}
    for name, m in models.items():
        m.fit(Xtr, ytr)
        p = m.predict_proba(Xte)[:, 1]

        rows.append({
            "model": name,
            "ROC_AUC": roc_auc_score(yte, p),
            "PR_AUC": average_precision_score(yte, p),
            "KS": ks_statistic(yte, p),
            "Brier": brier_score_loss(yte, p),
            "test_default_rate": float(yte.mean())
        })
        fitted[name] = m

    metrics = pd.DataFrame(rows).sort_values(["ROC_AUC","PR_AUC","KS"], ascending=False)
    return metrics, fitted

def save_models(fitted: Dict[str, Any], models_dir: str):
    for name, model in fitted.items():
        dump(model, f"{models_dir}/{name}.joblib")
