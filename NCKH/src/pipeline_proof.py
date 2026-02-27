import os
import json
import joblib
import warnings

import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer

warnings.filterwarnings("ignore")


def _print_step(msg: str):
    print(f"[OK] {msg}")


def main():
    # --- Paths (relative to NCKH/) ---
    here = os.path.dirname(os.path.abspath(__file__))          # .../NCKH/src
    project_root = os.path.abspath(os.path.join(here, ".."))   # .../NCKH

    raw_path = os.path.join(project_root, "data", "raw", "final_research.csv")
    processed_dir = os.path.join(project_root, "data", "processed")
    artifacts_dir = os.path.join(project_root, "models", "artifacts")

    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(artifacts_dir, exist_ok=True)

    # --- Import project code (config/features in same src folder) ---
    from config import (
        ExperimentConfig,
        FEATURE_COLUMNS,
        TARGET_COLUMN,
        GROUP_COLUMN,
        REQUIRED_RAW_COLUMNS,
    )
    from features import load_raw_data, add_proxy_default, add_features

    cfg = ExperimentConfig()

    # -----------------------------
    # 1) Load raw data
    # -----------------------------
    df = load_raw_data(raw_path)
    _print_step(f"Loaded raw data: {df.shape[0]} rows, {df.shape[1]} cols from data/raw/final_research.csv")

    # Basic validation
    missing = [c for c in REQUIRED_RAW_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in raw data: {missing}")
    _print_step("Validated required raw columns")

    # -----------------------------
    # 2) Create proxy_default (TARGET)
    # -----------------------------
    df, meta = add_proxy_default(
        df,
        train_year=cfg.TRAIN_YEAR,
        leverage_train_percentile=cfg.LEVERAGE_TRAIN_PERCENTILE,
        min_conditions=cfg.PROXY_DEFAULT_MIN_COUNT,
    )
    _print_step(f"Created target '{TARGET_COLUMN}' (proxy_default)")

    # Save labeling metadata (e.g., leverage threshold used)
    meta_path = os.path.join(processed_dir, "labeling_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    _print_step("Saved labeling metadata to data/processed/labeling_meta.json")

    # -----------------------------
    # 3) Feature engineering
    # -----------------------------
    df = add_features(df, missing_flag_cols=cfg.MISSING_FLAG_COLS)
    _print_step("Finished feature engineering")

    # -----------------------------
    # 4) Time split: train=2023, test=2024
    # -----------------------------
    train_df = df[df["year"] == cfg.TRAIN_YEAR].copy()
    test_df = df[df["year"] == cfg.TEST_YEAR].copy()

    if train_df.empty:
        raise ValueError(f"TRAIN split is empty for year={cfg.TRAIN_YEAR}")
    if test_df.empty:
        raise ValueError(f"TEST split is empty for year={cfg.TEST_YEAR}")

    _print_step(f"Time split done: TRAIN year={cfg.TRAIN_YEAR} ({len(train_df)} rows), TEST year={cfg.TEST_YEAR} ({len(test_df)} rows)")

    # -----------------------------
    # 5) Prepare X/y (only configured features)
    # -----------------------------
    X_train = train_df[FEATURE_COLUMNS].copy()
    y_train = train_df[TARGET_COLUMN].astype(int).copy()

    X_test = test_df[FEATURE_COLUMNS].copy()
    y_test = test_df[TARGET_COLUMN].astype(int).copy()  # not used for metrics; kept for completeness

    _print_step("Selected feature columns and target column")

    # -----------------------------
    # 6) Median imputation (FIT on TRAIN, TRANSFORM TEST) - avoid leakage
    # -----------------------------
    imputer = SimpleImputer(strategy="median")

    X_train_imp = pd.DataFrame(imputer.fit_transform(X_train), columns=FEATURE_COLUMNS, index=X_train.index)
    X_test_imp = pd.DataFrame(imputer.transform(X_test), columns=FEATURE_COLUMNS, index=X_test.index)

    _print_step("Median imputation done (fit on TRAIN -> transform TEST)")

    # Save imputer artifact
    imputer_path = os.path.join(artifacts_dir, "median_imputer.joblib")
    joblib.dump(imputer, imputer_path)
    _print_step("Saved imputer artifact to models/artifacts/median_imputer.joblib")

    # -----------------------------
    # 7) Train LightGBM (no metric printing)
    # -----------------------------
    try:
        import lightgbm as lgb
    except ImportError as e:
        raise ImportError(
            "Missing dependency: lightgbm. Install it with: pip install lightgbm"
        ) from e

    model = lgb.LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=cfg.RANDOM_STATE,
    )

    model.fit(X_train_imp, y_train)
    _print_step("Trained LightGBM model on TRAIN")

    model_path = os.path.join(artifacts_dir, "LightGBM_joblib.joblib")
    joblib.dump(model, model_path)
    _print_step("Saved model artifact to models/artifacts/LightGBM_joblib.joblib")

    # -----------------------------
    # 8) Predict PD on TEST
    # -----------------------------
    pd_pred = model.predict_proba(X_test_imp)[:, 1]
    _print_step("Predicted PD for TEST set")

    # -----------------------------
    # 9) Save scored output (for proof/pipeline submission)
    # -----------------------------
    scored = test_df[[GROUP_COLUMN, "year"]].copy()
    scored["PD"] = pd_pred

    out_path = os.path.join(processed_dir, "scored_test_data.csv")
    scored.to_csv(out_path, index=False)
    _print_step("Saved scored output to data/processed/scored_test_data.csv")

    # Done
    print("\n✅ PIPELINE COMPLETED (proof of algorithm). No metrics printed.\n")


if __name__ == "__main__":
    main()