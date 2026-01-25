import numpy as np
import pandas as pd

from config import REQUIRED_RAW_COLUMNS, TARGET_COLUMN
from utils import safe_div, safe_log1p, winsorize_series, EPS

def load_raw_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_RAW_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    df["company_id"] = df["company_id"].astype(str)
    df["year"] = df["year"].astype(int)
    return df

def add_proxy_default(df: pd.DataFrame, train_year: int,
                      leverage_train_percentile: float = 0.75,
                      min_conditions: int = 2):
    df = df.copy()

    # EBIT proxy
    df["ebit_proxy"] = (
        df["gross_profit"]
        - df["selling_expenses"].fillna(0)
        - df["admin_expenses"].fillna(0)
    )

    df["interest_coverage"] = df["ebit_proxy"] / (df["interest_expenses"].abs() + EPS)
    df["debt_to_equity_raw"] = safe_div(df["total_liabilities"], df["equity"])

    train = df[df["year"] == train_year].copy()
    lev_th = float(train["debt_to_equity_raw"].replace([np.inf,-np.inf], np.nan).dropna().quantile(leverage_train_percentile))

    cond_profit = (df["net_income"] < 0)
    cond_cfo = (df["cashflow_ops"] < 0)
    cond_icr = (df["interest_coverage"] < 1)
    cond_lev = (df["debt_to_equity_raw"] > lev_th)

    df["distress_score"] = cond_profit.astype(int) + cond_cfo.astype(int) + cond_icr.astype(int) + cond_lev.astype(int)
    df[TARGET_COLUMN] = (df["distress_score"] >= min_conditions).astype(int)

    meta = {"leverage_threshold": lev_th, "train_year": train_year}
    return df, meta

def add_features(df: pd.DataFrame, missing_flag_cols: list[str]) -> pd.DataFrame:
    df = df.copy()

    for c in missing_flag_cols:
        df[f"is_missing_{c}"] = df[c].isna().astype(int)

    # Liquidity
    df["current_ratio"] = safe_div(df["current_assets"], df["current_liabilities"])
    df["cash_ratio"] = safe_div(df["cash_and_equivalents"], df["current_liabilities"])

    # Leverage
    df["debt_to_equity"] = safe_div(df["total_liabilities"], df["equity"])
    df["debt_to_assets"] = safe_div(df["total_liabilities"], df["total_assets"])
    df["short_debt_ratio"] = safe_div(df["short_term_debt"], df["total_liabilities"])

    # Profitability
    df["gross_margin"] = safe_div(df["gross_profit"], df["revenue"])
    df["net_margin"] = safe_div(df["net_income"], df["revenue"])
    df["roa"] = safe_div(df["net_income"], df["total_assets"])
    df["roe"] = safe_div(df["net_income"], df["equity"])

    # Coverage already created by proxy step if you run it first; keep safe:
    if "interest_coverage" not in df.columns:
        df["ebit_proxy"] = df["gross_profit"] - df["selling_expenses"].fillna(0) - df["admin_expenses"].fillna(0)
        df["interest_coverage"] = df["ebit_proxy"] / (df["interest_expenses"].abs() + EPS)

    # Cashflow
    df["ocf_to_debt"] = safe_div(df["cashflow_ops"], df["total_liabilities"])
    df["ocf_margin"] = safe_div(df["cashflow_ops"], df["revenue"])

    # Expense intensity
    df["selling_to_rev"] = safe_div(df["selling_expenses"], df["revenue"])
    df["admin_to_rev"] = safe_div(df["admin_expenses"], df["revenue"])
    df["interest_to_rev"] = safe_div(df["interest_expenses"], df["revenue"])

    # Size
    df["log_assets"] = safe_log1p(df["total_assets"])

    df.replace([np.inf,-np.inf], np.nan, inplace=True)

    # winsorize engineered ratios to reduce outlier dominance
    ratio_cols = [c for c in df.columns if c.endswith(("ratio","margin","roe","roa","coverage","to_rev","to_debt"))]
    ratio_cols = list(set(ratio_cols + ["log_assets"]))
    for c in ratio_cols:
        if c in df.columns:
            df[c] = winsorize_series(df[c], 0.01, 0.99)

    return df
