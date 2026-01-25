from dataclasses import dataclass
from typing import List

@dataclass(frozen=True)
class Paths:
    PROJECT_ROOT: str = "/content/drive/MyDrive/NCKH"
    DATA_RAW: str = "/content/drive/MyDrive/NCKH/data/raw/final_research.csv"
    DATA_PROCESSED_DIR: str = "/content/drive/MyDrive/NCKH/data/processed"
    MODELS_DIR: str = "/content/drive/MyDrive/NCKH/models"
    ARTIFACTS_DIR: str = "/content/drive/MyDrive/NCKH/models/artifacts"

@dataclass(frozen=True)
class ExperimentConfig:
    TRAIN_YEAR: int = 2023
    TEST_YEAR: int = 2024
    RANDOM_STATE: int = 42

    PROXY_DEFAULT_MIN_COUNT: int = 2
    LEVERAGE_TRAIN_PERCENTILE: float = 0.75

    SCORE_BINS: int = 10
    MISSING_FLAG_COLS: List[str] = None

    def __post_init__(self):
        object.__setattr__(self, "MISSING_FLAG_COLS", ["cashflow_investing", "long_term_debt"])

REQUIRED_RAW_COLUMNS = [
    "company_id","year",
    "total_assets","equity","total_liabilities",
    "current_assets","current_liabilities","cash_and_equivalents",
    "short_term_debt","long_term_debt",
    "revenue","gross_profit","net_income",
    "selling_expenses","admin_expenses","interest_expenses",
    "cashflow_ops","cashflow_investing","cashflow_financing",
]

FEATURE_COLUMNS = [
    "current_ratio","cash_ratio",
    "debt_to_equity","debt_to_assets","short_debt_ratio",
    "gross_margin","net_margin","roa","roe",
    "interest_coverage",
    "ocf_to_debt","ocf_margin",
    "selling_to_rev","admin_to_rev","interest_to_rev",
    "is_missing_cashflow_investing","is_missing_long_term_debt",
    "log_assets",
]

TARGET_COLUMN = "proxy_default"
GROUP_COLUMN = "company_id"
