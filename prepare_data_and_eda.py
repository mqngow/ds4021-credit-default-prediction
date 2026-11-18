"""Reproducible 80/20 stratified split and concise proposal EDA for UCI Credit Card Default dataset."""

from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

DATA_DIR = Path(__file__).resolve().parent.parent / "Data"
RAW_PATH = DATA_DIR / "UCI_Credit_Card.csv"
DEV_PATH = DATA_DIR / "development_set.csv"
REPORTING_PATH = DATA_DIR / "reporting_set.csv"

TARGET_COL = "default.payment.next.month"
RANDOM_STATE = 42
REPORTING_FRACTION = 0.20

SEX_LABELS = {1: "Male", 2: "Female"}
EDUCATION_LABELS = {
    1: "Graduate school",
    2: "University",
    3: "High school",
    4: "Others",
    5: "Unknown (5)",
    6: "Unknown (6)",
    0: "Undocumented (0)",
}
MARRIAGE_LABELS = {
    1: "Married",
    2: "Single",
    3: "Others",
    0: "Undocumented (0)",
}


def split_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split dataset into 80% development and 20% untouched reporting sets, stratified on target."""
    dev_df, reporting_df = train_test_split(
        df,
        test_size=REPORTING_FRACTION,
        random_state=RANDOM_STATE,
        stratify=df[TARGET_COL],
    )
    dev_df.to_csv(DEV_PATH, index=False)
    reporting_df.to_csv(REPORTING_PATH, index=False)
    return dev_df, reporting_df


def print_categorical_summary(series: pd.Series, label_map: dict[int, str], title: str) -> None:
    """Print counts and percentages for a coded categorical variable."""
    counts = series.value_counts().sort_index()
    props = series.value_counts(normalize=True).sort_index() * 100
    print(f"\n{title}:")
    for code, count in counts.items():
        name = label_map.get(int(code), f"Code {code}")
        print(f"  {code} ({name}): {count:,} ({props[code]:.2f}%)")


def summarize_demographics(df: pd.DataFrame, label: str) -> None:
    """Print concise demographic and target summary statistics."""
    print(f"=== {label} (N = {len(df):,}, columns = {df.shape[1]}) ===")
    print(f"Missing values: {int(df.isna().sum().sum())}")
    print(f"Duplicate IDs: {int(df['ID'].duplicated().sum())}")

    age = df["AGE"]
    print(
        f"\nAGE (years): mean = {age.mean():.2f}, SD = {age.std(ddof=1):.2f}, "
        f"median = {age.median():.1f}, IQR = {age.quantile(0.75) - age.quantile(0.25):.1f} "
        f"(Q1={age.quantile(0.25):.1f}, Q3={age.quantile(0.75):.1f}), "
        f"min = {age.min()}, max = {age.max()}"
    )

    lim = df["LIMIT_BAL"]
    print(
        f"LIMIT_BAL (NT$): mean = {lim.mean():,.2f}, SD = {lim.std(ddof=1):,.2f}, "
        f"median = {lim.median():,.1f}, IQR = {lim.quantile(0.75) - lim.quantile(0.25):,.1f} "
        f"(Q1={lim.quantile(0.25):,.1f}, Q3={lim.quantile(0.75):,.1f}), "
        f"min = {lim.min():,.0f}, max = {lim.max():,.0f}"
    )

    print_categorical_summary(df["SEX"], SEX_LABELS, "SEX")
    print_categorical_summary(df["EDUCATION"], EDUCATION_LABELS, "EDUCATION")
    print_categorical_summary(df["MARRIAGE"], MARRIAGE_LABELS, "MARRIAGE")

    target_counts = df[TARGET_COL].value_counts().sort_index()
    target_props = df[TARGET_COL].value_counts(normalize=True).sort_index() * 100
    print("\nTarget (default.payment.next.month):")
    print(f"  0 (No default): {target_counts[0]:,} ({target_props[0]:.2f}%)")
    print(f"  1 (Default):    {target_counts[1]:,} ({target_props[1]:.2f}%)")


def main() -> None:
    df = pd.read_csv(RAW_PATH)
    dev_df, reporting_df = split_dataset(df)

    summarize_demographics(df, "Full Dataset")
    print()
    summarize_demographics(dev_df, "Development Set (80%)")

    print(
        f"\nReporting Set saved to {REPORTING_PATH} "
        f"(rows = {len(reporting_df):,}, size = {REPORTING_PATH.stat().st_size / 1024:.1f} KB, "
        f"default rate = {reporting_df[TARGET_COL].mean() * 100:.2f}%)."
    )


if __name__ == "__main__":
    main()
