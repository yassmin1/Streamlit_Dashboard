from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd

try:
    import duckdb  # optional
except Exception:
    duckdb = None

# -------------------------
# Config & constants
# -------------------------

REQUIRED_COLUMNS = [
    "C1_BANNER_ID",
    "C1_ACADEMIC_PERIOD_DESC",
    "C1_ACADEMIC_PERIOD",
    "C1_CALENDAR_YEAR",
    "C1_COLLEGE",
    "C1_GENDER_DESC",
    "C1_CURRENT_AGE",
    "C1_FTIC_DC_DESC",
    "C1_TYPE_MAJOR_DESC",
    "C1_FTPT_COLLEGE_CENSUS",
    "C1_THECB_ETHNICITY",
]

REQUIRED_COLUMNS_STU=[
    "Term", 
    "Student ID",
    "Major Desc",

]

COLUMN_MAPPING = {
    "C1_CALENDAR_YEAR": "Calendar Year",
    "C1_ACADEMIC_PERIOD_DESC": "Academic Period",
    "C1_CBM_TERM_DESC": "Term",
    "C1_COLLEGE": "SPC College",
    "C1_GENDER_DESC": "Gender",
    "C1_FTIC_DC_DESC": "Student Type",
    "C1_TYPE_MAJOR_DESC": "Major Type",
    "C1_FTPT_COLLEGE_CENSUS": "Full_Part Time",
    "C1_THECB_ETHNICITY": "Ethnicity",
    "C1_CURRENT_AGE": "Age",
    "Major Desc":"Major",
}

AGE_BINS = [0, 18, 25, 30, 35, 40, 50, 60, 200]
AGE_LABELS = ["Under 18", "18-24", "25-29", "30-34", "35-39", "40-49", "50-59", "60+"]


@dataclass
class ETLPaths:
    raw_inputs: List[Path]            # explicit files OR directories to glob under
    stu_inputs: List[Path]            # explicit files OR directories to glob under
    out_parquet: Path                 # curated parquet output
    out_csv: Path                     # curated csv output
    duckdb_path: Optional[Path] = None  # optional duckdb file
    duckdb_table: str = "cbmc1_merged"


# -------------------------
# Logging
# -------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
logger = logging.getLogger("cbmc1_etl")


# -------------------------
# Utilities
# -------------------------

def _iter_input_files(inputs: Iterable[Path]) -> Iterable[Path]:
    """Yield CSV/XLS/XLSX files from paths (files or directories)."""
    for p in inputs:
        p = Path(p)
        if p.is_dir():
            # read all CSV + Excel in this directory (one-level glob)
            for f in p.glob("*.csv"):
                yield f
            for f in p.glob("*.xls*"):
                yield f
        elif p.is_file():
            if p.suffix.lower() in {".csv", ".xls", ".xlsx"}:
                yield p
        else:
            logger.warning("Path does not exist: %s", p)


def _read_one_file(fp: Path, columns: List[str]) -> pd.DataFrame:
    """Read a single CSV/Excel, subset columns, robust to encodings and date parsing."""
    suffix = fp.suffix.lower()
    try:
        if suffix == ".csv":
            # fast-path: try utf-8, fall back to latin1
            try:
                df = pd.read_csv(fp, dtype=str, low_memory=False)
            except UnicodeDecodeError:
                df = pd.read_csv(fp, dtype=str, encoding="latin1", low_memory=False)
        elif suffix in {".xlsx", ".xls"}:
            # engine auto-chooses; for xls you may need xlrd installed
            df = pd.read_excel(fp, dtype=str)
        else:
            raise ValueError(f"Unsupported file type: {fp}")

        missing = [c for c in columns if c not in df.columns]
        if missing:
            logger.warning("Missing columns in %s: %s", fp.name, missing)

        keep = [c for c in columns if c in df.columns]
        df = df[keep].copy()
        df["__source_file"] = fp.name  # lineage
        return df

    except Exception as e:
        logger.exception("Failed to read %s: %s", fp, e)
        # Return empty DF with required columns so concat is safe
        return pd.DataFrame(columns=columns + ["__source_file"])


def extract_merge(inputs: Iterable[Path], columns: List[str] = REQUIRED_COLUMNS) -> pd.DataFrame:
    """Extract: read many CSV/Excel → vertical concat, subsetting columns early."""
    files = list(_iter_input_files(inputs))
    if not files:
        logger.error("No input files found.")
        return pd.DataFrame(columns=columns)

    logger.info("Found %d files. Reading...", len(files))
    dfs = [_read_one_file(fp, columns) for fp in files]
    merged = pd.concat(dfs, ignore_index=True)
    logger.info("Merged shape after extract: %s", merged.shape)
    return merged


def _age_from_dob(dob_series: pd.Series) -> pd.Series:
    """
    Compute exact age in years (month/day aware) from a single column (Series)
    of DATE_OF_BIRTH values.
    Works even if some dates are missing or invalid.
    """
    # 1. Convert column to datetime (handles strings, Excel serials, NaNs)
    dob = pd.to_datetime(dob_series, errors="coerce")

    # 2. Get today's date (scalar timestamp)
    today = pd.Timestamp.today()

    # 3. Extract components (safe for Series)
    year = dob.dt.year
    month = dob.dt.month
    day = dob.dt.day

    # 4. Compute base difference in years
    diff = today.year - year

    # 5. Subtract 1 if birthday hasn't occurred yet this year
    had_birthday = (today.month > month) | ((today.month == month) & (today.day >= day))
    age = diff - (~had_birthday).astype("Int64")

    # 6. Keep null where dob is NaT
    return age.where(dob.notna()).astype("Int64")



def merge_student_data(
    cbm_df: pd.DataFrame,
    stu_df: pd.DataFrame,
    left_keys: list[str] = ['C1_ACADEMIC_PERIOD', 'C1_BANNER_ID'],
    right_keys: list[str] = ['Term', 'Student ID'],
    how: str = 'left',
    suffixes: tuple[str, str] = ("_cbm", "_stu"),
) -> pd.DataFrame:
    """
    Merge CBM dataset with student dataset safely and cleanly.

    Parameters
    ----------
    cbm_df : pd.DataFrame
        The main CBM dataset.
    stu_df : pd.DataFrame
        The student data to join (lookup/enrichment).
    left_keys : list[str]
        Column names in cbm_df to join on.
    right_keys : list[str]
        Column names in stu_df to join on.
    how : str
        Type of join (default 'left').
    suffixes : tuple[str, str]
        Suffixes to apply to overlapping columns.

    Returns
    -------
    pd.DataFrame
        Merged DataFrame with duplicates removed based on left_keys.
    """
    try:
        logger.info(f"Merging on {left_keys} ↔ {right_keys} using '{how}' join...")

        merged = pd.merge(
            cbm_df,
            stu_df,
            left_on=left_keys,
            right_on=right_keys,
            how=how,
            suffixes=suffixes
        )

        before = len(merged)
        merged = merged.drop_duplicates(subset=left_keys, keep='first')
        after = len(merged)

        logger.info(f"Merge complete. Rows: {before} → {after} after deduplication.")
        return merged.reset_index(drop=True)

    except KeyError as e:
        logger.error(f"KeyError during merge: {e}")
        missing_keys = [k for k in left_keys if k not in cbm_df.columns] + \
                       [k for k in right_keys if k not in stu_df.columns]
        raise KeyError(f"Missing join keys: {missing_keys}") from e

    except Exception as e:
        logger.exception("Unexpected error during merge:")
        raise



def transform_cbmc1(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform: the merged dataframe
      - rename columns
      - compute Age + Age_Group (and drop raw DOB & Age)
      - trim whitespace, standardize categories
      - drop all-null rows
    """

    if df.empty:
        logger.warning("Empty dataframe passed to transform; returning as-is.")
        return df

    # Standardize column names & trim strings
    df = df.drop_duplicates().copy()
    # basic whitespace cleanup
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].astype(str).str.strip()

    # rename only existing columns
    existing_map = {k: v for k, v in COLUMN_MAPPING.items() if k in df.columns}
    df = df.rename(columns=existing_map)

    # age + age group
    if "Age" in df.columns:
        age = pd.to_numeric(df["Age"], errors="coerce").astype("Int64")
        df["Age_Group"] = pd.cut(
            age.astype("float"),
            bins=AGE_BINS,
            labels=AGE_LABELS,
            include_lowest=True,
            right=False,
        )
        df = df.drop(columns=["Age"], errors="ignore")
        # If you want to keep exact age, uncomment:
        # df["Age"] = age

    # Drop rows completely empty
    df = df.dropna().drop_duplicates().reset_index(drop=True)

    # Optional: cast some columns to category for smaller Parquet
    for cat_col in ["Term", "SPC College", "Gender", "Student Type", "Type_Major", "Full_Part_Time", "ETHNICITY", "Age_Group","Major"]:
        if cat_col in df.columns:
            df[cat_col] = df[cat_col].astype("category")

    logger.info("Transformed shape: %s | Columns: %s", df.shape, list(df.columns))
    return df


def load_parquet(df: pd.DataFrame, out_path_par: Path,out_path_csv: Path) -> Path:
    out_path_par.parent.mkdir(parents=True, exist_ok=True)
    # filter out feature not need to show 
    col_delete=['__source_file_cbm','C1_BANNER_ID','__source_file_stu','Student ID','SPC College','C1_ACADEMIC_PERIOD']
    df = df.drop(columns=col_delete, errors="ignore").dropna().reset_index(drop=True)
    # Use pyarrow by default if available

    df.to_parquet(out_path_par, index=False)
    df.to_csv(out_path_csv, index=False)
    logger.info("Wrote curated Parquet → %s (rows=%s)", out_path_par, len(df))
    return "save files in data/curated"


def load_duckdb(df: pd.DataFrame, db_path: Path, table: str) -> Optional[Path]:
    if duckdb is None:
        logger.warning("duckdb not installed; skipping DuckDB load.")
        return None
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(db_path))
    try:
        con.register("df_mem", df)
        con.execute(f"CREATE OR REPLACE TABLE {table} AS SELECT * FROM df_mem;")
        logger.info("Wrote DuckDB table %s in %s", table, db_path)
    finally:
        con.close()
    return db_path


def run_etl(paths: ETLPaths) -> Path:
    raw_df = extract_merge(paths.raw_inputs, REQUIRED_COLUMNS) # Extract + merge
    stu_df = extract_merge(paths.stu_inputs, REQUIRED_COLUMNS_STU) #
    input_df = merge_student_data(
    cbm_df=raw_df,
    stu_df=stu_df,
    left_keys=['C1_ACADEMIC_PERIOD', 'C1_BANNER_ID'],
    right_keys=['Term', 'Student ID'],
    how='left') # Merge CBM + Student data
    curated = transform_cbmc1(input_df) #
    out = load_parquet(curated, paths.out_parquet,paths.out_csv.with_suffix('.csv'))# Load Parquet (and CSV)
    if paths.duckdb_path:
        load_duckdb(curated, paths.duckdb_path, paths.duckdb_table)
    return out


# -------------------------
# CLI
# -------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CBMC1 ETL: merge CSV/Excel → Parquet (and optional DuckDB).")
    p.add_argument(
        "--cbmc_inputs",
        nargs="+",
        required=True,
        help="Files or directories (space-separated). Directories will be scanned for *.csv and *.xls*", 

    )
    p.add_argument(
        "--stu_inputs",
        nargs="+",
        required=True,
        help="Files or directories (space-separated). Directories will be scanned for *.csv and *.xls*", 

    )
    p.add_argument(
        "--out_parquet",
        required=True,
        help="Path to curated parquet, e.g. data/curated/merged.parquet",
    )    
    p.add_argument(
        "--out_csv",
        required=True,
        help="Path to curated csv, e.g. data/curated/merged.csv",    
    )
    p.add_argument(
        "--duckdb",
        default=None,
        help="Optional DuckDB file path, e.g. warehouse/warehouse.duckdb",
    )
    p.add_argument(
        "--duckdb-table",
        default="cbmc1_merged",
        help="DuckDB table name (default: cbmc1_merged)",
    )
    return p.parse_args()


def main():
    args = _parse_args()
    paths = ETLPaths(
        raw_inputs=[Path(p) for p in args.cbmc_inputs],
        stu_inputs=[Path(p) for p in args.stu_inputs],
        out_parquet=Path(args.out_parquet),
        out_csv=Path(args.out_csv),
        duckdb_path=Path(args.duckdb) if args.duckdb else None,
        duckdb_table=args.duckdb_table,
    )
    run_etl(paths)


if __name__ == "__main__":
    main()
