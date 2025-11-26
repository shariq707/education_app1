import io
from typing import Dict, List, Tuple, Any, Optional

import pandas as pd

ALLOWED_ATTENDANCE = {"present", "absent", "late"}


def normalize_headers_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    return df


def trim_strings(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.columns:
        if pd.api.types.is_object_dtype(df[col]):
            # Use pandas StringDtype to preserve <NA> instead of converting NaN to 'nan'
            df[col] = df[col].astype("string").str.strip()
    return df


def coerce_dates(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors='coerce').dt.strftime('%Y-%m-%dT%H:%M:%S')


def coerce_dates_date_only(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors='coerce').dt.date.astype('string')


# Broad set of forbidden special symbols; rows containing any of these in checked
# columns will be dropped during cleaning. We intentionally allow letters, digits,
# and spaces; hyphen is handled per-dataset via exclude_cols where needed (e.g., term).
FORBIDDEN_CHAR_PATTERN = r"[<>\?\\/\|\*@\$!\^\(\)\-\+=~`#%&.,\{\}\[\]:;\"']"


def drop_invalid_generic(df: pd.DataFrame, exclude_cols: Optional[List[str]] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Drop rows that have:
    - any missing values in any column (after trimming), including empty strings
    - any forbidden characters in any string field: > < ? / _ +
    - full-row duplicates

    Returns cleaned df and a summary dict with counts of drops.
    """
    summary: Dict[str, Any] = {
        "dropped_missing_any": 0,
        "dropped_forbidden_chars": 0,
        "dropped_full_duplicates": 0,
    }

    df = df.copy()
    exclude = set(exclude_cols or [])

    # Treat empty strings as missing
    df = df.replace(r"^\s*$", pd.NA, regex=True)

    # Drop any row with any missing value across any column
    before = len(df)
    df = df.dropna(how="any")
    summary["dropped_missing_any"] = before - len(df)

    # Drop any row that contains forbidden characters in any string-like column
    if len(df):
        mask_forbidden = pd.Series(False, index=df.index)
        for col in df.columns:
            if col in exclude:
                continue
            if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col]):
                contains = df[col].astype("string").str.contains(FORBIDDEN_CHAR_PATTERN, na=False)
                mask_forbidden = mask_forbidden | contains
        if mask_forbidden.any():
            summary["dropped_forbidden_chars"] = int(mask_forbidden.sum())
            df = df.loc[~mask_forbidden]

    # Drop full duplicates (across all columns)
    before = len(df)
    df = df.drop_duplicates()
    summary["dropped_full_duplicates"] = before - len(df)

    return df, summary


def clean_academic_records(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    summary: Dict[str, Any] = {"dropped_missing_required": 0, "deduplicated": 0}
    df = normalize_headers_df(df)
    df = trim_strings(df)
    # Generic invalid row removal (missing anywhere, forbidden chars, full duplicates)
    # Exclude 'term' so values like '2024-Fall' with hyphen are allowed
    df, gen_summary = drop_invalid_generic(df, exclude_cols=["term", "instructor", "remarks"])
    summary.update(gen_summary)
    # Normalize course_code to upper
    if "course_code" in df.columns:
        df["course_code"] = df["course_code"].str.upper()
    # Validate term format strictly (e.g., 2024-Fall)
    if "term" in df.columns:
        before = len(df)
        df["term"] = df["term"].astype("string")
        valid_term = df["term"].str.match(r"^[0-9]{4}-[A-Za-z]+$", na=False)
        df = df[valid_term]
        summary["dropped_invalid_term_format"] = summary.get("dropped_invalid_term_format", 0) + (before - len(df))
    # credits numeric
    if "credits" in df.columns:
        df["credits"] = pd.to_numeric(df["credits"], errors='coerce')
        if df["credits"].notna().any():
            median_val = df["credits"].median()
            df["credits"] = df["credits"].fillna(median_val)
    # Deduplicate by keys
    keys = ["student_id", "course_code", "term"]
    before = len(df)
    df = df.drop_duplicates(subset=[k for k in keys if k in df.columns])
    summary["deduplicated"] = before - len(df)
    return df, summary


def clean_demographics(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    summary: Dict[str, Any] = {"dropped_missing_required": 0, "deduplicated": 0}
    df = normalize_headers_df(df)
    df = trim_strings(df)
    # Generic invalid row removal (missing anywhere, forbidden chars, full duplicates)
    # Exclude DOB from forbidden-char checks so dates with '/' are allowed
    df, gen_summary = drop_invalid_generic(df, exclude_cols=["dob", "email", "phone", "address"])
    summary.update(gen_summary)
    # parse DOB to date-only
    if "dob" in df.columns:
        df["dob"] = coerce_dates_date_only(df["dob"])  # may produce <NA> for invalid
        before = len(df)
        df = df.dropna(subset=["dob"])  # drop invalid dob
        summary["dropped_invalid_dob"] = summary.get("dropped_invalid_dob", 0) + (before - len(df))
    # Deduplicate by student_id
    before = len(df)
    df = df.drop_duplicates(subset=["student_id"]) if "student_id" in df.columns else df
    summary["deduplicated"] = before - len(df)
    return df, summary


def clean_lms(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    summary: Dict[str, Any] = {"dropped_missing_required": 0, "deduplicated": 0}
    df = normalize_headers_df(df)
    df = trim_strings(df)
    # Generic invalid row removal (missing anywhere, forbidden chars, full duplicates)
    df, gen_summary = drop_invalid_generic(df, exclude_cols=["event_time", "details"])
    summary.update(gen_summary)
    # Normalize course code
    if "course_code" in df.columns:
        df["course_code"] = df["course_code"].str.upper()
    # Parse event_time to ISO
    if "event_time" in df.columns:
        df["event_time"] = coerce_dates(df["event_time"])  # may be NaT -> NaN string
        before = len(df)
        df = df.dropna(subset=["event_time"])  # drop invalid datetime
        summary["dropped_invalid_event_time"] = summary.get("dropped_invalid_event_time", 0) + (before - len(df))
    # Deduplicate by event_id
    before = len(df)
    df = df.drop_duplicates(subset=["event_id"]) if "event_id" in df.columns else df
    summary["deduplicated"] = before - len(df)
    return df, summary


def clean_attendance(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    summary: Dict[str, Any] = {"dropped_missing_required": 0, "deduplicated": 0, "dropped_invalid_status": 0}
    df = normalize_headers_df(df)
    df = trim_strings(df)
    # Generic invalid row removal (missing anywhere, forbidden chars, full duplicates)
    df, gen_summary = drop_invalid_generic(df, exclude_cols=["date", "remarks"])
    summary.update(gen_summary)
    # Normalize
    if "course_code" in df.columns:
        df["course_code"] = df["course_code"].str.upper()
    if "status" in df.columns:
        df["status"] = df["status"].str.lower()
        # map common variants
        df["status"] = df["status"].replace({
            "p": "present",
            "a": "absent",
            "l": "late",
            "present": "present",
            "absent": "absent",
            "late": "late",
        })
        before = len(df)
        df = df[df["status"].isin(ALLOWED_ATTENDANCE)]
        summary["dropped_invalid_status"] = before - len(df)
    # Parse date (date-only)
    if "date" in df.columns:
        df["date"] = coerce_dates_date_only(df["date"])  # may become <NA>
        before = len(df)
        df = df.dropna(subset=["date"])  # drop invalid dates
        summary["dropped_invalid_date"] = before - len(df)
    # Deduplicate by keys
    keys = ["student_id", "course_code", "date"]
    before = len(df)
    df = df.drop_duplicates(subset=[k for k in keys if k in df.columns])
    summary["deduplicated"] = summary["deduplicated"] + (before - len(df))
    return df, summary


def clean_with_pandas(dataset: str, file_storage) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    # Read CSV into DataFrame
    file_storage.stream.seek(0)
    df = pd.read_csv(file_storage.stream)
    dataset = dataset.lower()
    if dataset == "academic_records":
        return clean_academic_records(df)
    if dataset == "demographics":
        return clean_demographics(df)
    if dataset == "lms":
        return clean_lms(df)
    if dataset == "attendance":
        return clean_attendance(df)
    raise ValueError(f"Unsupported dataset: {dataset}")
