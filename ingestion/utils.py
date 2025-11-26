import csv
from datetime import datetime
from typing import Dict, List, Tuple, Any, Iterable

from .schemas import REQUIRED_FIELDS, OPTIONAL_FIELDS


def normalize_headers(headers: List[str]) -> List[str]:
    return [h.strip().lower().replace(" ", "_") for h in headers]


def validate_record(dataset: str, record: Dict[str, Any]) -> Tuple[bool, List[str]]:
    errors: List[str] = []
    required = REQUIRED_FIELDS.get(dataset, [])
    for f in required:
        if record.get(f) in (None, ""):
            errors.append(f"Missing required field: {f}")
    return (len(errors) == 0, errors)


def preprocess_record(dataset: str, record: Dict[str, Any]) -> Dict[str, Any]:
    # Common cleanups
    for k, v in list(record.items()):
        if isinstance(v, str):
            record[k] = v.strip()
    # Normalize common keys
    if "course_code" in record and isinstance(record["course_code"], str):
        record["course_code"] = record["course_code"].upper()
    # Dataset specific coercions
    if dataset == "academic_records":
        # grade as string, credits to float/int if possible
        if "credits" in record and record["credits"] not in (None, ""):
            try:
                record["credits"] = float(record["credits"]) if "." in str(record["credits"]) else int(record["credits"])
            except Exception:
                pass
    elif dataset == "lms":
        if "event_time" in record:
            record["event_time"] = parse_date_safe(record.get("event_time"))
    elif dataset == "attendance":
        if "date" in record:
            record["date"] = parse_date_safe(record.get("date"), date_only=True)
        if "status" in record and isinstance(record["status"], str):
            record["status"] = record["status"].lower()
    elif dataset == "demographics":
        if "dob" in record:
            record["dob"] = parse_date_safe(record.get("dob"), date_only=True)
    return record


def parse_date_safe(value: Any, date_only: bool = False):
    if value in (None, ""):
        return None
    # Try common formats without external deps
    candidates = [
        "%Y-%m-%dT%H:%M:%S.%fZ",
        "%Y-%m-%dT%H:%M:%S.%f",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d",
        "%d/%m/%Y",
        "%m/%d/%Y",
    ]
    s = str(value).strip()
    for fmt in candidates:
        try:
            dt = datetime.strptime(s, fmt)
            if date_only:
                return dt.date().isoformat()
            return dt.isoformat()
        except Exception:
            continue
    # Fallback: try fromisoformat for partial ISO strings
    try:
        dt = datetime.fromisoformat(s)
        if date_only:
            return dt.date().isoformat()
        return dt.isoformat()
    except Exception:
        return s  # keep original if cannot parse


def read_csv_stream(file_storage) -> Iterable[Dict[str, Any]]:
    # file_storage is werkzeug FileStorage
    file_storage.stream.seek(0)
    decoded = (line.decode("utf-8", errors="ignore") for line in file_storage.stream)
    reader = csv.DictReader(decoded)
    reader.fieldnames = normalize_headers(reader.fieldnames or [])
    for row in reader:
        normalized = {k.strip().lower().replace(" ", "_"): (v.strip() if isinstance(v, str) else v) for k, v in row.items()}
        yield normalized
