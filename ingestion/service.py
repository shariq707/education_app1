from typing import Any, Dict, Iterable, List, Tuple
from pymongo.collection import Collection

from .utils import validate_record, preprocess_record
from .schemas import REQUIRED_FIELDS


DATASET_KEYS = {
    "academic_records": ["student_id", "course_code", "term"],
    "demographics": ["student_id"],
    "lms": ["event_id"],
    "attendance": ["student_id", "course_code", "date"],
}


def get_collection(mongo_db, dataset: str) -> Collection:
    if dataset == "academic_records":
        return mongo_db.academic_records
    if dataset == "demographics":
        return mongo_db.demographics
    if dataset == "lms":
        return mongo_db.lms_events
    if dataset == "attendance":
        return mongo_db.attendance
    raise ValueError(f"Unsupported dataset: {dataset}")


def _build_key_query(dataset: str, record: Dict[str, Any]) -> Dict[str, Any]:
    keys = DATASET_KEYS.get(dataset, [])
    return {k: record.get(k) for k in keys}


def process_records(dataset: str, raw_records: Iterable[Dict[str, Any]], mongo_db) -> Dict[str, Any]:
    col = get_collection(mongo_db, dataset)

    processed: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []
    received = 0

    for idx, rec in enumerate(raw_records):
        received += 1
        ok, errs = validate_record(dataset, rec)
        if not ok:
            errors.append({"index": idx, "record": rec, "errors": errs})
            continue
        cleaned = preprocess_record(dataset, rec)
        processed.append(cleaned)

    upserted = 0
    inserted = 0
    updated = 0

    for rec in processed:
        key_q = _build_key_query(dataset, rec)
        # Only set non-key fields to avoid conflicts; set key fields only on insert
        non_key_fields = {k: v for k, v in rec.items() if k not in key_q}
        update_doc = {"$setOnInsert": key_q}
        if non_key_fields:
            update_doc["$set"] = non_key_fields
        result = col.update_one(key_q, update_doc, upsert=True)
        if result.upserted_id is not None:
            upserted += 1
            inserted += 1
        elif result.modified_count:
            updated += 1

    return {
        "dataset": dataset,
        "received": received,
        "valid": len(processed),
        "inserted": inserted,
        "updated": updated,
        "errors": errors,
    }
