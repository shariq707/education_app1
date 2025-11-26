from typing import Dict, List

# Required fields per dataset type (CSV headers / JSON keys)
REQUIRED_FIELDS: Dict[str, List[str]] = {
    "academic_records": ["student_id", "course_code", "term", "grade"],
    "demographics": ["student_id", "first_name", "last_name", "dob", "gender"],
    "lms": ["event_id", "student_id", "course_code", "event_type", "event_time"],
    "attendance": ["student_id", "course_code", "date", "status"],
}

# Optional fields per dataset type (helps with type coercion and cleaning)
OPTIONAL_FIELDS: Dict[str, List[str]] = {
    "academic_records": ["credits", "instructor", "remarks"],
    "demographics": ["email", "phone", "address"],
    "lms": ["resource_id", "details"],
    "attendance": ["remarks"],
}
