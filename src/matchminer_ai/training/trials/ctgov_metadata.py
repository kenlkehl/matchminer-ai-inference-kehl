"""ctgov status and dates retained beside trial text, never inferred from it."""


DATE_FIELDS = {
    "study_first_post_date": "studyFirstPostDateStruct",
    "study_start_date": "startDateStruct",
    "primary_completion_date": "primaryCompletionDateStruct",
    "study_completion_date": "completionDateStruct",
    "last_update_post_date": "lastUpdatePostDateStruct",
}

CTGOV_METADATA_COLUMNS = (
    "overall_status",
    "status_verified_date",
    *(column for name in DATE_FIELDS for column in (name, f"{name}_type")),
)


def extract_ctgov_metadata(protocol):
    """Preserve date precision and ACTUAL/ESTIMATED labels; missing values stay blank.

    Completion dates describe data collection, not necessarily accrual closure.
    A month-only date must not acquire an invented day when exported to CSV.
    """
    status = protocol.get("statusModule") or {}
    metadata = {
        "overall_status": status.get("overallStatus") or "",
        "status_verified_date": status.get("statusVerifiedDate") or "",
    }
    for column, source in DATE_FIELDS.items():
        date = status.get(source) or {}
        metadata[column] = date.get("date") or ""
        metadata[f"{column}_type"] = date.get("type") or ""
    return metadata
