from __future__ import annotations


def get_records(payload: dict) -> list[dict]:
    records = payload.get("Records")
    if isinstance(records, list):
        return records
    if isinstance(payload.get("s3"), dict):
        return [payload]
    return []


def get_event_name(record: dict, payload: dict) -> str:
    return (
        record.get("eventName")
        or payload.get("EventName")
        or payload.get("eventName")
        or ""
    )


def partition_records(payload: dict) -> tuple[list[dict], list[dict]]:
    created_records = []
    removed_records = []

    for record in get_records(payload):
        event_name = get_event_name(record, payload)
        if event_name.startswith("s3:ObjectCreated:"):
            created_records.append(record)
        elif event_name.startswith("s3:ObjectRemoved:"):
            removed_records.append(record)

    return created_records, removed_records
