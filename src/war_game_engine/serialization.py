"""Canonical JSON serialization and hashes for engine state."""

from __future__ import annotations

import hashlib
import json

from pydantic import BaseModel, JsonValue


def canonical_json_bytes(value: BaseModel | JsonValue) -> bytes:
    payload: JsonValue
    if isinstance(value, BaseModel):
        payload = value.model_dump(mode="json")
    else:
        payload = value
    return json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def canonical_sha256(value: BaseModel | JsonValue) -> str:
    digest = hashlib.sha256(canonical_json_bytes(value)).hexdigest()
    return f"sha256:{digest}"


__all__ = ["canonical_json_bytes", "canonical_sha256"]
