from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from war_game_engine.versioning import (
    CURRENT_VERSION_VECTOR,
    CertificateManifest,
    CertificateScope,
    SemanticVersionVector,
    certificate_invalidation_reasons,
)


def make_certificate(scope: CertificateScope) -> CertificateManifest:
    return CertificateManifest(
        contract_type="certificate_manifest",
        certificate_id=f"certificate-{scope}",
        scope=scope,
        criterion_id="criterion-1",
        issued_at=datetime(2026, 10, 21, tzinfo=timezone.utc),
        artifact_sha256=f"sha256:{'a' * 64}",
        versions=CURRENT_VERSION_VECTOR,
    )


def test_rules_change_invalidates_engine_and_policy_certificates() -> None:
    changed = CURRENT_VERSION_VECTOR.model_copy(
        update={"rules_version": "1.0.1"}
    )

    assert certificate_invalidation_reasons(make_certificate("engine"), changed) == (
        "rules_version",
    )
    assert certificate_invalidation_reasons(make_certificate("policy"), changed) == (
        "rules_version",
    )


def test_observation_and_action_encoding_changes_invalidate_policy_certificates() -> None:
    changed = CURRENT_VERSION_VECTOR.model_copy(
        update={
            "observation_encoding_version": "1.0.1",
            "action_encoding_version": "1.0.1",
        }
    )

    assert certificate_invalidation_reasons(make_certificate("policy"), changed) == (
        "observation_encoding_version",
        "action_encoding_version",
    )
    assert certificate_invalidation_reasons(make_certificate("engine"), changed) == ()
    assert certificate_invalidation_reasons(make_certificate("dataset"), changed) == (
        "observation_encoding_version",
        "action_encoding_version",
    )


def test_api_and_geometry_changes_invalidate_client_certificates() -> None:
    changed = CURRENT_VERSION_VECTOR.model_copy(
        update={
            "api_version": "1.0.1",
            "geometry_version": "1.0.1",
        }
    )

    assert certificate_invalidation_reasons(make_certificate("client"), changed) == (
        "api_version",
        "geometry_version",
    )


def test_unrelated_model_change_does_not_invalidate_engine_certificate() -> None:
    changed = CURRENT_VERSION_VECTOR.model_copy(update={"model_version": "1.0.1"})

    assert certificate_invalidation_reasons(make_certificate("engine"), changed) == ()
    assert certificate_invalidation_reasons(make_certificate("policy"), changed) == (
        "model_version",
    )


def test_replay_and_dataset_versions_invalidate_related_certificates() -> None:
    changed = CURRENT_VERSION_VECTOR.model_copy(
        update={
            "replay_version": "1.0.1",
            "dataset_version": "1.0.1",
        }
    )

    assert certificate_invalidation_reasons(make_certificate("policy"), changed) == (
        "replay_version",
        "dataset_version",
    )
    assert certificate_invalidation_reasons(make_certificate("replay"), changed) == (
        "replay_version",
    )
    assert certificate_invalidation_reasons(make_certificate("dataset"), changed) == (
        "replay_version",
        "dataset_version",
    )


def test_inference_config_version_invalidates_policy_certificates() -> None:
    changed = CURRENT_VERSION_VECTOR.model_copy(
        update={"inference_config_version": "1.0.1"}
    )

    assert certificate_invalidation_reasons(make_certificate("policy"), changed) == (
        "inference_config_version",
    )
    assert certificate_invalidation_reasons(make_certificate("engine"), changed) == ()


def test_contract_schema_change_invalidates_transport_certificates() -> None:
    changed = CURRENT_VERSION_VECTOR.model_copy(
        update={"contract_schema_version": "2.0"}
    )

    assert certificate_invalidation_reasons(make_certificate("client"), changed) == (
        "contract_schema_version",
    )
    assert certificate_invalidation_reasons(make_certificate("replay"), changed) == (
        "contract_schema_version",
    )
    assert certificate_invalidation_reasons(make_certificate("dataset"), changed) == (
        "contract_schema_version",
    )
    assert certificate_invalidation_reasons(make_certificate("engine"), changed) == ()


def test_semantic_versions_require_major_minor_patch() -> None:
    invalid_vector = CURRENT_VERSION_VECTOR.model_dump()
    invalid_vector["model_version"] = "1.0"

    with pytest.raises(ValidationError):
        SemanticVersionVector.model_validate(invalid_vector)
