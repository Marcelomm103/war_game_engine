from datetime import datetime
from typing import Annotated, Final, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field

CONTRACT_SCHEMA_VERSION: Final = "1.0"
API_VERSION: Final = "1.0.0"
RULES_VERSION: Final = "1.0.0"
OBSERVATION_SCHEMA_VERSION: Final = "1.0.0"
OBSERVATION_ENCODING_VERSION: Final = "1.0.0"
ACTION_SCHEMA_VERSION: Final = "1.0.0"
ACTION_ENCODING_VERSION: Final = "1.0.0"
REPLAY_VERSION: Final = "1.0.0"
DATASET_VERSION: Final = "1.0.0"
MODEL_VERSION: Final = "1.0.0"
INFERENCE_CONFIG_VERSION: Final = "1.0.0"
GEOMETRY_VERSION: Final = "1.0.0"

ContractSchemaVersionLiteral: TypeAlias = Literal["1.0"]
ApiVersionLiteral: TypeAlias = Literal["1.0.0"]
ObservationSchemaVersionLiteral: TypeAlias = Literal["1.0.0"]
ActionSchemaVersionLiteral: TypeAlias = Literal["1.0.0"]

SemanticVersion = Annotated[
    str,
    Field(pattern=r"^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$"),
]
VersionDimension = Literal[
    "rules_version",
    "api_version",
    "contract_schema_version",
    "observation_schema_version",
    "observation_encoding_version",
    "action_schema_version",
    "action_encoding_version",
    "replay_version",
    "dataset_version",
    "model_version",
    "inference_config_version",
    "geometry_version",
]
CertificateScope = Literal["engine", "policy", "client", "replay", "dataset"]


class VersionModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class SemanticVersionVector(VersionModel):
    contract_type: Literal["semantic_version_vector"]
    rules_version: SemanticVersion
    api_version: SemanticVersion
    contract_schema_version: str
    observation_schema_version: SemanticVersion
    observation_encoding_version: SemanticVersion
    action_schema_version: SemanticVersion
    action_encoding_version: SemanticVersion
    replay_version: SemanticVersion
    dataset_version: SemanticVersion
    model_version: SemanticVersion
    inference_config_version: SemanticVersion
    geometry_version: SemanticVersion

    def value_for(self, dimension: VersionDimension) -> str:
        return {
            "rules_version": self.rules_version,
            "api_version": self.api_version,
            "contract_schema_version": self.contract_schema_version,
            "observation_schema_version": self.observation_schema_version,
            "observation_encoding_version": self.observation_encoding_version,
            "action_schema_version": self.action_schema_version,
            "action_encoding_version": self.action_encoding_version,
            "replay_version": self.replay_version,
            "dataset_version": self.dataset_version,
            "model_version": self.model_version,
            "inference_config_version": self.inference_config_version,
            "geometry_version": self.geometry_version,
        }[dimension]


class CertificateManifest(VersionModel):
    contract_type: Literal["certificate_manifest"]
    certificate_id: str
    scope: CertificateScope
    criterion_id: str
    issued_at: datetime
    artifact_sha256: Annotated[str, Field(pattern=r"^sha256:[0-9a-f]{64}$")]
    versions: SemanticVersionVector


CURRENT_VERSION_VECTOR = SemanticVersionVector(
    contract_type="semantic_version_vector",
    rules_version=RULES_VERSION,
    api_version=API_VERSION,
    contract_schema_version=CONTRACT_SCHEMA_VERSION,
    observation_schema_version=OBSERVATION_SCHEMA_VERSION,
    observation_encoding_version=OBSERVATION_ENCODING_VERSION,
    action_schema_version=ACTION_SCHEMA_VERSION,
    action_encoding_version=ACTION_ENCODING_VERSION,
    replay_version=REPLAY_VERSION,
    dataset_version=DATASET_VERSION,
    model_version=MODEL_VERSION,
    inference_config_version=INFERENCE_CONFIG_VERSION,
    geometry_version=GEOMETRY_VERSION,
)

CERTIFICATE_DEPENDENCIES: dict[CertificateScope, tuple[VersionDimension, ...]] = {
    "engine": ("rules_version",),
    "policy": (
        "rules_version",
        "observation_schema_version",
        "observation_encoding_version",
        "action_schema_version",
        "action_encoding_version",
        "replay_version",
        "dataset_version",
        "model_version",
        "inference_config_version",
    ),
    "client": (
        "rules_version",
        "api_version",
        "contract_schema_version",
        "observation_schema_version",
        "action_schema_version",
        "geometry_version",
    ),
    "replay": (
        "rules_version",
        "api_version",
        "contract_schema_version",
        "action_schema_version",
        "replay_version",
    ),
    "dataset": (
        "rules_version",
        "contract_schema_version",
        "observation_schema_version",
        "observation_encoding_version",
        "action_schema_version",
        "action_encoding_version",
        "replay_version",
        "dataset_version",
    ),
}


def certificate_invalidation_reasons(
    certificate: CertificateManifest,
    current: SemanticVersionVector,
) -> tuple[VersionDimension, ...]:
    dependencies = CERTIFICATE_DEPENDENCIES[certificate.scope]
    return tuple(
        dimension
        for dimension in dependencies
        if certificate.versions.value_for(dimension) != current.value_for(dimension)
    )
