"""Deterministic, isolated random streams for game and policy decisions."""

from __future__ import annotations

import copy
import hashlib
from collections.abc import Sequence
from typing import TypeVar

import numpy as np

from .contracts import RandomStreamsState, RandomStreamState

T = TypeVar("T")
BASE_STREAMS = ("preparation", "objectives", "cards", "combat")


class InvalidRandomStreamStateError(ValueError):
    """Raised when a stored generator state cannot be restored."""


class RandomStreams:
    def __init__(self, root_seed: int) -> None:
        if root_seed < 0:
            raise ValueError("Random root seed must be non-negative.")
        self.root_seed = root_seed
        self._generators: dict[str, np.random.Generator] = {}
        for stream_id in BASE_STREAMS:
            self.generator(stream_id)

    @staticmethod
    def _validate_stream_id(stream_id: str) -> None:
        if stream_id in BASE_STREAMS:
            return
        if (
            stream_id.startswith("policy:")
            and stream_id.removeprefix("policy:").strip()
        ):
            return
        raise ValueError(
            "Random stream IDs must be a base stream or a non-empty policy stream."
        )

    def _derive_seed(self, stream_id: str) -> int:
        seed_bytes = self.root_seed.to_bytes(
            max(1, (self.root_seed.bit_length() + 7) // 8),
            byteorder="big",
        )
        digest = hashlib.sha256(
            b"war-random-streams-v1\0" + seed_bytes + b"\0" + stream_id.encode("utf-8")
        ).digest()
        return int.from_bytes(digest[:16], byteorder="big")

    def generator(self, stream_id: str) -> np.random.Generator:
        self._validate_stream_id(stream_id)
        generator = self._generators.get(stream_id)
        if generator is None:
            generator = np.random.Generator(
                np.random.PCG64(self._derive_seed(stream_id))
            )
            self._generators[stream_id] = generator
        return generator

    def policy_generator(self, policy_id: str) -> np.random.Generator:
        if not policy_id.strip():
            raise ValueError("Policy ID must not be blank.")
        return self.generator(f"policy:{policy_id}")

    def die(self, stream_id: str, sides: int = 6) -> int:
        if sides <= 0:
            raise ValueError("A die must have at least one side.")
        return int(self.generator(stream_id).integers(1, sides + 1))

    def shuffle(self, stream_id: str, values: Sequence[T]) -> tuple[T, ...]:
        indices = self.generator(stream_id).permutation(len(values))
        return tuple(values[int(index)] for index in indices)

    def sample(self, stream_id: str, values: Sequence[T], count: int) -> tuple[T, ...]:
        if count < 0 or count > len(values):
            raise ValueError("Sample size must be between zero and population size.")
        if count == 0:
            return ()
        indices = self.generator(stream_id).choice(
            len(values), size=count, replace=False
        )
        return tuple(values[int(index)] for index in indices)

    def snapshot(self) -> RandomStreamsState:
        return RandomStreamsState(
            schema_version="1.0",
            algorithm="PCG64",
            root_seed=self.root_seed,
            streams=tuple(
                RandomStreamState(
                    stream_id=stream_id,
                    bit_generator_state=copy.deepcopy(
                        dict(generator.bit_generator.state)
                    ),
                )
                for stream_id, generator in sorted(self._generators.items())
            ),
        )

    @classmethod
    def restore(cls, state: RandomStreamsState) -> RandomStreams:
        if state.algorithm != "PCG64":
            raise InvalidRandomStreamStateError(
                f"Unsupported random algorithm {state.algorithm!r}."
            )
        streams = cls(state.root_seed)
        for saved_stream in state.streams:
            generator = streams.generator(saved_stream.stream_id)
            try:
                generator.bit_generator.state = copy.deepcopy(
                    saved_stream.bit_generator_state
                )
            except (TypeError, ValueError) as error:
                raise InvalidRandomStreamStateError(
                    f"Invalid state for random stream {saved_stream.stream_id!r}."
                ) from error
        return streams


__all__ = [
    "BASE_STREAMS",
    "InvalidRandomStreamStateError",
    "RandomStreams",
]
