import pytest
from pydantic import ValidationError

from war_game_engine.contracts import RandomStreamsState, RandomStreamState
from war_game_engine.randomness import (
    BASE_STREAMS,
    InvalidRandomStreamStateError,
    RandomStreams,
)


def test_seed_reproduces_each_named_stream() -> None:
    first = RandomStreams(912345)
    second = RandomStreams(912345)

    assert tuple(first.die("preparation") for _ in range(8)) == tuple(
        second.die("preparation") for _ in range(8)
    )
    assert first.shuffle("cards", tuple(range(20))) == second.shuffle(
        "cards", tuple(range(20))
    )
    assert first.sample("objectives", tuple(range(14)), 6) == second.sample(
        "objectives", tuple(range(14)), 6
    )


def test_named_stream_seed_derivation_has_stable_vectors() -> None:
    streams = RandomStreams(128)

    assert tuple(streams.die(stream_id) for stream_id in BASE_STREAMS) == (
        1,
        3,
        5,
        1,
    )


def test_generator_instances_are_cached_per_stream_id() -> None:
    streams = RandomStreams(42)

    first = streams.generator("combat")
    first.integers(0, 100)
    second = streams.generator("combat")

    assert second is first


def test_drawing_from_one_stream_does_not_advance_another() -> None:
    untouched = RandomStreams(42)
    used = RandomStreams(42)
    expected_preparation_rolls = tuple(untouched.die("preparation") for _ in range(5))

    for _ in range(30):
        used.die("combat")
        used.generator("cards").random()

    actual_preparation_rolls = tuple(used.die("preparation") for _ in range(5))

    assert actual_preparation_rolls == expected_preparation_rolls


def test_policy_streams_are_separate_and_reproducible() -> None:
    first = RandomStreams(7)
    second = RandomStreams(7)

    assert tuple(
        int(first.policy_generator("policy-a").integers(0, 1000)) for _ in range(5)
    ) == tuple(
        int(second.policy_generator("policy-a").integers(0, 1000)) for _ in range(5)
    )
    first.policy_generator("policy-b").integers(0, 1000)
    assert int(first.policy_generator("policy-a").integers(0, 1000)) == int(
        second.policy_generator("policy-a").integers(0, 1000)
    )
    with pytest.raises(ValueError, match="Policy ID must not be blank"):
        first.policy_generator(" ")

    with pytest.raises(ValueError, match="base stream or a non-empty policy"):
        first.generator("unknown-stream")


def test_random_state_restores_every_stream_and_future_policy_streams() -> None:
    streams = RandomStreams(2**130 + 99)
    streams.die("preparation")
    streams.sample("objectives", tuple(range(14)), 3)
    streams.shuffle("cards", tuple(range(44)))
    streams.die("combat")
    streams.policy_generator("policy-a").integers(0, 10)
    saved = streams.snapshot()
    expected = {
        stream_id: int(streams.generator(stream_id).integers(0, 2**31))
        for stream_id in (*BASE_STREAMS, "policy:policy-a", "policy:policy-b")
    }

    restored = RandomStreams.restore(saved)
    actual = {
        stream_id: int(restored.generator(stream_id).integers(0, 2**31))
        for stream_id in (*BASE_STREAMS, "policy:policy-a", "policy:policy-b")
    }

    assert actual == expected
    assert saved.root_seed == 2**130 + 99


def test_random_stream_state_rejects_missing_or_unsupported_streams() -> None:
    with pytest.raises(ValidationError, match="Missing required random streams"):
        RandomStreamsState(
            schema_version="1.0",
            algorithm="PCG64",
            root_seed=1,
            streams=(
                RandomStreamState(
                    stream_id="preparation",
                    bit_generator_state={},
                ),
                RandomStreamState(stream_id="objectives", bit_generator_state={}),
                RandomStreamState(stream_id="cards", bit_generator_state={}),
                RandomStreamState(stream_id="other", bit_generator_state={}),
            ),
        )


def test_corrupted_generator_state_fails_explicitly() -> None:
    saved = RandomStreams(4).snapshot()
    corrupted = saved.model_dump(mode="python")
    streams = list(corrupted["streams"])
    streams[0]["bit_generator_state"] = {"bit_generator": "not-pcg64"}
    corrupted["streams"] = tuple(streams)
    invalid_state = RandomStreamsState.model_validate(corrupted)

    with pytest.raises(InvalidRandomStreamStateError, match="Invalid state"):
        RandomStreams.restore(invalid_state)
