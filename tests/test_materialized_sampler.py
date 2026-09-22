from __future__ import annotations

from rfsep.data.materialized import RecoverableEpochBatchSampler


def test_recoverable_sampler_prefetch_does_not_commit():
    sampler = RecoverableEpochBatchSampler(32, 8, 123)
    iterator = iter(sampler)
    first, second = next(iterator), next(iterator)
    assert sampler.offset == 0
    sampler.commit()
    state = sampler.state_dict()
    resumed = RecoverableEpochBatchSampler(32, 8, 123)
    resumed.load_state_dict(state)
    assert next(iter(resumed)) == second
    assert len(set(first + second)) == 16


def test_recoverable_sampler_epoch_has_no_duplicates():
    sampler = RecoverableEpochBatchSampler(40, 8, 7)
    indices = [index for batch in sampler for index in batch]
    assert sorted(indices) == list(range(40))
    for _ in range(5):
        sampler.commit()
    assert sampler.epoch == 1 and sampler.offset == 0 and sampler.exposures == 40
