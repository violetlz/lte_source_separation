import numpy as np
import pytest

from rfsep.evaluation.metrics import bit_errors


def test_manual_bit_error_count_and_shape_guard():
    truth = np.array([[0, 1, 1], [1, 0, 0]], dtype=np.uint8)
    predicted = truth.copy()
    predicted[0, 1] ^= 1
    predicted[1, 2] ^= 1
    assert bit_errors(predicted, truth) == (2, 6)
    with pytest.raises(ValueError, match="shapes differ"):
        bit_errors(predicted.reshape(-1), truth)
