import numpy as np
import pytest

from blcnn.audio import SIGNAL_SAMPLES, normalize_rms
from blcnn.cochleagram import CUTOFF_RATIO, INPUT_SHAPE, model_input


@pytest.fixture(scope="module")
def subbands():
    signal = normalize_rms(np.random.RandomState(0).randn(SIGNAL_SAMPLES, 2))
    return model_input(signal)


def test_shape_and_dtype(subbands):
    assert subbands.shape == INPUT_SHAPE == (39, 48000, 2)
    assert subbands.dtype == np.float32


def test_subbands_are_signed(subbands):
    # rectification happens later, inside the TensorFlow graph
    assert subbands.min() < 0 < subbands.max()


def test_cutoff_below_minus_60_db(subbands):
    peak = np.abs(subbands).max()
    nonzero = np.abs(subbands[subbands != 0])
    assert nonzero.min() >= peak / CUTOFF_RATIO


def test_uses_middle_second():
    # sound only in the first 0.5 s: outside the middle second -> no input energy
    signal = np.zeros((SIGNAL_SAMPLES, 2))
    signal[:12000] = np.random.RandomState(1).randn(12000, 2)
    early = model_input(signal)
    signal_mid = np.roll(signal, SIGNAL_SAMPLES // 2, axis=0)
    middle = model_input(signal_mid)
    assert np.abs(early).max() < 0.01 * np.abs(middle).max()


def test_rejects_wrong_shape():
    with pytest.raises(ValueError):
        model_input(np.zeros((1000, 2)))
