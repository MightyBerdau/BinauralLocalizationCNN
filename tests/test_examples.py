"""Tests for the KEMAR example stimuli (``blcnn example``); skipped without slab."""

import os
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

pytest.importorskip("slab")

from blcnn.audio import SAMPLE_RATE, SIGNAL_SAMPLES  # noqa: E402
from blcnn.examples import (EXAMPLE_LOCATIONS, ExamplesError, example_filename,  # noqa: E402
                            make_examples)
from blcnn.predict import predict_folder  # noqa: E402
from blcnn.weights import WeightsError, select_networks  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
WEIGHTS_DIR = Path(os.environ.get("BLCNN_WEIGHTS_DIR", str(REPO / "weights")))
NETS = os.environ.get("BLCNN_TEST_NETS", "1")


@pytest.fixture(scope="module")
def example_dir(tmp_path_factory):
    out = tmp_path_factory.mktemp("examples")
    make_examples(str(out))
    return out


def _rms(signal):
    return np.sqrt(np.mean(signal ** 2, axis=0))


def test_files_written(example_dir):
    expected = sorted(example_filename(az, el) for az, el in EXAMPLE_LOCATIONS)
    assert sorted(os.listdir(str(example_dir))) == expected
    for name in expected:
        signal, rate = sf.read(str(example_dir / name))
        assert rate == SAMPLE_RATE
        assert signal.shape == (SIGNAL_SAMPLES, 2)
        assert 0 < np.max(np.abs(signal)) <= 0.5 + 1e-4


def test_lateral_sources_louder_at_near_ear(example_dir):
    left, _ = sf.read(str(example_dir / example_filename(90, 0)))
    right, _ = sf.read(str(example_dir / example_filename(270, 0)))
    assert _rms(left)[0] > 2 * _rms(left)[1]    # 90 deg = left
    assert _rms(right)[1] > 2 * _rms(right)[0]  # 270 deg = right


def test_deterministic(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    make_examples(str(a), locations=[(30, 0)])
    make_examples(str(b), locations=[(30, 0)])
    name = example_filename(30, 0)
    assert np.array_equal(sf.read(str(a / name))[0], sf.read(str(b / name))[0])


def test_unknown_location(tmp_path):
    with pytest.raises(ExamplesError):
        make_examples(str(tmp_path), locations=[(31, 0)])


def _has_networks():
    try:
        return bool(select_networks(str(WEIGHTS_DIR), NETS))
    except WeightsError:
        return False


@pytest.mark.weights
@pytest.mark.skipif(not _has_networks(), reason="network weights not found in {} (nets {})"
                    .format(WEIGHTS_DIR, NETS))
def test_predicts_true_locations(example_dir, tmp_path):
    """The networks localize the KEMAR examples correctly, front/back and elevation included.

    Also checks the azimuth convention of ``decode_class`` (0 = front, 90 = left).
    Median-plane sources (azimuth 0 or 180) carry no ITD/ILD and single networks
    misjudge them (net1: all three as azimuth 0, elevation 10); they are checked
    only with the full ensemble, which gets all nine right.
    """
    rows = predict_folder(str(example_dir), str(tmp_path / "pred.csv"),
                          weights_dir=str(WEIGHTS_DIR), nets=NETS)
    predicted = {row["file"]: (row["azimuth_deg"], row["elevation_deg"]) for row in rows}
    expected = {example_filename(az, el): (az, el) for az, el in EXAMPLE_LOCATIONS
                if NETS == "all" or az not in (0, 180)}
    assert {name: predicted[name] for name in expected} == expected
