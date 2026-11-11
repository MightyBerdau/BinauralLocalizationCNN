"""Equivalence test: blcnn vs. the original (legacy/) testing code.

The same preprocessed subbands are fed through
  (a) blcnn's placeholder-based graph (blcnn.model.LocalizationNetwork) and
  (b) the original TFRecord pipeline (legacy/call_model_training_valid_pad_francl.py),
and their 504-way output probabilities are compared.

Needs the network weights; skipped otherwise. Configure with environment variables:
  BLCNN_WEIGHTS_DIR  folder with net1 ... net10 (default: <repo>/weights)
  BLCNN_TEST_NETS    networks to compare, e.g. "1,3" or "all" (default: "1")

The legacy pipeline always mixes a background record into every stimulus; blcnn
does not. The background must therefore be made truly negligible: the model's
compression (x ** 0.3) strongly amplifies values near zero, and the subbands
contain many exact zeros (-60 dB cutoff). At 150 dB SNR the scaled background
(~1e-8) still changes probabilities by ~5e-4; at 500 dB (~1e-26) the remaining
difference is float32 rounding (~1e-6).
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from blcnn.audio import SAMPLE_RATE, SIGNAL_SAMPLES, prepare_signal
from blcnn.cochleagram import INPUT_SHAPE, model_input
from blcnn.weights import WeightsError, select_networks

REPO = Path(__file__).resolve().parents[1]
WEIGHTS_DIR = Path(os.environ.get("BLCNN_WEIGHTS_DIR", str(REPO / "weights")))
NETS = os.environ.get("BLCNN_TEST_NETS", "1")
ANGLES = [-90, -60, -30, -10, 10, 30, 60, 90]  # lateral angle, positive = right
SNR_DB = 500  # background scaled to ~1e-26 of the stimulus; see module docstring
PROB_TOLERANCE = 1e-4  # max. absolute difference of any class probability


def _networks():
    try:
        return select_networks(str(WEIGHTS_DIR), NETS)
    except WeightsError:
        return []


pytestmark = [
    pytest.mark.weights,
    pytest.mark.skipif(not _networks(), reason="network weights not found in {} (nets {})"
                       .format(WEIGHTS_DIR, NETS)),
]


def _binaural_noise(angle, rng):
    """0.5-s noise burst in a 2-s window with a spherical-head ITD/ILD."""
    mono = np.zeros(SIGNAL_SAMPLES)
    start, length, ramp = SIGNAL_SAMPLES // 2 - SAMPLE_RATE // 4, SAMPLE_RATE // 2, 480
    envelope = np.ones(length)
    envelope[:ramp], envelope[-ramp:] = np.hanning(2 * ramp)[:ramp], np.hanning(2 * ramp)[ramp:]
    mono[start:start + length] = rng.randn(length) * envelope
    theta = np.deg2rad(abs(angle))
    itd = 0.0875 / 343.0 * (theta + np.sin(theta))
    freqs = np.fft.rfftfreq(SIGNAL_SAMPLES, 1.0 / SAMPLE_RATE)
    gain = np.where(freqs > 1500, 10 ** (-10 * np.sin(theta) / 20), 1.0)
    far = np.fft.irfft(np.fft.rfft(mono) * np.exp(-2j * np.pi * freqs * itd) * gain,
                       SIGNAL_SAMPLES)
    return 0.05 * np.stack([far, mono] if angle > 0 else [mono, far], axis=1)


def _example(image, idx, azimuth):
    import tensorflow as tf

    def int64(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[int(value)]))

    return tf.train.Example(features=tf.train.Features(feature={
        "train/image": tf.train.Feature(bytes_list=tf.train.BytesList(value=[image.tobytes()])),
        "train/image_height": int64(image.shape[0]),
        "train/image_width": int64(image.shape[1]),
        "train/azim": int64(azimuth),
        "train/elev": int64(0),
        "train/file_idx": int64(idx),
    })).SerializeToString()


def _write_records(path, examples):
    import tensorflow as tf
    path.parent.mkdir(parents=True, exist_ok=True)
    options = tf.io.TFRecordOptions(tf.compat.v1.io.TFRecordCompressionType.GZIP)
    with tf.io.TFRecordWriter(str(path), options) as writer:
        for example in examples:
            writer.write(example)


def _link_or_copy_network(network, target_root):
    """Mirror a network folder into ``target_root`` (hard links if possible), so the
    legacy code writes its result files there instead of into the weights folder."""
    target = target_root / Path(network.folder).name
    target.mkdir(parents=True)
    for name in os.listdir(network.folder):
        source = os.path.join(network.folder, name)
        if not (name == "config_array.npy" or name.startswith("model.ckpt-")):
            continue
        try:
            os.link(source, str(target / name))
        except OSError:
            shutil.copy2(source, str(target / name))
    return target


@pytest.fixture(scope="module")
def stimuli(tmp_path_factory):
    """Model inputs of the test stimuli, produced by blcnn's preprocessing from .wav files."""
    folder = tmp_path_factory.mktemp("wavs")
    rng = np.random.RandomState(0)
    inputs = []
    for angle in ANGLES:
        path = folder / "noise_{:+03d}.wav".format(angle)
        sf.write(str(path), _binaural_noise(angle, rng), SAMPLE_RATE, subtype="FLOAT")
        signal, warnings = prepare_signal(str(path))
        assert warnings == []
        inputs.append(model_input(signal))
    return np.stack(inputs)


@pytest.fixture(scope="module")
def workdir():
    # on the weights' drive, so checkpoints can be hard-linked instead of copied
    path = Path(tempfile.mkdtemp(prefix=".blcnn_equivalence_", dir=str(WEIGHTS_DIR.parent)))
    yield path
    shutil.rmtree(str(path), ignore_errors=True)


def _legacy_probabilities(network, stimuli, workdir):
    """Run the original testing code; return probabilities ordered by stimulus index."""
    weights_copy = _link_or_copy_network(network, workdir / "weights")
    stim_dir, bkgd_dir = workdir / "records" / "stimuli", workdir / "records" / "bkgd"
    if not (stim_dir / "train0.tfrecords").exists():
        _write_records(stim_dir / "train0.tfrecords",
                       [_example(image, i, ANGLES[i] % 360) for i, image in enumerate(stimuli)])
        silence = np.zeros(INPUT_SHAPE, np.float32)
        silence[:, ::4800, :] = 1e-3  # non-zero RMS; scaled to -SNR_DB re. the stimulus
        _write_records(bkgd_dir / "train0.tfrecords",
                       [_example(silence, 0, 0) for _ in range(len(stimuli))])

    step = network.checkpoint.rsplit("-", 1)[1]
    as_posix = lambda p: str(p).replace(os.sep, "/")  # the legacy code splits paths on "/"
    command = [
        sys.executable, "-u", "legacy/call_model_training_valid_pad_francl.py",
        str(network.number), "0", "None",
        as_posix(bkgd_dir / "train*.tfrecords"), as_posix(stim_dir / "train*.tfrecords"),
        step, as_posix(workdir / "weights"), str(SNR_DB), str(SNR_DB),
        "False", "False", "False", "False", "False", "False", "True", "False", "True", "True",
    ]
    env = dict(os.environ, BATCH_SIZE="2", TF_CPP_MIN_LOG_LEVEL="2")
    result = subprocess.run(command, cwd=str(REPO), env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, universal_newlines=True)
    if result.returncode != 0 and ("OOM" in result.stdout or "ResourceExhausted" in result.stdout
                                   or result.returncode == 0xC0000409):
        # The original graph always builds the Adam training step and initializes its
        # two slot variables per weight (~3x the checkpoint size in memory), which does
        # not fit for the large networks (net4/5/7/10, ~3.5 GB) on a 16 GB machine.
        pytest.skip("legacy pipeline ran out of memory for net{} (exit code {})"
                    .format(network.number, result.returncode))
    assert result.returncode == 0, result.stdout[-3000:]

    out_dir = weights_copy
    data = np.load(str(out_dir / "batch_conditional_stimuli_iter{}.npy".format(step)),
                   allow_pickle=True)
    keys = json.loads((out_dir / "keys_test_stimuli_iter{}.json".format(step)).read_text())
    idx_pos = keys.index("train/file_idx")
    probabilities = np.full((len(stimuli), 504), np.nan)
    for probs, labels in data:
        probabilities[int(np.squeeze(labels[idx_pos]))] = probs
    assert not np.isnan(probabilities).any(), "legacy output is missing stimuli"
    return probabilities


@pytest.mark.parametrize("network", _networks(), ids=lambda n: "net{}".format(n.number))
def test_blcnn_matches_legacy(network, stimuli, workdir):
    from blcnn.model import LocalizationNetwork

    with LocalizationNetwork(network, device="cpu") as model:
        new = np.concatenate([model.predict(stimuli[i:i + 1]) for i in range(len(stimuli))])
    legacy = _legacy_probabilities(network, stimuli, workdir)

    diff = np.abs(new - legacy).max()
    print("net{}: max |p_blcnn - p_legacy| = {:.2e}".format(network.number, diff))
    np.testing.assert_array_equal(new.argmax(axis=1), legacy.argmax(axis=1))
    assert diff < PROB_TOLERANCE
