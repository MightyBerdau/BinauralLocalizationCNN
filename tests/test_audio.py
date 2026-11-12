import numpy as np
import pytest
import soundfile as sf

from blcnn.audio import (SAMPLE_RATE, SIGNAL_SAMPLES, TARGET_RMS, AudioError, fit_length,
                         load_audio, normalize_rms, prepare_signal, resample)


def _write(path, signal, sample_rate=SAMPLE_RATE, subtype=None):
    sf.write(str(path), signal, sample_rate, subtype=subtype)
    return path


def _noise(n, channels=2, seed=0):
    return 0.05 * np.random.RandomState(seed).randn(n, channels)


def test_load_audio_stereo_24bit(tmp_path):
    path = _write(tmp_path / "a.wav", _noise(1000), subtype="PCM_24")
    signal, rate = load_audio(path)
    assert signal.shape == (1000, 2) and rate == SAMPLE_RATE
    assert signal.dtype == np.float64


def test_load_audio_rejects_mono(tmp_path):
    path = _write(tmp_path / "mono.wav", _noise(1000, channels=1)[:, 0])
    with pytest.raises(AudioError, match="expected at least 2 channels"):
        load_audio(path)


def test_prepare_signal_uses_first_two_of_many_channels(tmp_path):
    multi = _noise(SIGNAL_SAMPLES, channels=4)
    signal, warnings = prepare_signal(_write(tmp_path / "multi.wav", multi))
    assert signal.shape == (SIGNAL_SAMPLES, 2)
    assert len(warnings) == 1 and "4 channels" in warnings[0]
    np.testing.assert_allclose(signal, normalize_rms(multi[:, :2]), atol=1e-4)


def test_load_audio_rejects_unreadable(tmp_path):
    path = tmp_path / "broken.wav"
    path.write_bytes(b"not audio")
    with pytest.raises(AudioError, match="cannot read audio file"):
        load_audio(path)


def test_resample_length():
    signal = _noise(44100)
    assert resample(signal, 44100).shape == (SAMPLE_RATE, 2)
    assert resample(signal, SAMPLE_RATE) is signal


def test_fit_length_crops_to_first_samples():
    signal = np.arange(2 * (SIGNAL_SAMPLES + 10)).reshape(-1, 2).astype(float)
    out, action = fit_length(signal)
    assert action == "cropped"
    np.testing.assert_array_equal(out, signal[:SIGNAL_SAMPLES])


def test_fit_length_pads_symmetrically():
    signal = np.ones((10, 2))
    out, action = fit_length(signal, n_samples=20)
    assert action == "padded" and out.shape == (20, 2)
    assert out[:5].sum() == 0 and out[15:].sum() == 0 and out[5:15].sum() == 20


def test_normalize_rms_keeps_level_difference():
    signal = np.stack([np.ones(100), 0.5 * np.ones(100)], axis=1)
    out = normalize_rms(signal)
    assert np.isclose(np.sqrt(np.mean(out ** 2)), TARGET_RMS)
    assert np.allclose(out[:, 0] / out[:, 1], 2.0)


def test_normalize_rms_rejects_silence():
    with pytest.raises(AudioError, match="silent"):
        normalize_rms(np.zeros((100, 2)))


def test_prepare_signal_warnings(tmp_path):
    exact = prepare_signal(_write(tmp_path / "ok.wav", _noise(SIGNAL_SAMPLES)))
    assert exact[0].shape == (SIGNAL_SAMPLES, 2) and exact[1] == []

    long_signal, warnings = prepare_signal(_write(tmp_path / "long.wav", _noise(3 * SAMPLE_RATE)))
    assert long_signal.shape == (SIGNAL_SAMPLES, 2)
    assert len(warnings) == 1 and "first 2 s" in warnings[0]

    _, warnings = prepare_signal(_write(tmp_path / "short.wav", _noise(44100), 44100))
    assert len(warnings) == 2
    assert "resampled" in warnings[0] and "zero-padded" in warnings[1]
