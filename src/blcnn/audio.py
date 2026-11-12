"""Load binaural audio and bring it into the form the model was tested with.

Follows the test procedure of Francl & McDermott (2022, Methods): 2-s stereo
signals at 48 kHz, normalized to an RMS amplitude of 0.1.
"""

from math import gcd

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

SAMPLE_RATE = 48000          # Hz, sample rate of the cochlear model input
SIGNAL_SECONDS = 2.0         # length of each stimulus before the cochlear model
SIGNAL_SAMPLES = int(SAMPLE_RATE * SIGNAL_SECONDS)
TARGET_RMS = 0.1


class AudioError(ValueError):
    """A file cannot be used as model input (e.g. not stereo, silent)."""


def load_audio(path):
    """Read an audio file with at least 2 channels.

    Returns ``(signal, sample_rate)`` with ``signal`` as float64 array of shape
    ``(n_samples, n_channels)`` (column 0 = left ear, column 1 = right ear; any
    further channels are returned unchanged, see ``prepare_signal``).
    """
    try:
        signal, sample_rate = sf.read(str(path), dtype="float64", always_2d=True)
    except RuntimeError as err:  # libsndfile errors: unreadable / unsupported file
        # libsndfile messages read "Error opening '<path>': <reason>"; keep the reason
        reason = str(err).rsplit(": ", 1)[-1].rstrip(".")
        raise AudioError("cannot read audio file ({})".format(reason))
    if signal.shape[1] < 2:
        raise AudioError("expected at least 2 channels (binaural left/right), found {}"
                         .format(signal.shape[1]))
    if signal.shape[0] == 0:
        raise AudioError("file contains no samples")
    return signal, sample_rate


def resample(signal, sample_rate, target_rate=SAMPLE_RATE):
    """Resample ``signal`` (n_samples, channels) to ``target_rate``."""
    if sample_rate == target_rate:
        return signal
    divisor = gcd(int(sample_rate), int(target_rate))
    return resample_poly(signal, target_rate // divisor, int(sample_rate) // divisor, axis=0)


def fit_length(signal, n_samples=SIGNAL_SAMPLES):
    """Crop to the first ``n_samples`` or zero-pad symmetrically to ``n_samples``.

    Returns ``(signal, action)`` with ``action`` one of ``None``, ``"cropped"``,
    ``"padded"``.
    """
    length = signal.shape[0]
    if length > n_samples:
        return signal[:n_samples], "cropped"
    if length < n_samples:
        before = (n_samples - length) // 2
        after = n_samples - length - before
        return np.pad(signal, ((before, after), (0, 0))), "padded"
    return signal, None


def normalize_rms(signal, target_rms=TARGET_RMS):
    """Scale both ears by one common factor so the overall RMS is ``target_rms``.

    A common factor keeps the interaural level difference intact.
    """
    rms = np.sqrt(np.mean(np.square(signal)))
    if rms == 0:
        raise AudioError("signal is silent (RMS = 0)")
    return signal * (target_rms / rms)


def prepare_signal(path):
    """Load ``path`` and return ``(signal, warnings)``.

    ``signal`` has shape (96000, 2): 2 s at 48 kHz, RMS 0.1. ``warnings`` lists
    deviations from the recommended input (2-s stereo file at 48 kHz).
    """
    warnings = []
    signal, sample_rate = load_audio(path)
    if signal.shape[1] > 2:
        warnings.append("{} channels; used only the first 2 (left/right)"
                        .format(signal.shape[1]))
        signal = signal[:, :2]
    if sample_rate != SAMPLE_RATE:
        warnings.append("resampled from {} Hz to {} Hz".format(sample_rate, SAMPLE_RATE))
        signal = resample(signal, sample_rate)
    duration = signal.shape[0] / SAMPLE_RATE
    signal, action = fit_length(signal)
    if action == "cropped":
        warnings.append("longer than 2 s ({:.3f} s); used the first 2 s".format(duration))
    elif action == "padded":
        warnings.append("shorter than 2 s ({:.3f} s); zero-padded to 2 s".format(duration))
    return normalize_rms(signal), warnings
