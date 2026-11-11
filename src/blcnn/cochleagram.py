"""Cochlear filterbank front end: 2-s stereo signal -> model input subbands.

Reproduces ``cochleagram_wrapper()`` of the authors' BinauralDataGen
(https://github.com/afrancl/BinauralDataGen): 39 ERB-spaced bandpass subbands
between 30 Hz and 20 kHz, followed by the paper's test procedure of keeping the
middle 1 s. Rectification, downsampling to 8 kHz and compression happen inside
the TensorFlow graph (see ``model.py``), as in the original code.
"""

import numpy as np

from ._vendor.pycochleagram import cochleagram as cgm
from .audio import SAMPLE_RATE, SIGNAL_SAMPLES

LOW_LIM = 30          # Hz
HI_LIM = 20000        # Hz
PADDING_SIZE = 10000  # samples of zero padding during filtering (BinauralDataGen)
N_CHANNELS = 39
SEGMENT_SAMPLES = SAMPLE_RATE  # 1 s
SEGMENT_START = (SIGNAL_SAMPLES - SEGMENT_SAMPLES) // 2  # middle 1 s of 2 s
CUTOFF_RATIO = 1000.0  # values more than 60 dB below the peak are set to 0
INPUT_SHAPE = (N_CHANNELS, SEGMENT_SAMPLES, 2)


def ear_subbands(ear_signal):
    """Subbands of one ear: shape (39, n_samples), edge (low/high-pass) filters removed."""
    with np.errstate(divide="ignore"):  # pycochleagram logs freq 0 when plotting is set up
        subbands = cgm.human_cochleagram(
            ear_signal, SAMPLE_RATE, low_lim=LOW_LIM, hi_lim=HI_LIM, sample_factor=1,
            padding_size=PADDING_SIZE, ret_mode="subband")
    return subbands[1:-1]


def model_input(signal):
    """Convert a prepared 2-s stereo signal (96000, 2) into the model input.

    Returns float32 array of shape (39, 48000, 2): subbands of the middle 1 s,
    left and right ear stacked in the last dimension.
    """
    if signal.shape != (SIGNAL_SAMPLES, 2):
        raise ValueError("expected signal of shape {}, got {}"
                         .format((SIGNAL_SAMPLES, 2), signal.shape))
    segment = slice(SEGMENT_START, SEGMENT_START + SEGMENT_SAMPLES)
    subbands = np.stack([ear_subbands(signal[:, ear])[:, segment] for ear in (0, 1)],
                        axis=-1)
    peak = np.abs(subbands).max()
    subbands[np.abs(subbands) < peak / CUTOFF_RATIO] = 0
    return subbands.astype(np.float32)
