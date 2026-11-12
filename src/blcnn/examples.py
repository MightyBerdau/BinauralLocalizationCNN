"""Example stimuli: pink noise rendered at known locations with KEMAR HRTFs.

Uses the MIT KEMAR head-related impulse responses (Gardner & Martin, 1995) that
ship with the ``slab`` package, the same HRIR set the networks were trained with
(Francl & McDermott, 2022, Methods). The stimuli are anechoic; the training
renderings also included a simulated room.

Requires the optional dependency: ``pip install ".[examples]"``.
"""

import logging
import os

import numpy as np
import soundfile as sf

from .audio import SAMPLE_RATE, SIGNAL_SECONDS, resample

log = logging.getLogger(__name__)

# (azimuth, elevation) in degrees; azimuth counterclockwise from the front
# (90 = left), as in slab's KEMAR data and the network output. All positions are
# KEMAR measurement points and network output bins. 30 and 150 (and 330 and 210)
# are front/back mirror images with nearly the same ITD and ILD.
EXAMPLE_LOCATIONS = [
    (0, 0), (30, 0), (90, 0), (150, 0), (180, 0), (270, 0), (330, 0),
    (30, 20), (0, 60),
]
RAMP_SECONDS = 0.01
PEAK = 0.5  # common peak level of all files; keeps ILDs, avoids clipping
SEED = 0


class ExamplesError(RuntimeError):
    """The example stimuli cannot be generated (e.g. slab not installed)."""


def _import_slab():
    try:
        import slab
    except ImportError:
        raise ExamplesError('generating examples needs the "slab" package; install it with '
                            'pip install ".[examples]" or uv sync --extra examples')
    except OSError as err:  # sounddevice (imported by slab) without the PortAudio library
        raise ExamplesError("importing slab failed ({}); on Linux install PortAudio, e.g. "
                            "apt install libportaudio2".format(err))
    return slab


def example_filename(azimuth, elevation):
    return "az{:03d}_el{:02d}.wav".format(azimuth, elevation)


def pink_noise(n_samples, rng):
    """Pink (1/f power) noise with unit RMS."""
    spectrum = np.fft.rfft(rng.randn(n_samples))
    freqs = np.fft.rfftfreq(n_samples)
    freqs[0] = freqs[1]
    noise = np.fft.irfft(spectrum / np.sqrt(freqs), n_samples)
    noise -= noise.mean()
    return noise / np.sqrt(np.mean(noise ** 2))


def _source_index(sources, azimuth, elevation):
    match = np.flatnonzero(np.isclose(sources[:, 0], azimuth % 360)
                           & np.isclose(sources[:, 1], elevation))
    if match.size != 1:
        raise ExamplesError("no KEMAR measurement at azimuth {}, elevation {}"
                            .format(azimuth, elevation))
    return int(match[0])


def make_examples(output_dir, locations=EXAMPLE_LOCATIONS, seed=SEED):
    """Write one 2-s binaural 48-kHz .wav file per location to ``output_dir``.

    Every file contains the same pink-noise token, so they differ only in the
    HRTF. Returns the list of written file paths.
    """
    slab = _import_slab()
    hrtf = slab.HRTF.kemar()
    rate = int(hrtf.samplerate)
    sources = hrtf.sources.vertical_polar

    n_samples = int(round(SIGNAL_SECONDS * rate))
    noise = pink_noise(n_samples, np.random.RandomState(seed))
    ramp = int(RAMP_SECONDS * rate)
    window = np.hanning(2 * ramp)
    noise[:ramp] *= window[:ramp]
    noise[-ramp:] *= window[ramp:]
    sound = slab.Sound(noise, samplerate=rate)

    rendered = []
    for azimuth, elevation in locations:
        binaural = hrtf.apply(_source_index(sources, azimuth, elevation), sound)
        rendered.append(resample(np.asarray(binaural.data)[:n_samples], rate, SAMPLE_RATE))
    scale = PEAK / max(np.max(np.abs(signal)) for signal in rendered)

    os.makedirs(output_dir, exist_ok=True)
    paths = []
    for (azimuth, elevation), signal in zip(locations, rendered):
        path = os.path.join(output_dir, example_filename(azimuth, elevation))
        sf.write(path, signal * scale, SAMPLE_RATE, subtype="PCM_24")
        paths.append(path)
        log.info("wrote %s", path)
    return paths
