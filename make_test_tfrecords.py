"""Generate a small synthetic test set in the TFRecord format the model expects.

Stimuli: 0.5 s broadband noise bursts (in a 1 s / 48000-sample window), spatialized
with a crude spherical-head model (Woodworth ITD + 10 dB*sin(az) ILD above 1.5 kHz).
NO real HRTFs are used, so there are no spectral elevation / front-back cues.

Each record: 'train/image' = float32 bytes of shape [39, 48000, 2] (cochlear subbands
from pycochleagram, 30 Hz - 20 kHz, edge filters removed), plus 'train/azim',
'train/elev' (degrees, int64), 'train/image_height', 'train/image_width'.

Usage:  .venv/Scripts/python.exe make_test_tfrecords.py <out_dir>
Writes <out_dir>/lateral_noise/train0.tfrecords and <out_dir>/bkgd/train0.tfrecords
"""
import os
import sys
import numpy as np
import tensorflow as tf
from pycochleagram import erbfilter as erb
from pycochleagram import subband as sb

SR = 48000
N_SAMPLES = 48000
LOW_LIM, HI_LIM = 30, 20000
HEAD_RADIUS, SPEED_OF_SOUND = 0.0875, 343.0

# lateral angles in degrees; positive = source on the RIGHT (right ear leads)
ANGLES = [-90, -60, -30, -10, 10, 30, 60, 90]
REPS = 4  # different noise tokens per angle (8*4 = 32 = two batches of 16)

n = int(np.floor(erb.freq2erb(HI_LIM) - erb.freq2erb(LOW_LIM)) - 1)  # 39
filts, _, _ = erb.make_erb_cos_filters_nx(N_SAMPLES, SR, n, LOW_LIM, HI_LIM, 1,
                                          pad_factor=None, full_filter=True)
filts = filts[1:-1]  # drop low/high-pass edge filters -> 39 channels


def spatialize(mono, az_deg):
    """Return (left, right) using Woodworth ITD and a high-frequency ILD."""
    th = np.deg2rad(abs(az_deg))
    itd = HEAD_RADIUS / SPEED_OF_SOUND * (th + np.sin(th))  # seconds, >= 0
    ild_db = 10.0 * np.sin(th)
    spec = np.fft.rfft(mono)
    freqs = np.fft.rfftfreq(len(mono), 1.0 / SR)
    far = spec * np.exp(-2j * np.pi * freqs * itd)           # delayed far ear
    far = far * np.where(freqs > 1500, 10 ** (-ild_db / 20), 1.0)
    near_sig, far_sig = mono, np.fft.irfft(far, len(mono))
    return (far_sig, near_sig) if az_deg > 0 else (near_sig, far_sig)


def to_image(left, right):
    sub_l = sb.generate_subbands(left, filts)
    sub_r = sb.generate_subbands(right, filts)
    return np.stack([sub_l, sub_r], axis=-1).astype(np.float32)  # [39, 48000, 2]


def example(img, azim, elev):
    i64 = lambda v: tf.train.Feature(int64_list=tf.train.Int64List(value=[int(v)]))
    return tf.train.Example(features=tf.train.Features(feature={
        'train/image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img.tobytes()])),
        'train/image_height': i64(img.shape[0]),
        'train/image_width': i64(img.shape[1]),
        'train/azim': i64(azim),
        'train/elev': i64(elev),
    })).SerializeToString()


def main(out_dir):
    opts = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.GZIP)
    stim_dir, bkgd_dir = os.path.join(out_dir, 'lateral_noise'), os.path.join(out_dir, 'bkgd')
    os.makedirs(stim_dir, exist_ok=True)
    os.makedirs(bkgd_dir, exist_ok=True)
    rng = np.random.RandomState(0)
    burst_len = SR // 2
    ramp = np.hanning(2 * 480)  # 10 ms on/off ramps
    env = np.ones(burst_len)
    env[:480], env[-480:] = ramp[:480], ramp[480:]

    with tf.python_io.TFRecordWriter(os.path.join(stim_dir, 'train0.tfrecords'), opts) as w:
        for az in ANGLES:
            for _ in range(REPS):
                mono = np.zeros(N_SAMPLES)
                start = SR // 4
                mono[start:start + burst_len] = rng.randn(burst_len) * env
                mono *= 0.1 / np.sqrt(np.mean(mono[start:start + burst_len] ** 2))
                left, right = spatialize(mono, az)
                w.write(example(to_image(left, right), az % 360, 0))
            print('wrote angle', az)

    # Near-silent background (a few impulses so its RMS is non-zero); at SNR 80 dB it is
    # negligible. One background record is consumed per stimulus, so write as many.
    with tf.python_io.TFRecordWriter(os.path.join(bkgd_dir, 'train0.tfrecords'), opts) as w:
        for _ in range(len(ANGLES) * REPS):
            img = np.zeros([39, N_SAMPLES, 2], np.float32)
            img[:, ::4800, :] = 1e-3
            w.write(example(img, 0, 0))
    print('done:', out_dir)


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'test_data')
