"""TensorFlow 1.15 inference graph for one pretrained network.

Mirrors the graph of legacy/tf_record_CNN_spherical_gradcheckpoint_valid_pad.py
in testing mode, but takes the subbands from a placeholder instead of a
TFRecord pipeline:

    subbands -> half-wave rectification -> lowpass + downsampling to 8 kHz
             -> rectification -> power 0.3 -> CNN (NetBuilder) -> softmax

The original code mixes a background-noise record into each stimulus before
rectification; without noise that step returns the stimulus unchanged (it only
rescales the mixture back to the stimulus' peak), so it is omitted here.
"""

import contextlib
import io
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # hide TF C++ info/warning logs

import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402
from scipy.signal import windows  # noqa: E402

from ._vendor.NetBuilder_valid_pad import NetBuilder  # noqa: E402
from .cochleagram import INPUT_SHAPE  # noqa: E402
from .weights import load_config_array  # noqa: E402

tf1 = tf.compat.v1
tf1.logging.set_verbosity(tf1.logging.ERROR)

N_CLASSES_LOCALIZATION = 504
N_CLASSES_RECOGNITION = 780  # unused by the localization networks, required by NetBuilder
INPUT_RATE = 48000
COCHLEAGRAM_RATE = 8000
DOWNSAMPLE_FACTOR = INPUT_RATE // COCHLEAGRAM_RATE
FILTER_SIZE = 4097
# The original code passes beta=10.06 to downsample(), but downsample() does not
# forward it, so make_downsample_filt_tensor() uses its default beta=5.0. The
# networks were trained with that filter, so 5.0 is used here.
FILTER_BETA = 5.0
COMPRESSION_EXPONENT = 0.3


def downsample_filter():
    """Kaiser-windowed sinc lowpass, as make_downsample_filt_tensor() in the original."""
    times = np.arange(-FILTER_SIZE / 2, int(FILTER_SIZE / 2))
    response = np.sinc(times / DOWNSAMPLE_FACTOR) / DOWNSAMPLE_FACTOR
    response = windows.kaiser(FILTER_SIZE, FILTER_BETA) * response
    return response.reshape(1, FILTER_SIZE, 1, 1).astype(np.float32)


def build_graph(config_array, per_example_downsampling=True):
    """Build the inference graph in the current default graph.

    Returns ``(subbands_placeholder, probabilities)``.
    """
    subbands = tf1.placeholder(tf.float32, (None,) + INPUT_SHAPE, name="subbands")
    rectified = tf.nn.relu(subbands)
    left, right = tf.unstack(rectified, axis=3)
    both_ears = tf.expand_dims(tf.concat([left, right], axis=0), axis=3)
    lowpass = tf.constant(downsample_filter())

    def convolve(batch):
        return tf.nn.conv2d(batch, lowpass, strides=[1, 1, DOWNSAMPLE_FACTOR, 1],
                            padding="SAME")

    if per_example_downsampling:
        # On CPU the full-batch convolution with this long filter exhausts memory;
        # one example at a time gives identical results.
        downsampled = tf.map_fn(lambda x: convolve(x[None])[0], both_ears,
                                parallel_iterations=1)
    else:
        downsampled = convolve(both_ears)
    downsampled = tf.squeeze(tf.nn.relu(downsampled), axis=3)
    left_ds, right_ds = tf.split(downsampled, 2, axis=0)
    cochleagram = tf.pow(tf.stack([left_ds, right_ds], axis=3), COMPRESSION_EXPONENT)

    with contextlib.redirect_stdout(io.StringIO()):  # NetBuilder prints every layer
        logits = NetBuilder().build(
            config_array, cochleagram, False, False, tf.float32, "VALID",
            N_CLASSES_LOCALIZATION, N_CLASSES_RECOGNITION, False, None)
    return subbands, tf.nn.softmax(logits)


class LocalizationNetwork(object):
    """One pretrained network, restored into its own graph and session."""

    def __init__(self, network_files, device="cpu"):
        self.number = network_files.number
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.subbands, self.probabilities = build_graph(
                load_config_array(network_files), per_example_downsampling=(device == "cpu"))
            # The layer configs request /gpu:<n> devices; soft placement maps them to
            # the CPU when no GPU is used.
            config = tf1.ConfigProto(allow_soft_placement=True)
            if device == "cpu":
                config.device_count["GPU"] = 0
            self.session = tf1.Session(graph=self.graph, config=config)
            tf1.train.Saver().restore(self.session, network_files.checkpoint)

    def predict(self, batch):
        """Class probabilities, shape (batch, 504), for subbands of shape (batch, 39, 48000, 2)."""
        return self.session.run(self.probabilities, {self.subbands: batch})

    def close(self):
        self.session.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.close()


def gpu_available():
    return tf.test.is_gpu_available()
