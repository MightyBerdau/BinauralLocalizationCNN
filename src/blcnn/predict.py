"""Inference pipeline: folder of binaural audio files -> CSV of predicted locations."""

import csv
import logging
import os
import shutil
import tempfile
import time

import numpy as np

from .audio import AudioError, prepare_signal
from .cochleagram import model_input
from .weights import select_networks

log = logging.getLogger(__name__)

N_AZIMUTHS = 72
AZIMUTH_STEP = 5     # degrees
ELEVATION_STEP = 10  # degrees
AUDIO_EXTENSIONS = (".wav",)

RESULT_COLUMNS = ["file", "network", "azimuth_deg", "elevation_deg", "probability", "warnings"]
ERROR_COLUMNS = ["file", "error"]


def decode_class(index):
    """Map an output class (0 ... 503) to (azimuth, elevation) in degrees.

    72 azimuths (5 deg steps) x 7 elevations (0 ... 60 deg, 10 deg steps). Derived
    from the label construction in the original code (class = 36 * elev/5 + azim/5);
    azimuth increases counterclockwise (90 deg = left), as observed on synthetic
    left/right stimuli.
    """
    index = int(index)
    return (index % N_AZIMUTHS) * AZIMUTH_STEP, (index // N_AZIMUTHS) * ELEVATION_STEP


def find_audio_files(audio_dir):
    """All .wav files below ``audio_dir`` (recursive), as sorted relative paths."""
    if not os.path.isdir(audio_dir):
        raise FileNotFoundError("audio folder not found: {}".format(audio_dir))
    found = []
    for root, _, names in os.walk(audio_dir):
        for name in names:
            if name.lower().endswith(AUDIO_EXTENSIONS):
                path = os.path.relpath(os.path.join(root, name), audio_dir)
                found.append(path.replace(os.sep, "/"))
    return sorted(found)


def errors_path(output_csv):
    stem, ext = os.path.splitext(output_csv)
    return stem + "_errors" + (ext or ".csv")


def _write_csv(path, columns, rows):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _preprocess(audio_dir, files, cache_dir):
    """Write the model input of each usable file to ``cache_dir``.

    Returns ``(usable, warnings, errors)``: usable relative paths, a dict of warning
    lists per file, and a list of error rows.
    """
    usable, warnings, errors = [], {}, []
    for i, rel_path in enumerate(files):
        try:
            signal, file_warnings = prepare_signal(os.path.join(audio_dir, rel_path))
            subbands = model_input(signal)
        except AudioError as err:
            log.warning("skipping %s: %s", rel_path, err)
            errors.append({"file": rel_path, "error": str(err)})
            continue
        for message in file_warnings:
            log.warning("%s: %s", rel_path, message)
        np.save(os.path.join(cache_dir, "{}.npy".format(len(usable))), subbands)
        usable.append(rel_path)
        warnings[rel_path] = file_warnings
        log.info("preprocessed %d/%d: %s", i + 1, len(files), rel_path)
    return usable, warnings, errors


def _run_network(network_files, cache_dir, n_inputs, device, batch_size):
    """Probabilities (n_inputs, 504) of one network for all cached inputs."""
    from .model import LocalizationNetwork  # imports TensorFlow; only needed here

    probabilities = []
    with LocalizationNetwork(network_files, device=device) as network:
        for start in range(0, n_inputs, batch_size):
            indices = range(start, min(start + batch_size, n_inputs))
            batch = np.stack([np.load(os.path.join(cache_dir, "{}.npy".format(i)))
                              for i in indices])
            probabilities.append(network.predict(batch))
    return np.concatenate(probabilities)


def predict_folder(audio_dir, output_csv, weights_dir="weights", nets="all", device="cpu",
                   batch_size=1):
    """Predict the source location of every .wav file in ``audio_dir`` with each network.

    Each selected network judges each file on its own (its most probable location
    bin); the networks are not combined. Writes ``output_csv`` (one row per usable
    file and network, sorted by file, then network) and, if any file was skipped,
    ``<output>_errors.csv``. Returns the list of result rows.
    """
    networks = select_networks(weights_dir, nets)
    files = find_audio_files(audio_dir)
    if not files:
        raise FileNotFoundError("no .wav files found in {}".format(audio_dir))
    log.info("%d audio file(s), network(s) %s, device %s", len(files),
             ",".join(str(n.number) for n in networks), device)

    output_dir = os.path.dirname(os.path.abspath(output_csv))
    os.makedirs(output_dir, exist_ok=True)
    # Cache the preprocessed inputs (about 15 MB per file) next to the output rather
    # than in the system temp folder, which may be on a small drive.
    cache_dir = tempfile.mkdtemp(prefix=".blcnn_cache_", dir=output_dir)
    try:
        usable, warnings, errors = _preprocess(audio_dir, files, cache_dir)
        rows = []
        for network_files in networks if usable else []:
            started = time.time()
            probabilities = _run_network(network_files, cache_dir, len(usable), device,
                                         batch_size)
            log.info("network %d done in %.1f s", network_files.number,
                     time.time() - started)
            for rel_path, probs in zip(usable, probabilities):
                best = int(np.argmax(probs))
                azimuth, elevation = decode_class(best)
                rows.append({
                    "file": rel_path,
                    "network": network_files.number,
                    "azimuth_deg": azimuth,
                    "elevation_deg": elevation,
                    "probability": round(float(probs[best]), 4),
                    "warnings": "; ".join(warnings[rel_path]),
                })
        rows.sort(key=lambda row: (row["file"], row["network"]))
    finally:
        shutil.rmtree(cache_dir, ignore_errors=True)

    _write_csv(output_csv, RESULT_COLUMNS, rows)
    log.info("wrote %d prediction(s) to %s", len(rows), output_csv)
    if errors:
        _write_csv(errors_path(output_csv), ERROR_COLUMNS, errors)
        log.warning("%d file(s) skipped, see %s", len(errors), errors_path(output_csv))
    return rows
