"""Locate the pretrained networks in the weights folder.

Accepts the folder names of the published download (``net1`` ... ``net10``) and
the names used by the original code (``arch_number_<N>_init_0``).
"""

import glob
import os
import re
from collections import namedtuple

import numpy as np

NetworkFiles = namedtuple("NetworkFiles", ["number", "folder", "config_array", "checkpoint"])

_FOLDER_PATTERNS = [re.compile(r"^net(\d+)$"), re.compile(r"^arch_number_(\d+)_init_0$")]


class WeightsError(ValueError):
    """The weights folder or a network in it is missing or incomplete."""


def _network_number(folder_name):
    for pattern in _FOLDER_PATTERNS:
        match = pattern.match(folder_name)
        if match:
            return int(match.group(1))
    return None


def _latest_checkpoint(folder):
    """Checkpoint prefix (``.../model.ckpt-<step>``) with the highest step."""
    steps = []
    for index_file in glob.glob(os.path.join(folder, "model.ckpt-*.index")):
        match = re.search(r"model\.ckpt-(\d+)\.index$", index_file)
        if match:
            steps.append(int(match.group(1)))
    if not steps:
        raise WeightsError("no model.ckpt-<step>.index in {}".format(folder))
    return os.path.join(folder, "model.ckpt-{}".format(max(steps)))


def find_networks(weights_dir):
    """Return {number: NetworkFiles} for all networks found in ``weights_dir``."""
    if not os.path.isdir(weights_dir):
        raise WeightsError("weights folder not found: {}".format(weights_dir))
    networks = {}
    for name in sorted(os.listdir(weights_dir)):
        folder = os.path.join(weights_dir, name)
        number = _network_number(name)
        if number is None or not os.path.isdir(folder):
            continue
        if number in networks:
            raise WeightsError("network {} found twice: {} and {}"
                               .format(number, networks[number].folder, folder))
        config_array = os.path.join(folder, "config_array.npy")
        if not os.path.isfile(config_array):
            raise WeightsError("missing config_array.npy in {}".format(folder))
        networks[number] = NetworkFiles(number, folder, config_array, _latest_checkpoint(folder))
    if not networks:
        raise WeightsError("no network folders (net1 ... net10) in {}".format(weights_dir))
    return networks


def select_networks(weights_dir, selection="all"):
    """Networks to use, sorted by number.

    ``selection`` is ``"all"`` or a comma-separated list of numbers, e.g. ``"1,3,8"``.
    """
    available = find_networks(weights_dir)
    if selection.strip().lower() == "all":
        return [available[n] for n in sorted(available)]
    try:
        wanted = sorted({int(part) for part in selection.split(",") if part.strip()})
    except ValueError:
        raise WeightsError("invalid network selection {!r}; use e.g. '1,3,8' or 'all'"
                           .format(selection))
    missing = [n for n in wanted if n not in available]
    if missing:
        raise WeightsError("network(s) {} not found in {} (available: {})".format(
            ", ".join(map(str, missing)), weights_dir,
            ", ".join(map(str, sorted(available)))))
    if not wanted:
        raise WeightsError("no networks selected")
    return [available[n] for n in wanted]


def load_config_array(network):
    """The layer configuration of a network (object array, needs allow_pickle)."""
    return np.load(network.config_array, allow_pickle=True)
