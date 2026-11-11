#!/bin/sh
# Local (Windows / Git Bash, CPU) version of slurm_CNN_spherical_testing_francl.sh.
# Run from the repo root:  sh legacy/run_test_local.sh
# Use forward slashes in all paths (the code splits paths on "/").

PYTHON=.venv/Scripts/python.exe
# Batch size (original: 16). Small values keep CPU memory use low; results are unaffected in testing mode.
export BATCH_SIZE=2  # must be >= 2 (the code squeezes the batch dimension)

# Folder that CONTAINS the net<ID> folders from the weights download
# (arch_number_<ID>_init_0 folder names work too)
model_path=weights
# Network number, i.e. which net<ID> folder to use
arch_ID=1
# Checkpoint step(s) to load: model.ckpt-<model_version>.* (comma-separated for several)
model_version=100000
init=0
regularizer=None

# Stimuli to localize (gzipped TFRecords of 39x48000x2 subbands, see README/BinauralDataGen)
train_path_pattern=test_data/lateral_noise/train*.tfrecords
# Background-noise TFRecords: always required, they are mixed into every stimulus
bkgd_train_path_pattern=test_data/bkgd/train*.tfrecords

# Signal-to-noise ratio in dB. In testing mode, values <=30 (min) and <=40 (max) are
# overridden to 30-35 dB by the code. 80/80 makes the background negligible.
SNR_min=80
SNR_max=80

# Stimulus-type flags: leave all False for ordinary stimuli labelled with train/azim + train/elev
manually_added=False
freq_label=False
sam_tones=False
transposed_tones=False
precedence_effect=False
narrowband_noise=False
stacked_channel=True
all_positions_bkgd=False
background_textures=True
testing=True

"$PYTHON" -u legacy/call_model_training_valid_pad_francl.py "$arch_ID" "$init" "$regularizer" \
  "$bkgd_train_path_pattern" "$train_path_pattern" "$model_version" "$model_path" \
  "$SNR_max" "$SNR_min" "$manually_added" "$freq_label" "$sam_tones" "$transposed_tones" \
  "$precedence_effect" "$narrowband_noise" "$stacked_channel" "$all_positions_bkgd" \
  "$background_textures" "$testing"
