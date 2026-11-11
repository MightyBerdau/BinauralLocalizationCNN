# BinauralLocalizationCNN — local (CPU) version

Deep neural networks that localize sound sources in 3D from binaural (two-ear) audio.

## Purpose

This repository is a fork of [afrancl/BinauralLocalizationCNN](https://github.com/afrancl/BinauralLocalizationCNN),
the code accompanying

> Francl, A. & McDermott, J. H. Deep neural network models of sound localization reveal how perception is
> adapted to real-world environments. *Nature Human Behaviour* **6**, 111–133 (2022).
> https://doi.org/10.1038/s41562-021-01244-z

The original code was written to run on a GPU computing cluster (SLURM jobs inside a Singularity container
with TensorFlow 1.13). **This fork aims to make the pretrained networks run on an ordinary Windows PC without
a GPU and without an HPC environment.** It has only been tested for **testing/prediction**, not for training.

## Table of contents

- [Purpose](#purpose)
- [Requirements](#requirements)
- [Installation](#installation)
  - [1. Install Python 3.7 (64-bit)](#1-install-python-37-64-bit)
  - [2. Create the virtual environment](#2-create-the-virtual-environment)
  - [3. Download the network weights](#3-download-the-network-weights)
- [Input data](#input-data)
  - [Record format](#record-format)
  - [Preparing your own audio](#preparing-your-own-audio)
  - [Where to get data](#where-to-get-data)
- [Usage](#usage)
  - [Configure and run](#configure-and-run)
  - [Output files](#output-files)
  - [Converting the output to a direction](#converting-the-output-to-a-direction)
  - [Caveats](#caveats)
- [Repository structure](#repository-structure)
- [Changes compared to the original repository](#changes-compared-to-the-original-repository)
- [Running on a cluster (original setup)](#running-on-a-cluster-original-setup)
- [Resources](#resources)
- [Citation](#citation)

## Requirements

- Windows with **Git Bash** (the commands below use POSIX shell syntax)
- **Python 3.7, 64-bit** — TensorFlow 1.x does not support newer Python versions
- No GPU needed
- Disk space for the weights: 0.65 GB (net1) up to ~20 GB (all ten networks)

## Installation

### 1. Install Python 3.7 (64-bit)

```sh
winget install --id Python.Python.3.7 --scope user --architecture x64
```

Alternatively, use the installer `python-3.7.9-amd64.exe` from python.org.

Check: `py -0p` should list `-V:3.7  ...\Python37\python.exe` (not `3.7-32`).

### 2. Create the virtual environment

[pyproject.toml](pyproject.toml) declares the required Python version (3.7) and the direct dependencies:
TensorFlow 1.15.5 (CPU) and compatible versions of numpy, scipy, protobuf, etc. Don't upgrade these
packages; newer versions break TensorFlow 1.15. The optional `analysis` extra (pandas, seaborn,
scikit-learn) is only needed for the scripts in [analysis_and_plotting/](analysis_and_plotting/).

From the repository root, in Git Bash, either with [uv](https://docs.astral.sh/uv/) (recommended):

```sh
uv sync --extra analysis --python "$(py -3.7 -c 'import sys; print(sys.executable)')"
```

This creates `.venv` and installs exactly the versions recorded in [uv.lock](uv.lock), which uv
generates from pyproject.toml (don't edit it by hand). Note: `uv sync` without `--extra analysis`
removes the analysis packages again.

Or with pip:

```sh
py -3.7 -m venv .venv
.venv/Scripts/python.exe -m pip install --upgrade "pip<24.1"
.venv/Scripts/python.exe -m pip install --default-timeout=300 ".[analysis]"
```

This installs only the dependencies (the repository itself is not a package). pip ignores
[uv.lock](uv.lock), so packages that are not pinned in pyproject.toml may resolve to different versions.
For the exact locked versions without uv, install from a requirements file exported by someone with uv
(`uv export --all-extras --format requirements-txt > requirements.txt`) using `pip install -r requirements.txt`.

Verify the installation:

```sh
.venv/Scripts/python.exe -c "import tensorflow as tf, tensorflow.contrib.slim; print(tf.__version__)"
```

This should print `1.15.5`. TensorFlow also prints many deprecation warnings, and possibly a harmless
`SyntaxError ... 'import`; both can be ignored.

### 3. Download the network weights

1. Download the weights from the
   [Dropbox link](https://www.dropbox.com/sh/af6vaotxt41i7pe/AACfTzMxMLfv-Edmn33S4gTpa?dl=0).
   You get a zip containing folders `net1` … `net10`.
2. The code expects folders named **`arch_number_<N>_init_0`**, not `net<N>`. Extract the network(s) you
   need into `weights/` and rename them. For example, for net1:

   ```sh
   mkdir -p weights
   unzip -q "Binaural Localization Net Weights.zip" "net1/*" -d weights
   mv weights/net1 weights/arch_number_1_init_0
   ```

   Each folder must contain `config_array.npy` and `model.ckpt-100000.*`.
   The networks are large (net1: 0.65 GB; net4, net5, net7 and net10: about 3.5 GB each; about 20 GB in
   total), so only extract the ones you need.

## Input data

### Record format

The model doesn't read `.wav` files directly. It reads **gzip-compressed TFRecords** in which each sound is
already split into cochlear subbands. Each record contains:

| key | content |
|---|---|
| `train/image` | float32 bytes, shape `[39, 48000, 2]`: 39 subbands (30 Hz–20 kHz, pycochleagram), 1 s at 48 kHz, left/right ear |
| `train/azim`, `train/elev` | true direction in degrees (int64) |
| `train/image_height`, `train/image_width` | int64 |

You always also need a **set of background-noise TFRecords** in the same format: the code mixes background
noise into every sound. Use SNR = 80 dB to make the noise negligible.

### Preparing your own audio

The model judges exactly **1 s** of audio and returns **one** location for it. Prepare each 1-s segment as
follows. This follows the paper's test procedure (Methods section of the paper cited above) and the
cochleagram settings in BinauralDataGen's `cochleagram_wrapper()`.

1. **Binaural stereo audio at 48 kHz.** Channel 0 is the left ear, channel 1 the right. The audio must
   contain head/ear cues: recorded with in-ear microphones, or rendered with HRIRs/BRIRs. Resample other
   sample rates to 48 kHz.
2. **Cut 2 s around your segment:** 0.5 s before and 0.5 s after the second you want. Pad with zeros if
   needed.
3. **Scale the 2-s stereo signal to an RMS of 0.1**, using one common factor for both ears so the level
   difference between them is kept.
4. **Compute subbands for each ear** with the [pycochleagram](pycochleagram/) included in this repo:
   ```python
   from pycochleagram import cochleagram as cgm
   sub = cgm.human_cochleagram(ear, 48000, low_lim=30, hi_lim=20000, sample_factor=1,
                               padding_size=10000, ret_mode='subband')  # 41 x 96000
   sub = sub[1:-1]                    # drop low-/high-pass edge filters -> 39 bands
   ```
5. **Keep the middle 1 s** (`sub[:, 24000:72000]`) and stack both ears into a float32 array of shape
   `[39, 48000, 2]`. Set all values with magnitude below `max(|x|)/1000` to 0 (−60 dB cutoff).
6. **Write one TFRecord example per segment** (GZIP compression, file name `train*.tfrecords`, one folder
   per stimulus set) with the keys from the table above. Labels are in degrees; use 0 if the direction is
   unknown. Optionally add an int64 key such as `train/file_idx`. Every extra key is copied to the output,
   so you can match predictions to files (the output order is shuffled).
7. **Background set:** in a separate folder, with **at least as many records as stimuli**. The records need
   `train/azim`/`train/elev` and a non-zero RMS. With SNR = 80 dB a near-silent signal is fine; see
   [make_test_tfrecords.py](make_test_tfrecords.py) for an example of both record types.

Make the number of stimuli a multiple of `BATCH_SIZE`, because an incomplete last batch is dropped.

### Where to get data

- **Precomputed training data** used for the original model:
  [Google Drive](https://drive.google.com/drive/folders/1pUyMmkurxEWFeofbLKy5-QVWrcMbM8Wz?usp=sharing)
- **Your own `.wav` files**, prepared as described above. The authors' wrapper
  [BinauralDataGen](https://github.com/afrancl/BinauralDataGen) does a similar conversion, but it is set up
  for their cluster, takes one *random* 1-s window from the first 2 s of each file, and won't run locally
  without changes.
- **A synthetic test set** for a quick check (no real HRTFs: left/right cues only, so front/back and
  elevation are not meaningful):

  ```sh
  .venv/Scripts/python.exe make_test_tfrecords.py test_data
  ```

  This writes `test_data/lateral_noise/` (32 noise bursts at 8 left/right angles) and `test_data/bkgd/`,
  about 450 MB in total.

## Usage

### Configure and run

Edit the variables at the top of [run_test_local.sh](run_test_local.sh):

| variable | meaning |
|---|---|
| `model_path` | folder containing the `arch_number_<N>_init_0` folders (default `weights`) |
| `arch_ID` | which network to use (`1` = net1) |
| `model_version` | checkpoint number (`100000`) |
| `train_path_pattern` | the sounds to localize |
| `bkgd_train_path_pattern` | the background-noise records |
| `SNR_min`, `SNR_max` | leave both at `80`. In test mode the code replaces values up to 30 (min) and 40 (max) with 30–35 dB |
| `BATCH_SIZE` | keep it small (2) on a normal PC; must be at least 2 |

Use forward slashes in all paths. Then run, from the repo root:

```sh
sh run_test_local.sh
```

The defaults run net1 on the synthetic test set, which takes about 30 s on CPU.

### Output files

The results are written **into the network's folder**, e.g. `weights/arch_number_1_init_0/`
(`<stim>` is the name of the folder containing the input TFRecords):

| file | content |
|---|---|
| `batch_conditional_<stim>_iter100000.npy` | one entry per sound: `(probabilities[504], labels)` |
| `keys_test_<stim>_iter100000.json` | names of the labels, in order (azimuth and elevation in 5° units, i.e. degrees / 5) |
| `accuracies_test_<stim>_iter100000.json` | fraction of exact azimuth+elevation matches |

### Converting the output to a direction

```python
import numpy as np
data = np.load("weights/arch_number_1_init_0/batch_conditional_lateral_noise_iter100000.npy",
               allow_pickle=True)
for probs, labels in data:
    c = int(np.argmax(probs))
    azimuth, elevation = (c % 72) * 5, (c // 72) * 10   # degrees
```

Azimuth increases **counterclockwise**: 90° is left and 270° is right. This was worked out from the code
and a synthetic left/right test, not from the authors' documentation.

### Caveats

- The order of the sounds in the output is **shuffled**. Use the stored labels to match predictions to
  sounds.
- Sounds left over after the last full batch are dropped (e.g. with `BATCH_SIZE=2`, an odd last one).
- Leave the special stimulus flags in the script (`sam_tones`, `precedence_effect`, …) set to `False` for
  ordinary sounds. They switch the code to the authors' psychoacoustic experiments.
- Training has not been tested locally and is unlikely to be practical on a CPU.

## Repository structure

| path | description |
|---|---|
| [call_model_training_valid_pad_francl.py](call_model_training_valid_pad_francl.py) | main entry point: parses the experiment parameters, validates the model folder and starts training/testing |
| [tf_record_CNN_spherical_gradcheckpoint_valid_pad.py](tf_record_CNN_spherical_gradcheckpoint_valid_pad.py) | data pipeline, network construction and train/test loop |
| [NetBuilder_valid_pad.py](NetBuilder_valid_pad.py) | builds the network from `config_array.npy` |
| [run_test_local.sh](run_test_local.sh) | **local** test/prediction script (this fork) |
| [make_test_tfrecords.py](make_test_tfrecords.py) | generates a synthetic test set (this fork) |
| [pyproject.toml](pyproject.toml), [uv.lock](uv.lock) | Python version and dependencies; exact pinned versions in the lock file (this fork) |
| [pycochleagram/](pycochleagram/) | cochlear filterbank used to create the model input |
| [multi_source_training/](multi_source_training/) | code for the multi-source models (see its README.txt) |
| [analysis_and_plotting/](analysis_and_plotting/) | analysis and plotting scripts from the original authors |
| `slurm_*.sh`, `interactive_*.sh` | original cluster scripts (SLURM / Singularity) |

## Changes compared to the original repository

- [layer_generator.py](layer_generator.py): a placeholder. The module is imported but missing from the
  original repo, and it is never used.
- [call_model_training_valid_pad_francl.py](call_model_training_valid_pad_francl.py):
  `np.load(..., allow_pickle=True)` for `config_array.npy` (needed by the installed numpy version).
- [tf_record_CNN_spherical_gradcheckpoint_valid_pad.py](tf_record_CNN_spherical_gradcheckpoint_valid_pad.py):
  - the first downsampling convolution now processes one example at a time. On CPU, the full-batch version
    exhausts memory and aborts. The results are identical.
  - the unused `tf.add_check_numerics_ops()` is disabled, because it doesn't work with the change above.
  - the batch size can be set with the `BATCH_SIZE` environment variable (default 16, as in the original).
- New files: [run_test_local.sh](run_test_local.sh), [make_test_tfrecords.py](make_test_tfrecords.py),
  [pyproject.toml](pyproject.toml), [uv.lock](uv.lock), `.gitignore`.

## Running on a cluster (original setup)

The original workflow is still included unchanged:

- Example parameters for testing and training are in
  [slurm_CNN_spherical_testing_francl.sh](slurm_CNN_spherical_testing_francl.sh) and
  [slurm_CNN_spherical_training_francl.sh](slurm_CNN_spherical_training_francl.sh). Before running, change
  the model save folder to point to your directory with the model architecture config file, and the data
  folder to point to your data. The Python code itself contains no absolute paths.
- The authors provide a [Singularity image](https://sylabs.io/singularity/) with all required packages
  (TensorFlow 1.13):
  https://www.dropbox.com/s/ey74fiw4uquww0n/tfv1.13_tcmalloc.simg?dl=0

## Resources

- Original repository: https://github.com/afrancl/BinauralLocalizationCNN
- Network weights: https://www.dropbox.com/sh/af6vaotxt41i7pe/AACfTzMxMLfv-Edmn33S4gTpa?dl=0
- Precomputed training data: https://drive.google.com/drive/folders/1pUyMmkurxEWFeofbLKy5-QVWrcMbM8Wz?usp=sharing
- Input generation (BinauralDataGen): https://github.com/afrancl/BinauralDataGen
- Cochlear model (pycochleagram): https://github.com/mcdermottLab/pycochleagram

## Citation

If you use the models, please cite the original paper:

> Francl, A. & McDermott, J. H. Deep neural network models of sound localization reveal how perception is
> adapted to real-world environments. *Nature Human Behaviour* **6**, 111–133 (2022).
> https://doi.org/10.1038/s41562-021-01244-z
