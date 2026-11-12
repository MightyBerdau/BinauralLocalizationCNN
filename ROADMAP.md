# Roadmap

Plan for turning this fork into a usable, reproducible inference pipeline for the pretrained networks of
Francl & McDermott (2022). Status as of 2026-11-12.

## Goal

**One command from a folder of binaural audio files to a CSV of predicted source locations**, on an
ordinary Windows/Linux PC (CPU) and on a Linux GPU server. The original code requires users to prepare
TFRecords themselves and was built for the authors' SLURM cluster, which makes it hard to reproduce.

## Decisions

| topic | decision |
|---|---|
| scope | inference only (pretrained weights); no training, no spatialization of mono audio |
| deep-learning stack | stay on **TensorFlow 1.15** (no port to TF2/PyTorch: too much work) |
| input | binaural stereo audio (left, right ear); recommended: **2-s files at 48 kHz** |
| segment | **middle 1 s** of each file (paper's test procedure), after scaling the 2-s signal to RMS 0.1 |
| other sample rates | resampled to 48 kHz, with a warning |
| files shorter than 2 s | zero-padded symmetrically to 2 s, with a warning |
| files longer than 2 s | first 2 s used, with a warning |
| mono / unreadable / silent files | skipped and listed in `<output>_errors.csv`; the run continues |
| networks | user's choice, **default: all ten**; softmax probabilities averaged across the selected networks, then argmax (soft-voting ensemble). Deviates from the paper, which judges per network and averages *results* across networks (README, "How the predictions are computed") |
| output | CSV with the top prediction per file |
| weights | users unzip the authors' download into `weights/` and keep the folder names `net1` … `net10` |
| interface | installable package `blcnn` with CLI (`python -m blcnn predict …`) and Python API |
| original code | kept in `legacy/` (with CPU fixes), still runnable |
| environments | `uv` (pyproject + lock file) locally; GPU via the NVIDIA TF1 container (documented, no own Dockerfile) |
| validation | equivalence test against the legacy code path |
| users | own group: Windows PCs (CPU) and a Linux GPU server |
| release | public fork, no contact with the original authors for now |

## Milestones

### 1. Restructure — done
Original code moved to `legacy/`; package skeleton in `src/blcnn/` with vendored pycochleagram and
NetBuilder (see `src/blcnn/_vendor/README.md`). The legacy pipeline gives identical results after the move.

### 2. CPU pipeline — done
`python -m blcnn predict AUDIO_DIR -o predictions.csv [--nets all] [--device cpu|gpu]`.
Audio → 2 s, RMS 0.1 → 39 subbands → middle 1 s → TF 1.15 graph (from a placeholder, no TFRecords or
background records) → averaged probabilities → CSV. 30 unit tests (no weights needed).
All ten networks on CPU: 11 files in 83 s on a 16 GB Windows PC.

### 3. Equivalence with the original code — done (7 of 10 networks)
`tests/test_equivalence.py`: same subbands through blcnn and through the legacy TFRecord pipeline.
Same predicted class for all stimuli; max. probability difference 5e-7 … 3e-6 for net1, 2, 3, 5, 6, 8, 9.
**Open:** net4, net7, net10 (and net5 is borderline) — the *legacy* run runs out of memory on 16 GB RAM
(it always allocates Adam optimizer slots, ~3× the checkpoint size). Run on a larger machine:
`BLCNN_TEST_NETS=4,5,7,10 python -m pytest tests/test_equivalence.py -rs`.

### 3b. Example stimuli — done
`python -m blcnn example [DIR]` writes nine 2-s pink-noise stimuli rendered with the MIT KEMAR HRIRs
bundled in [slab](https://pypi.org/project/slab/) (optional extra `examples`) — the HRIR set the networks
were trained with (paper, Methods, ref. 53). Locations cover left/right, front/back mirror pairs and
elevations 0/20/60. The ten-network ensemble localizes all nine exactly; single networks (net1) miss the
median-plane ones (azimuth 0/180). `tests/test_examples.py`.

### 4. GPU on a Linux server — not started
Blocked: no server chosen yet. To do once one is available:

- Find out: GPU model, NVIDIA driver version (`nvidia-smi`), whether Docker or Apptainer/Singularity can be
  used, how much RAM.
- Pick an NVIDIA TF1 container tag (`nvcr.io/nvidia/tensorflow:<version>-tf1-py3`) that supports the GPU
  and driver. These images bring their own TensorFlow 1.15 build and likely **Python 3.8** (check).
- Make the package installable there:
  - widen `requires-python` (currently `>=3.7.1,<3.8`) to include the container's Python;
  - make `tensorflow` an optional extra (e.g. `blcnn[cpu]`) so `pip install` inside the container doesn't
    replace NVIDIA's TensorFlow build;
  - check the pinned numpy/scipy versions against the container.
- Test `--device gpu` (on GPU the downsampling runs on the full batch instead of per example; default
  batch size 16).
- Run the equivalence test for all networks there (closes milestone 3).
- Document the exact `docker run` / `apptainer exec` commands in the README.

## Planned extensions (not scheduled)

- **Segment selection:** a CSV listing file + start time per 1-s segment, instead of the middle second.
- **Sliding windows:** a time course of predictions over longer recordings.
- **True labels:** optional CSV with true azimuth/elevation → error columns in the output.
- **Full output distribution:** save the 504-way probabilities (e.g. `.npz`) for later analysis.
- **Per-network results** in addition to the ensemble average (one run, all networks), so the paper's
  per-network analysis doesn't need one `--nets N` run per network.
- **Front–back folding** of azimuths, as used in the paper for frontal-only experiments.
- **Plots** (e.g. predicted vs. true azimuth).
- **Weights download/import command** with checksums.
- **SLURM / Apptainer** job scripts for HPC clusters.

## Open questions and known issues

- **Azimuth convention:** class → direction mapping is derived from the original label construction
  (`class = 36 * elev/5 + azim/5`, i.e. 72 azimuths × 7 elevations). "0° = front, 90° = left"
  (counterclockwise) is confirmed with the KEMAR example stimuli, including front/back and elevation
  (milestone 3b; slab's KEMAR data use the same convention, checked by ILD/ITD at ±90°). The authors never
  answered the corresponding issue (afrancl/BinauralLocalizationCNN#4). Still unchecked: real recordings
  (with a room) of known direction.
- **License:** the original repository has no license. Forking on GitHub is covered by GitHub's terms, but
  reuse beyond that is not clearly permitted.
- **Background noise in the original pipeline:** because of the model's `x ** 0.3` compression, the
  background mixed in by the original code still has a small effect even at 80–150 dB SNR. blcnn uses no
  background (equivalent to infinite SNR).
- **Downsampling filter:** the original passes `beta=10.06` to `downsample()` but never forwards it; the
  networks were trained with the default `beta=5.0`, which blcnn uses.
- **Windows + Python 3.7:** the `blcnn.exe` launcher created by uv doesn't start; use `python -m blcnn`.
