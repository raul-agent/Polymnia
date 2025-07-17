# Multimodal Dataset Creation Pipeline

This repository contains a modular pipeline for creating multimodal datasets from video clips. It includes automatic downloading, filtering, pose extraction, prosodic analysis, lexical alignment, and integration into structured formats like Parquet and DuckDB.



## 🔧 Installation

Create a virtual environment and install dependencies:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
````

> `parole.sh` (used for prosody analysis) is part of the [parole system](https://github.com/daedalusLAB/parole), which includes its own installer (`install_parole.sh`) for Praat, R scripts, and dependencies. See `parole/` and install it before running `ecos.py`.

## 📦 Requirements

See `requirements.txt`:

```txt
argparse
opencv-python
ultralytics
praat-parselmouth
moviepy
pandas
duckdb
pyarrow
git+https://github.com/m-bain/whisperx.git
```

External dependencies:

* `curl` and `ffmpeg` (for video/audio processing)
* R + packages (`arrow`, `logger`, `multimolang`)
* OpenPose (compiled binary)
* Praat (called by `parole.sh`)



## 1️⃣ `download_clips.py` — Download video clips from NewsScape

Downloads `.mp4` video clips from `gallo.case.edu` based on a tabulated `.txt` file. Adds a configurable offset to clip timestamps. Retries failed downloads and logs errors.

**Inputs**:

* `--txt_file`: Tab-separated file with clip metadata
* `--searchterm`: Label for clip naming
* `--offset`: Seconds before/after clip (default: 2.5)
* `--output_dir`: Target folder for videos

**Example**:

```bash
python3 download_clips.py \
  --txt_file clips.txt \
  --searchterm climate_change \
  --output_dir ./output/videos/raw
```



## 2️⃣ `is_there_a_person_in_the_video.py` — Filter usable clips

Uses YOLOv8-Pose to accept only videos with exactly **one visible person** (head, shoulders, and optionally hands). Extracts 15 representative frames per clip.

**Inputs**:

* `--videos_folder`: Folder with `.mp4` files
* `--matched_videos`: Output folder for accepted clips
* `--discarded_videos`: Folder to store discarded clips
* `--check_hands`: Require hand keypoints (default: False)

**Example**:

```bash
python3 is_there_a_person_in_the_video.py \
  --videos_folder ./output/videos/masked \
  --matched_videos ./output/videos/1_person \
  --discarded_videos ./output/videos/discarded \
  --check_hands True
```



## 3️⃣ `argos.py` — Run OpenPose and classify by number of people

Runs OpenPose on each video and saves body keypoints in `JSON_FILES/`. Then calls an R script to standardize coordinates and classify videos into `1_persons/`, `2_persons/`, etc.

**Inputs**:

* `--videos_folder`: Folder with filtered `.mp4` videos
* `--output_folder`: Root output directory
* `--openpose_path`: Path to OpenPose build
* `--face_hands`: Enable face/hands detection
* `--skeletons`: Save skeleton render videos

**Example**:

```bash
python3 argos.py \
  --videos_folder ./output/videos/1_person \
  --output_folder ./output \
  --openpose_path /opt/openpose \
  --face_hands True \
  --skeletons True
```



## 4️⃣ `max_people_classification.R` — Coordinate normalization and person counting

Called automatically by `argos.py`. Uses the `multimolang::dfMaker()` function to:

* Normalize OpenPose JSONs into structured Parquet
* Detect number of people per video
* Write `num_persons.txt` and save `.parquet` to:
  `dataset/{n_persons}_persons/parquet_files/`

Dependencies are installed locally in `R_libs/` if not available.



## 5️⃣ `ecos.py` — Run prosody analysis via Parole

Calls `parole.sh` for each video using the `.mp4` and corresponding `.parquet` coordinates. Generates prosodic curves (pitch, intensity, etc.) aligned to frames.

**Inputs**:

* `--input_folder`: Root folder with dataset
* `--n_persons`: Number of people to select (e.g., 1)

**Output**:

* Parquet files stored in `parquet_speech_analysis/`

**Example**:

```bash
python3 ecos.py \
  --input_folder ./output \
  --n_persons 1
```



## 6️⃣ `apate.py` — Run WhisperX and align words to frames

* Extracts audio from `.mp4` using `ffmpeg`
* Runs `whisperx` for transcription + word timing
* Aligns words with video frames
* Saves frame-level lexical info as `.json`

**Output**:

* `whisperx/` and `whisperx_framer_output/` directories with aligned data

**Example**:

```bash
python3 apate.py \
  --input_folder ./output \
  --n_persons 1
```



## 7️⃣ `hefesto.py` — Merge all modalities into DuckDB

Merges pose data, prosody, and lexical info into a unified DuckDB database. Performs outer joins by `frame`. Skips missing or corrupted files and logs discrepancies.

**Inputs**:

* Pose `.parquet`: `parquet_files/`
* Prosody `.parquet`: `parquet_speech_analysis/`
* Words `.json`: `whisperx_framer_output/`

**Output**:

* `dataset.duckdb` with a `multi_data` table

**Example**:

```bash
python3 hefesto.py \
  --input_folder ./output \
  --n_persons 1 \
  --db_name ./output/dataset.duckdb
```



## 📁 Output Structure

```
output/
├── videos/
│   ├── raw/               # Downloaded videos
│   ├── masked/            # ASD-processed videos
│   ├── discarded/         # Rejected videos
│   └── 1_person/          # Accepted videos (1 visible speaker)
├── dataset/
│   ├── OpenPose/          # OpenPose JSONs
│   ├── 1_persons/
│   │   ├── parquet_files/
│   │   ├── parquet_speech_analysis/
│   │   ├── whisperx/
│   │   └── whisperx_framer_output/
│   └── dataset.duckdb     # Final database
```



## 📌 Notes

* All modules are idempotent and modular — rerun safely with `--resume` or `--start_at_argos`.
* Processing can be parallelized using `concurrent.futures` where applicable.
* All logs and errors are saved for reproducibility.
* This pipeline assumes a well-formed tabulated `txt` and structured `.mp4` filenames.



## 👥 Authors

Developed by **DaedalusLAB**
For multimodal linguistic research, gesture-prosody alignment, and AI-enhanced corpus construction.




