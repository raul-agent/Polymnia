<p align="center">
  <img src="assets/logo/polymnia_logo.png" alt="POLYMNIA logo" width="50%">
</p>

# POLYMNIA

**A modular pipeline for multimodal language research.** POLYMNIA combines body,
face and hand keypoints, person-relative coordinates, prosodic features and
frame-aligned words in a queryable DuckDB dataset. An optional vision-language
model (VLM) stage adds seven video-level annotations.

Start with the workflow below. For installation details and compatibility notes,
see [SETUP.md](SETUP.md) (Spanish, with commands and local deployment examples).
Use the repository's commit history to track changes.

## Pipeline at a glance

```text
NewsScape tab file → download → TalkNet-ASD → person filter → OpenPose / dfMaker
                                                         ↓
raw audio ──────────────────────────────────────→ Parole + WhisperX
                                                         ↓
                                                    DuckDB merge
                                                         ↓
                                              optional VLM annotations
```

| Stage | CLI command | Purpose |
|---|---|---|
| Validate / download | `check-tab`, `download` | Check timing fields; retrieve NewsScape clips |
| Identify the speaker | `asd` | Produce speaker-masked videos with TalkNet-ASD |
| Filter clips | `filter` | Select clips meeting the single-visible-person criteria |
| Extract movement | `argos` | Run OpenPose, transform coordinates with dfMaker, classify by person count |
| Extract speech | `voice` | Run Parole (`ecos`) and WhisperX (`apate`) in parallel |
| Assemble the dataset | `merge` | Align pose, prosody and words by frame in DuckDB |
| Annotate videos | `theia` | Query a vision model for scene and speaker attributes |

The filter and OpenPose are different detectors: a clip accepted by `filter` can
still be classified into a multiple-person category by `argos`.

## Installation

The current environment targets **Linux with an NVIDIA GPU**. Python dependencies
are managed with [uv](https://docs.astral.sh/uv/); the project selects Python 3.12
and PyTorch 2.8-compatible CUDA 12.8 builds on non-macOS x86_64/AMD64.
Other platforms are not established as supported end-to-end configurations.

From the repository root:

```bash
uv sync --frozen --extra asd
./bin/polymnia help
./bin/polymnia doctor
```

`uv sync` installs Python dependencies, **not the external tools or model weights**:

| Dependency | Required setup |
|---|---|
| OpenPose | Compiled binary; default root `/opt/openpose`, or `--openpose_path` for `argos` |
| TalkNet-ASD | Separate [RaulKite checkout](https://github.com/RaulKite/TalkNet-ASD) and its face-detector / TalkSet weights; default `../TalkNet-ASD`, override with `POLYMNIA_ASD_PATH`. See its [compatibility fixes](https://github.com/RaulKite/TalkNet-ASD/pull/1). |
| R | `multimolang`, `arrow`, `logger`; local packages in `R_libs/`, configuration in `config_dfMaker.json` |
| Parole / Praat | [Parole](https://github.com/daedalusLAB/parole) under `create_datasets/parole/`, with its dependencies and Praat configured |
| System tools | Bash, `curl`, GNU `timeout`, `ffmpeg`, `ffprobe` and `Rscript` |
| DuckDB CLI | Executable at `tools/bin/duckdb` for `polymnia duckdb`; Python DuckDB is installed by uv |
| Vision endpoint | Optional OpenAI-compatible chat-completions API with image support |

The wrapper sets the Python and R paths and uses the existing uv environment;
no manual virtual-environment activation is needed. Run `doctor` before processing.

## Input format

The downloader reads a **headerless, tab-separated file** with at least six fields:

| Position | Meaning |
|---|---|
| 1 | NewsScape filename/path; basename becomes part of the output clip ID |
| 2 | NewsScape clip/program identifier used by the download service |
| 3 / 4 | Start time: whole seconds / fractional integer |
| 5 / 6 | End time: whole seconds / fractional integer |

Extra fields are ignored by the downloader. The fractional divisor is **1000 when
the normalized filename starts with `2017`, otherwise 100**. This is the literal
current rule, not a general guarantee for later years. The standalone downloader's
padding default is 2.5 seconds on each side; the workflow below specifies it explicitly.

Run `check-tab` first: it checks parsing, reversed intervals and agreement with
ranges embedded in filenames. The download wrapper also checks, but warnings do
not abort downloads. `verify_videos.py` checks the resulting media independently.

## Run the workflow

Run from the repository root with **absolute paths**. Paths and clip filenames
must also be free of whitespace and shell metacharacters: downstream scripts build
unquoted shell commands. Use trusted inputs; replace the path and search label below.

```bash
set -euo pipefail
TAB="/absolute/path/clips.txt"
OUT="$PWD/output"
mkdir -p "$OUT"
./bin/polymnia check-tab "$TAB" --offset 2.5
./bin/polymnia download --txt_file "$TAB" --searchterm climate \
  --offset 2.5 --output_dir "$OUT/videos/raw"
./bin/polymnia python tools/verify_videos.py "$OUT/videos/raw"
./bin/polymnia asd --input_dir "$OUT/videos/raw" --output_dir "$OUT/videos/masked"
./bin/polymnia filter --videos_folder "$OUT/videos/masked" \
  --matched_videos "$OUT/videos/1_person" --discarded_videos "$OUT/videos/discarded"
./bin/polymnia argos --videos_folder "$OUT/videos/1_person" --output_folder "$OUT"

# Process every category actually produced, not a fixed list of person counts.
for category in "$OUT"/dataset/*_persons; do
  [ -d "$category/parquet_files" ] || continue
  n="${category##*/}"; n="${n%_persons}"
  [[ "$n" =~ ^[0-9]+$ ]] || continue
  ./bin/polymnia voice --input_folder "$OUT" --n_persons "$n"
  ./bin/polymnia merge --input_folder "$OUT" --n_persons "$n" \
    --db_name "$OUT/multi_${n}p.duckdb"
done
```

Pose extraction uses the **filtered masked videos** in this example; speech stages
use their matching **raw videos**. `voice` writes `ecos.log` and `apate.log` at the
output root, overwriting those logs for each category. Check outputs as well as
exit codes: earlier stages can discard clips or skip missing files.

Outputs include `videos/{raw,masked,1_person,discarded}/`, OpenPose JSONs in
`dataset/OpenPose/{id}/JSON_FILES/`, and `dataset/{n}_persons/` containing
`parquet_files/`, `parquet_speech_analysis/`, `whisperx/` and
`whisperx_framer_output/`. The example keeps a separate DuckDB file per category.

## Optional body-visibility filter

`argos` can drop clips whose joints of interest are barely visible, before the
dfMaker step. It is off by default; add `--filter-body-visibility` to turn it on.
Point it at the **masked** videos, like a normal `argos` run.

```bash
./bin/polymnia argos --videos_folder "$OUT/videos/1_person" --output_folder "$OUT" \
  --filter-body-visibility --required-body-points 4,7 --min-visible-percent 40
```

Each requested BODY_25 index (default `4,7`, both wrists) must be visible with
confidence at least `--min-keypoint-confidence` (default `0.2`) in at least
`--min-visible-percent` (default `40`) of **all** frames of the source video, as
reported by the video metadata. The percentage is computed per point and the
windows do not have to coincide: a clip showing one wrist in the first half and
the other in the second half passes at 40%. Frames with no person, frames with
missing OpenPose JSON and joints with score `0` count as absent. A frame with
more than one person rejects the clip.

Each clip gets a report in `dataset/body_visibility/{clip}.json`. Completed
measurements include per-point counts, the denominator and configuration; errors
include their cause. Evaluator errors mark statistics as partial, with NULL presence
counts when the frame inventory cannot be validated. Rejected clips keep their video and
OpenPose JSON and never reach dfMaker; a `.parquet` left by an earlier run is
moved to `dataset/body_visibility/rejected/{reason}/parquet_files/` so downstream
stages cannot pick it up. Unusable measurements (failed OpenPose, malformed JSON,
non-finite coordinates, non-BODY_25 skeleton, unexpected frame indices) are
reported as `data_error` and make the run exit non-zero instead of guessing;
ordinary rejections exit `0`. On a gated rerun, the JSON of the previous run is
preserved in `dataset/OpenPose/{clip}/previous_json/` and never deleted, so
coverage is always measured on the frames produced by that invocation.

Wrist detections are a proxy, not proof that the entire hand is visible. Use a fresh
DuckDB file for a filtered rebuild: this gate does not remove previously merged rows.

## Optional visual annotations

After merging, export `THEIA_API_BASE`, `THEIA_MODEL` and `THEIA_API_KEY` into the
process environment. Do not put credentials in Git, shared commands or datasets.
A repository `.env` is ignored by Git, but this version's API-key fallback does
not reliably load a key supplied **only** there; use exported variables.

```bash
./bin/polymnia theia --input_folder "$OUT" --n_persons 1 \
  --db_name "$OUT/multi_1p.duckdb"
```

Theia selects IDs from `multi_data`, samples up to five proportional frame
positions, and makes separate scene (raw) and speaker (masked + raw context)
requests. It validates responses and retries failures; **schema validity is not
annotation accuracy**. Unknown values may be NULL.

| Field | Accepted values (also NULL) |
|---|---|
| `indoor_outdoor` | `indoor`, `outdoor` |
| `show_type` | `news anchor`, `monologue`, `weather report`, `interview` |
| `hands_free`, `screen_interaction` | Boolean |
| `sitting_standing` | `sitting`, `standing` |
| `sex` | `male`, `female` (model estimate, not verified identity) |
| `age` | Integer 0–120 (model estimate) |

Results go to `dataset/{n}_persons/video_annotations.csv`, the `video_annotations`
table, and seven columns in `multi_data`. Inspect `status` / `error` before use.
`--no-db` writes CSV only. Resume skips CSV rows marked `ok` and retries errors;
if all rows are already `ok`, it returns without synchronizing a new database.
For endpoint/model overrides or `--max-tokens`, use
`./bin/polymnia python theia.py --help`: these flags are not forwarded by the
`theia` wrapper. Repeat annotation for each category you want to include.

## Querying and interpreting the data

`multi_data` is **long format**, not one row per frame: pose rows carry clip,
frame, person and keypoint identifiers (`id`, `frame`, `people_id`, `type_points`,
`points`). Speech values repeat across keypoint rows; annotations repeat across
all rows of a clip. Count distinct IDs for clip totals and collapse frame-level
features before statistical analysis. Frame indices are 1-based in the merged pose data.

```sql
SELECT count(*) AS rows, count(DISTINCT id) AS clips FROM multi_data;
SELECT show_type, count(*) AS clips
FROM video_annotations WHERE status = 'ok' GROUP BY show_type;
WITH per_frame AS (
  SELECT id, frame, max(pitch) AS pitch, bool_or(vad) AS vad
  FROM multi_data GROUP BY id, frame
)
SELECT avg(pitch) FILTER (WHERE vad) AS mean_voiced_pitch_hz FROM per_frame;
```

`x` / `y` are pixel coordinates; `nx` / `ny` are dfMaker person-relative
transformed coordinates, **not [0,1] screen coordinates**. Missing keypoints or
anchors can produce NULLs. Parole interpolates prosody onto frame timestamps:
non-null pitch does not prove phonation. VAD is useful for selecting speech, but
is not a guarantee that every pitch estimate is valid.

## Verification and limitations

Run `./bin/polymnia test` for the CPU-focused prosody-alignment and Theia tests
(Theia uses a local fake HTTP server, not a paid endpoint). These do not replace
end-to-end checks of external models, downloads and generated datasets.

- `merge` adapts Parole's nested output into derived flat files expected by
  Hefesto, preserving originals. Missing prosody can skip an entire clip.
- Reruns are stage-specific: **merge appends rows**, so use a fresh database for
  a clean rebuild. Do not assume every stage is idempotent or supports resume.
- Speaker detection and person counting are model decisions, not ground truth.
  Keep raw media and inspect discarded clips before interpreting coverage.
- For new naming conventions, review `config_dfMaker.json` metadata extraction.

## Research context, authors and license

Developed by **DaedalusLAB** for multimodal linguistic research, gesture–prosody
alignment and corpus construction. POLYMNIA treats language as a physical,
dynamic process: movement, acoustic features and lexical timing provide measurable
traces of the interaction among modalities, rather than isolated linguistic units.

The repository is distributed under [GNU GPL v3](LICENSE). External software,
model weights and source media have their own terms; this license does not grant
redistribution rights for NewsScape videos. No release DOI is currently documented
here; cite the repository URL and the exact commit used for reproducibility.
