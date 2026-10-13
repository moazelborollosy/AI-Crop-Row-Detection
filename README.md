# AI Crop Row Detection

Camera based crop row detection for the **HORSCH LEEB Innovation Challenge 2026**. The project estimates straight crop rows from images and video, draws row overlays, reports confidence, and derives illustrative steering guidance. It includes a Streamlit interface and three methods: classical computer vision, CNN regression, and a hybrid of learned row likelihood with geometric fitting.

> **Run it without a dataset or model:** install the dependencies, start the app, upload a field image, and select **Classical CV**. The challenge dataset, raw videos, and trained checkpoint are not included in this repository.

## Features

- Image upload, video upload, and a camera attached to the computer running Streamlit.
- Row overlays, confidence, lateral offset, row heading, processing rate, and latency.
- Side by side comparison of the available detection methods.
- Video mode that initializes each hybrid fit from the previous frame.
- Accuracy views for labelled images and clips when the challenge data is available.

The implementation assumes **straight, regularly spaced rows converging toward a vanishing point**. Its steering output is a demonstration, not a validated vehicle controller.

## Quick start

```bash
git clone https://github.com/moazelborollosy/AI-Crop-Row-Detection.git
cd AI-Crop-Row-Detection
python -m venv .venv
```

Activate the environment with `.venv\Scripts\activate` on Windows PowerShell or `source .venv/bin/activate` on macOS/Linux, then run:

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Open the local URL printed by Streamlit. Select **Image > Upload**, choose a crop field image, and use **Classical CV**. The app offers only this method when `models/rownet.pt` is absent. For video, select **Video file > Upload**; for a local camera, select **Live camera**.

The **Metrics** tab reads the saved evaluation snapshot in [`benchmarks/metrics.json`](benchmarks/metrics.json). If you run `evaluate.py` with the original data, the app uses the newly generated `outputs/metrics.json` instead.

## Detection methods

| Method | Row evidence | Pattern estimate | Checkpoint needed |
| --- | --- | --- | --- |
| Classical CV | Excess green vegetation map with Otsu thresholding | Global geometric fit | No |
| CNN regression | MobileNetV3-small regression head | Direct prediction | Yes |
| Hybrid | CNN row likelihood map | Geometric fit initialized by regression, with global search fallback | Yes |
| Hybrid video mode | CNN row likelihood map | Fit starts from the previous frame's pattern | Yes |

The pattern is `(Vx, Vy, c, d)`: a vanishing point `(Vx, Vy)`, offset `c`, and bottom edge row spacing `d`, all in normalized image coordinates. Rows cross the bottom of the image at `x = Vx + c + k*d` for integer `k`. The fitter scores the likelihood map along predicted rows against the spaces between them.

The main implementation is in [`rowdet/model.py`](rowdet/model.py), [`rowdet/classical.py`](rowdet/classical.py), [`rowdet/fit.py`](rowdet/fit.py), and [`rowdet/pipeline.py`](rowdet/pipeline.py).

## Measured results

The [evaluation snapshot](benchmarks/metrics.json) was produced with a 40 epoch checkpoint trained on 211 HRSD frames after removing frames that overlap the HRDED evaluation clip windows. HRDED covers **1,490 frames from nine labelled clips**. The HRSD test split covers **31 frames** from the provider's random frame split, drawn from the same source videos as training.

**HRDED held out clips**

| Method | Mean row error (lower) | CRDA (higher) | Within 0.10 spacing | Within 0.25 spacing | Lateral offset MAE | Mean detection time |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Constant pattern baseline | 0.274 | 0.089 | 2.3% | 37.9% | 0.254 rows | - |
| Classical CV | 0.042 | 0.789 | 95.9% | 98.9% | 0.033 rows | 167 ms |
| CNN regression | 0.220 | 0.180 | 18.9% | 59.2% | 0.207 rows | 23 ms |
| Hybrid, single image | 0.037 | 0.812 | 97.9% | 100% | 0.031 rows | 59 ms |
| **Hybrid, video mode** | **0.031** | **0.850** | **99.2%** | **100%** | **0.021 rows** | **41 ms** |

**HRSD test split**

| Method | Mean row error (lower) | CRDA (higher) | Within 0.10 spacing |
| --- | ---: | ---: | ---: |
| Constant pattern baseline | 0.227 | 0.183 | 6.5% |
| Classical CV | 0.065 | 0.715 | 80.6% |
| CNN regression | 0.172 | 0.221 | 29.0% |
| **Hybrid, single image** | **0.040** | **0.806** | **93.5%** |

**Metric definitions:** row error is the mean distance between detected and labelled rows in units of true row spacing; lower is better. CRDA is a tolerance based crop row detection accuracy score from 0 to 1; higher is better. “Within 0.10” is the share of frames with mean row error at most one tenth of a row spacing. Timings are detection only on the laptop CPU used for the recorded run, not app frame rates or hardware guarantees. The constant baseline ignores the image.

The hybrid performed best on these clips. Classical CV was also strong because the crop was greener than the soil and weed pressure was light. Direct regression often found spacing and vanishing point but missed the lateral row position. These measurements do **not** establish performance in heavily weeded fields or on a vehicle.

## Training and evaluation

The challenge dataset is not redistributed here. With authorized access to it, place it under `data/` using the names expected by [`rowdet/data.py`](rowdet/data.py):

```text
data/
├── labeld_data/
│   ├── hrsd_segmentation_dataset/
│   │   ├── all_images/labels/labels.lbl
│   │   ├── train_data/image/*.jpg
│   │   ├── validation_data/image/*.jpg
│   │   └── test_data/image/*.jpg
│   └── hrded_evaluation_dataset/
│       └── <clip_name>/
│           ├── images/*.jpg
│           └── labels/labels.lbl
└── unlabeld_data/raw_data/videos_ipad/
    ├── full_videos/*.mp4
    └── clipped_videos/*.mp4
```

The spellings `labeld_data` and `unlabeld_data` match the source dataset and code. `full_videos` enables built in video samples; you can upload your own video without it. Training and held out evaluation need the labelled HRSD and HRDED data.

```bash
python train.py --epochs 40             # writes models/rownet.pt
python evaluate.py                     # writes outputs/metrics.json
python -m unittest discover -s tests -v
```

`train.py --holdout-videos <video-id>` supports leaving source videos out of training. Its `--pseudo` option can add frames produced by `pseudo_label.py`. The generated `models/`, `outputs/`, and `data/` directories are ignored by Git. The benchmark JSON is a saved snapshot; a new evaluation writes into `outputs/`.

For direct use in Python:

```python
import cv2
from rowdet.pipeline import RowDetector

frame = cv2.imread("field.jpg")
detector = RowDetector()
method = "hybrid" if detector.has_model else "classical"
result = detector.predict(frame, (method,))[method]
print(result.params, result.confidence)
```

## Limits and data needs

- The evaluation data contains sugar beet at one growth stage from one camera and five iPad videos. HRDED clips are time disjoint from training frames, but come from the same fields. Performance in new fields is unknown.
- The target use case includes heavily weeded fields, while the available videos have light weed pressure. Dense weeds, crop gaps, curved rows, and headlands need labelled evaluation.
- The fitted pattern assumes straight, evenly spaced rows. Curves and irregular spacing require a different model.
- Confidence is based on row contrast. All held out frames scored above the app's default 0.35 steering threshold, so its rejection behavior has not been established.
- Centimeter offsets and heading angles depend on the row spacing and camera field of view entered in the app. Steering gains, camera calibration, and machine control were not validated.
- The inter row vegetation coloring is an indicator only; there are no weed labels for measuring weed detection accuracy.

## Project structure

| Path | Purpose |
| --- | --- |
| `app.py` | Streamlit image, video, and camera interface |
| `rowdet/` | Geometry, model, classical detector, fitting, pipeline, steering, and video processing |
| `train.py` | Train RowNet from labelled frames |
| `evaluate.py` | Evaluate methods on held out labelled frames |
| `pseudo_label.py`, `label_check.py` | Optional pseudo label and manual label review utilities |
| `benchmarks/metrics.json` | Saved evaluation snapshot shown by the app |
| `tests/` | Geometry, fitting, and pipeline tests |

## Acknowledgment

Developed for the HORSCH LEEB Innovation Challenge 2026, use case 5: crop row detection for crop care machines.
