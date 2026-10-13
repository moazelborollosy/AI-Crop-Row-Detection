"""Streamlit front end:  streamlit run app.py

Every number shown is computed live by the pipeline or read from outputs/metrics.json,
which evaluate.py writes from real held-out data. Nothing is mocked.
"""
import torch  # noqa: F401  must load before pandas/pyarrow: on Windows the reverse order breaks torch's DLLs

import base64
import json
import time
from pathlib import Path

import altair as alt
import cv2
import numpy as np
import streamlit as st

from rowdet import data as D
from rowdet import geometry as G
from rowdet import video as V
from rowdet.pipeline import RowDetector, accuracy, interrow_vegetation, overlay
from rowdet.steering import guidance

ROOT = Path(__file__).resolve().parent
LOCAL_METRICS = ROOT / "outputs" / "metrics.json"
METRICS = LOCAL_METRICS if LOCAL_METRICS.exists() else ROOT / "benchmarks" / "metrics.json"
LABEL = {"classical": "Classical CV", "regression": "CNN regression", "hybrid": "Hybrid (CNN + geometry)",
         "hybrid+video": "Hybrid, video mode (fit starts from previous frame)", "constant": "Constant pattern (no image)"}
HEX = {"classical": "#2a78d6", "regression": "#eb6834", "hybrid": "#1baf7a", "hybrid+video": "#eda100",
       "constant": "#898781"}
ORDER = ["classical", "regression", "hybrid", "hybrid+video", "constant"]

st.set_page_config(page_title="Crop Row Detection | HORSCH LEEB", page_icon="🌱", layout="wide")


def bgr(method):
    h = HEX[method].lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (4, 2, 0))


@st.cache_resource
def detector():
    return RowDetector()


@st.cache_data
def samples():
    if not D.LABELED.exists():
        return {}
    out = {f"HRDED clip {k.replace('_Dataset', '')}": v for k, v in D.load_hrded().items()}
    out["HRSD test split"] = D.load_hrsd()["test"]
    return out


@st.cache_data
def metrics(mtime):
    return json.loads(METRICS.read_text())


def show(img):
    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), use_column_width=True)


@st.cache_data
def demo_videos():
    return V.demo_videos()


@st.cache_data
def labelled_clips():
    return V.labelled_clips()


ACC_KEYS = ("row_err", "crda", "iou", "pixel_acc", "shift_rows", "shift_cm", "heading_deg")
IOU_HELP = ("Detected rows and labelled rows are each drawn as bands of +-0.1 row spacings and compared pixel by pixel. "
            "IoU = overlap / union of the two. Pixel accuracy also counts the soil between rows, so it is always higher.")
SHIFT_HELP = ("Difference between the lateral offset computed from the detected rows and from the labelled rows. "
              "Row spacings come from the image; cm use the row spacing entered in the sidebar.")


def start_video(key, position):
    ss.update(vid_running=True, vid_key=key, vid_pos=position, vid_log=[], vid_acc=[], vid_wall=0.0, vid_error=None)


def stop_video():
    ss.vid_running = False


def show_frame(slot, bgr_img):
    """Push a video frame as an inline JPEG, so no per-frame file piles up in Streamlit's media store."""
    b64 = base64.b64encode(cv2.imencode(".jpg", bgr_img, [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes()).decode()
    slot.markdown(f'<img src="data:image/jpeg;base64,{b64}" style="width:100%;border-radius:6px">', unsafe_allow_html=True)


def show_hud(slot, v, live, min_conf):
    """Per-frame readout. v is the dict stored for the frame currently shown."""
    s = v["steer"]
    with slot.container():
        st.metric("Confidence", f"{v['conf']:.2f}")
        (st.success if v["conf"] >= min_conf else st.error)(s.advice)
        a, b = st.columns(2)
        a.metric("Lateral offset", f"{s.offset_rows:+.2f} rows", f"{s.offset_cm:+.1f} cm (assumed spacing)", delta_color="off")
        b.metric("Row heading", f"{s.heading_deg:+.1f} deg", "assumed FOV", delta_color="off")
        a, b = st.columns(2)
        a.metric("Processing rate", f"{v['fps']:.1f} frames/s", help="Frames analysed and displayed per second, measured over the last 30 frames.")
        b.metric("Latency", f"{v['latency']:.0f} ms", f"detection {v['infer']:.0f} ms", delta_color="off",
                 help="Frame read -> overlay sent to the browser (server side; browser rendering is not included).")
        st.caption(f"Live for {v['time']:.1f} s - camera frames skipped before this one: {v['skipped']}" if live
                   else f"Video position {v['time']:.1f} s - source frames skipped before this one: {v['skipped']}")


det = detector()
methods = list(det.available_methods())
ss = st.session_state
for k, v in (("vid_running", False), ("vid_key", None), ("vid_pos", 0.0), ("vid_log", []), ("vid_acc", []), ("vid_wall", 0.0),
             ("vid_error", None), ("vid_last", None), ("vid_fps", 30.0), ("upload", (None, None))):
    ss.setdefault(k, v)

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Input")
    mode = st.radio("Mode", ["Image", "Video file", "Live camera"], horizontal=True)
    image, gt, name = None, None, ""
    video, live = None, mode == "Live camera"  # video = {key, source, name, seconds}
    step, realtime, start_at, width = 1, False, 0.0, 960
    if mode == "Image":
        source = st.radio("Image source", ["Upload", "Dataset sample"], horizontal=True)
        if source == "Upload":
            up = st.file_uploader("Field image", type=["jpg", "jpeg", "png", "bmp"])
            if up:
                image = cv2.imdecode(np.frombuffer(up.getvalue(), np.uint8), cv2.IMREAD_COLOR)
                name = up.name
        else:
            sets = samples()
            if sets:
                key = st.selectbox("Set", list(sets)) or next(iter(sets))
                i = st.slider("Frame", 0, len(sets[key]) - 1, 0) if len(sets[key]) > 1 else 0
                s = sets[key][i]
                image, gt, name = cv2.imread(str(s.path)), G.canon(s.params), s.path.name
            else:
                st.info("Dataset folder not found - upload an image instead.")
    elif mode == "Video file":
        vsrc = st.radio("Video source", ["iPad demo video", "Labelled clip", "Upload"], horizontal=True,
                        help="Labelled clips have a hand label for every frame, so accuracy is measured when they end.")
        if vsrc == "Labelled clip":
            clips = labelled_clips()
            if clips:
                i = st.selectbox("Clip", range(len(clips)), format_func=lambda i: f"{clips[i]['name']} - {clips[i]['frames']} frames - "
                                 f"{clips[i]['seconds']:.0f} s - held out from training") or 0
                video = {"key": clips[i]["source"], **clips[i]}
            else:
                st.info("Labelled clips not found under data/labeld_data - pick another video source.")
        elif vsrc == "iPad demo video":
            vids = demo_videos()
            if vids:
                i = st.selectbox("Video", range(len(vids)), format_func=lambda i: f"{vids[i]['name'][:8]} - {vids[i]['seconds']:.0f} s - "
                                 + ("frames were used for training" if vids[i]["label_source"] else "never seen by the model")) or 0
                video = {"key": vids[i]["path"], "source": vids[i]["path"], "name": vids[i]["name"], "seconds": vids[i]["seconds"]}
            else:
                st.info("No iPad videos found under data/unlabeld_data - upload a video instead.")
        else:
            upv = st.file_uploader("Video", type=["mp4", "mov", "m4v", "avi", "mkv"])
            if upv:
                key = f"{upv.name}-{upv.size}"
                if ss.upload[0] != key:
                    ss.upload = (key, V.save_upload(upv.name, upv.getbuffer(), ss.upload[1]))
                try:
                    video = {"key": key, "source": ss.upload[1], "name": upv.name, "seconds": V.probe(ss.upload[1])[1]}
                except RuntimeError as e:
                    st.error(str(e))
    else:
        cam = int(st.number_input("Camera index", 0, 9, 0, help="Camera of the computer running this app; 0 is usually the built-in one."))
        video = {"key": f"camera-{cam}", "source": cam, "name": f"camera {cam}", "seconds": 0.0}
    name = video["name"] if video else name
    if mode != "Image":
        st.header("Video processing")
        width = st.selectbox("Processing width (px)", [480, 640, 960, 1280], index=2,
                             help="Frames are downscaled to this width before detection and drawing. Smaller is faster.") or 960
        if not live:
            step = st.slider("Analyse every n-th frame", 1, 10, 3, help="Skipped frames are neither analysed nor shown.")
            realtime = st.checkbox("Follow the video clock", True, help="Play at the video's own speed: drop extra frames when "
                                   "analysis is slower, wait when it is faster. Off: analyse every n-th frame as fast as possible.")
            if video and video["seconds"] > 1 and not video.get("labels"):  # labelled clips always play from frame 0
                start_at = st.slider("Start at (s)", 0.0, float(int(video["seconds"])), 0.0, 1.0)

    st.header("Method")
    method = st.selectbox("Detector", methods, index=len(methods) - 1, format_func=LABEL.get) or methods[-1]
    if not det.has_model:
        st.warning("No trained model at models/rownet.pt - only the classical method is available. Run `python train.py`.")

    st.header("Steering assumptions")
    track = st.radio("Track", ["row", "gap"], horizontal=True,
                     format_func={"row": "On a row", "gap": "Between rows"}.get) or "row"
    spacing = st.number_input("Real row spacing (cm)", 10.0, 150.0, 50.0, 5.0,
                              help="Not in the dataset. Converts the offset from row spacings to cm.")
    hfov = st.number_input("Camera horizontal FOV (deg)", 20.0, 120.0, 60.0, 5.0,
                           help="Not in the dataset. Needed to express the heading in degrees.")
    min_conf = st.slider("Minimum confidence to steer", 0.0, 1.0, 0.35, 0.05)
    show_veg = st.checkbox("Colour vegetation by row membership", False)
    show_gt = st.checkbox("Show ground truth (dataset samples)", True)

st.title("Crop Row Detection")
st.caption("HORSCH LEEB use case 5 - camera image -> crop-row pattern -> steering guidance")

frame_slot = hud_slot = None  # set to the video panel placeholders in video mode
tab_detect, tab_compare, tab_perf, tab_about = st.tabs(["Detect", "Compare approaches", "Held-out performance", "Method & limits"])

if ss.vid_running and (video is None or video["key"] != ss.vid_key):
    ss.vid_running = False  # the source was changed or removed while playing
shown = ss.vid_last if ss.vid_last and video and ss.vid_last["key"] == video["key"] else None
if shown and not ss.vid_running:
    image, gt = shown["raw"], shown.get("gt")  # lets the comparison tab work on the video frame currently shown
results = det.predict(image, methods) if image is not None else {}

# ---------------------------------------------------------------- detect
with tab_detect:
    if mode != "Image":
        st.markdown(f":red[**● LIVE CAMERA**] &nbsp; {name} of this computer, analysed as frames arrive" if live
                    else f":blue[**▶ PRERECORDED VIDEO**] &nbsp; {name or 'no video selected'}")
        b1, b2, note = st.columns([1, 1, 5])
        b1.button("Start", type="primary", use_container_width=True, disabled=ss.vid_running or video is None,
                  on_click=start_video, args=(video["key"] if video else None, start_at))
        b2.button("Stop", use_container_width=True, disabled=not ss.vid_running, on_click=stop_video)
        note.caption("Every frame of this clip has a hand label. Accuracy of the run is shown when the clip ends or is stopped."
                     if video and video.get("labels") else
                     "This input has no labels, so accuracy cannot be measured on it. Pick a labelled clip to measure "
                     "accuracy on video; results on all labelled data are in the Held-out performance tab.")
        if ss.vid_error:
            st.error(ss.vid_error)
        left, right = st.columns([2.2, 1])
        frame_slot, hud_slot = left.empty(), right.empty()
        if not ss.vid_running:
            if shown:
                show_frame(frame_slot, shown["image"])
                show_hud(hud_slot, shown, live, min_conf)
                s = V.summarise(ss.vid_log, ss.vid_wall, ss.vid_fps)
            else:
                frame_slot.info("Press Start. Each analysed frame is shown with its detected rows and steering guidance.")
                s = {}
            if s:
                st.markdown("**Last run (measured)**")
                c = st.columns(4)
                c[0].metric("Frames analysed", s["frames_processed"], f"in {s['wall_seconds']:.1f} s", delta_color="off")
                c[1].metric("Processing rate", f"{s['processing_fps']:.1f} frames/s")
                c[2].metric("Latency (median)", f"{s['latency_ms_median']:.0f} ms", f"95 % under {s['latency_ms_p95']:.0f} ms", delta_color="off")
                c[3].metric("Mean confidence", f"{s['confidence_mean']:.2f}",
                            f"{s['low_confidence_share'] * 100:.0f} % of frames below threshold", delta_color="off")
                if ss.vid_acc:
                    A = {k: np.array([a[i] for a in ss.vid_acc]) for i, k in enumerate(ACC_KEYS)}
                    st.markdown(f"**Accuracy of this run** - {len(ss.vid_acc)} analysed frames compared with their hand labels")
                    c = st.columns(4)
                    c[0].metric("Row segmentation IoU", f"{A['iou'].mean() * 100:.1f} %",
                                f"pixel accuracy {A['pixel_acc'].mean() * 100:.1f} %", delta_color="off", help=IOU_HELP)
                    c[1].metric("Steering shift error", f"{A['shift_rows'].mean():.3f} rows",
                                f"{A['shift_cm'].mean():.1f} cm (assumed spacing), worst {A['shift_cm'].max():.1f} cm",
                                delta_color="off", help=SHIFT_HELP)
                    c[2].metric("Row heading error", f"{A['heading_deg'].mean():.2f} deg", "assumed FOV", delta_color="off")
                    c[3].metric("Row error", f"{A['row_err'].mean():.3f}",
                                f"{(A['row_err'] <= 0.10).mean() * 100:.0f} % of frames within 0.10", delta_color="off",
                                help="Mean distance between detected and labelled rows, in row spacings.")
                used = 100 * (1 - s["source_frames_skipped_share"])
                if live:
                    st.caption(f"{used:.0f} % of the frames the camera delivered were analysed; the rest arrived while a frame was being processed.")
                else:
                    pace = ("faster than real time" if s["playback_speed"] > 1.03 else
                            "in step with real time" if s["playback_speed"] >= 0.97 else "slower than real time")
                    st.caption(f"Played at x{s['playback_speed']:.2f} of the {ss.vid_fps:.0f} fps video's own speed "
                               f"({pace}) while analysing {used:.0f} % of its frames.")
    elif image is None:
        st.info("Upload a field image or pick a dataset sample in the sidebar.")
    else:
        r = results[method]
        h, w = image.shape[:2]
        if abs(w / h - 16 / 9) > 0.25:
            st.warning(f"Image is {w}x{h}. The model was trained on 16:9 forward-looking frames; other formats are untested.")
        steer = guidance(r.params, spacing, hfov, track, r.confidence, min_conf)
        weeds, share = None, 0.0
        if show_veg:
            share, inrow, inter = interrow_vegetation(image, r.params)
            weeds = (inrow, inter)
        left, right = st.columns([2.2, 1])
        with left:
            show(overlay(image, r.params, bgr(method), gt if show_gt else None, weeds, track))
            legend = f"Coloured lines: detected rows ({LABEL[method]}). Magenta marker: tracked target. White tick: image centre."
            if gt is not None and show_gt:
                legend += " Thin white lines: ground truth."
            if show_veg:
                legend += " Green / red: vegetation near / away from a detected row."
            st.caption(f"{name} - {legend}")
        with right:
            st.metric("Confidence", f"{r.confidence:.2f}", help="Heuristic score: how much brighter the row-likelihood map is "
                      "on the detected rows than half-way between them. See the performance tab for how well it tracks error.")
            st.progress(float(min(max(r.confidence, 0.0), 1.0)))
            (st.success if r.confidence >= min_conf else st.error)(steer.advice)
            a, b = st.columns(2)
            a.metric("Lateral offset", f"{steer.offset_rows:+.2f} rows", f"{steer.offset_cm:+.1f} cm (assumed spacing)", delta_color="off")
            b.metric("Row heading", f"{steer.heading_deg:+.1f} deg", "assumed FOV", delta_color="off")
            if gt is not None:
                acc = accuracy(r.params, gt, track, spacing, hfov)
                st.markdown("**Accuracy on this image** (against its hand label)")
                a, b = st.columns(2)
                a.metric("Row segmentation IoU", f"{acc['iou'] * 100:.1f} %", f"pixel accuracy {acc['pixel_acc'] * 100:.1f} %",
                         delta_color="off", help=IOU_HELP)
                b.metric("Steering shift error", f"{acc['shift_rows']:.3f} rows", f"{acc['shift_cm']:.1f} cm (assumed spacing)",
                         delta_color="off", help=SHIFT_HELP)
                a, b = st.columns(2)
                a.metric("Row error vs. label", f"{acc['row_err']:.3f}", "row spacings", delta_color="off")
                b.metric("CRDA vs. label", f"{acc['crda']:.2f}")
            else:
                st.caption("This image has no label, so accuracy cannot be measured on it. Dataset samples have labels.")
            st.caption(f"Runtime {r.ms:.0f} ms on this CPU")
            st.dataframe([{"Vx": round(float(r.params[0]), 3), "Vy": round(float(r.params[1]), 3),
                           "c": round(float(r.params[2]), 3), "d": round(float(r.params[3]), 3)}],
                         hide_index=True, use_container_width=True)
            if show_veg:
                st.metric("Inter-row vegetation", f"{share * 100:.0f} %", help="Share of green pixels farther than 0.25 row "
                          "spacings from a detected row. Indicator only: the dataset has no weed labels to validate it.")
        with st.expander("Row-likelihood map used for this result"):
            heat = cv2.applyColorMap((np.clip(r.prob, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
            show(cv2.resize(heat, (w, h), interpolation=cv2.INTER_LINEAR))
            st.caption("Classical: smoothed excess-green vegetation mask. CNN methods: learned row likelihood.")
            st.json({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.extra.items()})

# ---------------------------------------------------------------- compare
with tab_compare:
    if image is None:
        st.info("Stop the video to compare the approaches on the frame shown." if ss.vid_running
                else "Pick an image, or play and stop a video, to compare the approaches on it.")
    else:
        if mode != "Image" and shown:
            st.caption(f"Video frame at {shown['time']:.1f} s of {shown['name']}")
        cols = st.columns(len(results))
        for col, (m, r) in zip(cols, results.items()):
            with col:
                st.subheader(LABEL[m])
                show(overlay(image, r.params, bgr(m), gt if show_gt else None, None, track))
                line = f"confidence **{r.confidence:.2f}** - {r.ms:.0f} ms"
                if gt is not None:
                    err, crda = G.row_error(r.params, gt)
                    line += f" - row error **{err:.3f}** - CRDA **{crda:.2f}**"
                st.markdown(line)
    st.markdown("""
| | Classical CV | Deep learning (CNN regression) | Hybrid (CNN + geometry) |
|---|---|---|---|
| Idea | Excess-green vegetation mask, vanishing-point pattern fitted to it | Network regresses the 4 pattern parameters end to end | Network predicts a row-likelihood map; geometry fits the pattern to it, starting from the regression head |
| Needs labels | No | Yes | Yes |
| Fails when | Weeds or residue are as green as the crop, crops are tiny | Scene differs from training data; no way to notice | Both heads fail together |
| Confidence | Row contrast of the fit on the vegetation mask | None of its own (scored on the row map) | Row contrast of the fit on the learned row map |
| Output guarantee | Always a valid straight-row pattern | Unconstrained numbers | Always a valid straight-row pattern |

Measured accuracy and speed of each approach are in the **Held-out performance** tab.
""")

# ---------------------------------------------------------------- performance
with tab_perf:
    if not METRICS.exists():
        st.info("No evaluation results yet. Run `python evaluate.py` to produce outputs/metrics.json.")
    else:
        M = metrics(METRICS.stat().st_mtime)
        names = {"hrded": "HRDED clips (time-disjoint from training)", "hrsd_test": "HRSD test split (same videos as training)",
                 "hrded_unseen_videos": "HRDED clips from videos never used for training"}
        ds = st.selectbox("Held-out set", sorted(M["datasets"], key=lambda k: k != "hrded"),
                          format_func=lambda k: names.get(k, k))
        res = {m: M["datasets"][ds][m] for m in ORDER if m in M["datasets"][ds]}
        n = next(iter(res.values()))["n"]
        st.caption(f"{n} frames - evaluated {M['meta']['created']} - row error is the mean distance between predicted and "
                   "labelled rows in units of the row spacing (0 = perfect, 0.25 = chance for a random lateral shift).")

        best = min((m for m in res if m != "constant"), key=lambda m: res[m]["row_err_mean"])
        c = st.columns(4)
        c[0].metric("Best method", LABEL[best])
        c[1].metric("Mean row error", f"{res[best]['row_err_mean']:.3f}")
        c[2].metric("Frames within 0.10 spacing", f"{res[best]['acc_0.10'] * 100:.0f} %")
        c[3].metric("Runtime per frame", f"{res[best]['ms_mean']:.0f} ms")

        domain = list(res)
        color = alt.Color("method:N", scale=alt.Scale(domain=[LABEL[m] for m in domain], range=[HEX[m] for m in domain]),
                          legend=alt.Legend(title=None, orient="top"))
        left, right = st.columns(2)
        with left:
            st.markdown("**Mean row error by method** (lower is better)")
            rows = [{"method": LABEL[m], "error": s["row_err_mean"]} for m, s in res.items()]
            base = alt.Chart(alt.Data(values=rows)).encode(
                y=alt.Y("method:N", sort=[LABEL[m] for m in domain], title=None, axis=alt.Axis(labelLimit=220)),
                x=alt.X("error:Q", title="Mean row error (row spacings)"),
                tooltip=[alt.Tooltip("method:N"), alt.Tooltip("error:Q", format=".3f")])
            bars = base.mark_bar(cornerRadiusEnd=4, height=22).encode(color=color.legend(None))
            text = base.mark_text(align="left", dx=6).encode(text=alt.Text("error:Q", format=".3f"))
            st.altair_chart((bars + text).properties(height=240), use_container_width=True)
        with right:
            st.markdown("**Share of frames at or below a row-error threshold** (higher is better)")
            rows = [{"method": LABEL[m], "threshold": t, "share": v}
                    for m, s in res.items() for t, v in zip(M["meta"]["thresholds"], s["cdf"])]
            base = alt.Chart(alt.Data(values=rows)).encode(
                x=alt.X("threshold:Q", title="Row error threshold (row spacings)"),
                y=alt.Y("share:Q", title="Share of frames", axis=alt.Axis(format="%")), color=color)
            pts = base.mark_point(size=160, opacity=0).encode(
                tooltip=[alt.Tooltip("method:N"), alt.Tooltip("threshold:Q", format=".2f"), alt.Tooltip("share:Q", format=".1%")])
            st.altair_chart((base.mark_line(strokeWidth=2) + pts).properties(height=240), use_container_width=True)

        st.dataframe([{"Method": LABEL[m], "Frames": s["n"], "Row error (mean)": round(s["row_err_mean"], 3),
                       "Row error (median)": round(s["row_err_median"], 3), "Row error (p90)": round(s["row_err_p90"], 3),
                       "CRDA": round(s["crda"], 3), "Within 0.10": f"{s['acc_0.10'] * 100:.1f} %",
                       "Within 0.25": f"{s['acc_0.25'] * 100:.1f} %", "Lateral offset MAE (rows)": round(s["lat_mae_rows"], 3),
                       "Vx MAE": round(s["vx_mae"], 3), "Spacing error": f"{s['d_rel_mae'] * 100:.1f} %",
                       "ms / frame": round(s["ms_mean"])} for m, s in res.items()],
                     hide_index=True, use_container_width=True)

        conf = [{"Method": LABEL[m], "Spearman (confidence vs. error)": round(s["confidence"]["spearman_conf_vs_err"], 2),
                 "Row error, confident half": round(s["confidence"]["row_err_top_half"], 3),
                 "Row error, unsure half": round(s["confidence"]["row_err_bottom_half"], 3),
                 "Frames with confidence >= 0.35": f"{s['confidence']['coverage_at_0.35'] * 100:.0f} %"}
                for m, s in res.items() if "confidence" in s ]
        if conf:
            st.markdown("**Does the confidence score track the error?** A negative correlation means it does.")
            st.dataframe(conf, hide_index=True, use_container_width=True)

        if ds == "hrded" and "hrded_per_clip" in M:
            st.markdown("**Mean row error per clip**")
            st.dataframe([{"Clip": clip.replace("_Dataset", ""), "Frames": next(iter(v.values()))["n"],
                           **{LABEL[m]: round(v[m]["row_err_mean"], 3) for m in ORDER if m in v}}
                          for clip, v in M["hrded_per_clip"].items()], hide_index=True, use_container_width=True)
        with st.expander("Model and evaluation details"):
            st.json(M["meta"])

# ---------------------------------------------------------------- about
with tab_about:
    st.markdown("""
**Pipeline.** Image -> row-likelihood map -> straight-row pattern through one vanishing point `(Vx, Vy, c, d)`
-> confidence -> steering guidance. The pattern is the label format of the HORSCH datasets, so predictions are scored
directly against the provided labels.

**What is learned and what is rule-based.** The row-likelihood map and the regressed pattern come from a small CNN
(MobileNetV3-small encoder, two heads) trained on the HRSD training split. The pattern fit, confidence score,
steering law and the vegetation colouring are rule-based.

**Steering guidance.** The lateral offset in row spacings and the vanishing-point position follow from the image alone.
Centimetres and degrees use the row spacing and field of view entered in the sidebar; the dataset contains no camera
calibration, mounting geometry or steering ground truth, so the steering command is an untuned illustration.

**Known limits.**
- All labelled data comes from five iPad videos of sugar beet recorded on three days in May 2024. Performance on other
  crops, growth stages, cameras or lighting is untested.
- Rows are modelled as straight and evenly spaced; curved rows and headlands are outside the model.
- The evaluation clips come from the same videos as the training frames (different time windows), so the numbers
  describe generalisation within those fields, not to new fields.
- The inter-row vegetation indicator has no weed labels behind it.

**Video and live camera.** Frames are analysed one after another with the same detector. For the hybrid method the fit
starts from the previous frame's rows, which is what makes video faster than single images. Unlabelled video gets no
accuracy figures; the rate and latency shown are measured on this computer during the run.
""")

# ---------------------------------------------------------------- video loop
# Runs last so every tab above is already rendered. Stop (or any widget change) reruns the
# script, which interrupts this loop; the finally block then frees the file or camera.
if mode != "Image" and ss.vid_running and video is not None:
    try:
        src = V.VideoSource(video["source"], live=live, fps=video.get("fps"))
    except RuntimeError as e:
        ss.vid_running, ss.vid_error = False, str(e)
        st.rerun()
    try:
        src.seek(ss.vid_pos)
        ss.vid_fps = src.fps
        base, t0, got_frame = ss.vid_wall, time.perf_counter(), False
        for f in V.stream(src, det, method, step, width, realtime, track=track, row_spacing_cm=spacing,
                          hfov_deg=hfov, min_confidence=min_conf, color=bgr(method)):
            td = time.perf_counter()
            show_frame(frame_slot, f.image)
            labels = video.get("labels")
            frame_gt = G.canon(labels[f.index]) if labels and f.index < len(labels) else None
            if frame_gt is not None:
                acc = accuracy(f.result.params, frame_gt, track, spacing, hfov)
                ss.vid_acc.append(tuple(acc[k] for k in ACC_KEYS))
            v = {"key": video["key"], "name": name, "raw": f.raw, "image": f.image, "time": f.time_s, "gt": frame_gt,
                 "conf": f.result.confidence, "steer": f.steering, "fps": f.fps, "infer": f.infer_ms,
                 "skipped": f.skipped, "latency": f.latency_ms + (time.perf_counter() - td) * 1e3}
            show_hud(hud_slot, v, live, min_conf)
            ss.vid_last = v
            ss.vid_log.append((v["latency"], f.infer_ms, v["conf"], f.skipped, v["conf"] < min_conf))
            ss.vid_wall = base + time.perf_counter() - t0
            if not live:
                ss.vid_pos = f.time_s
            got_frame = True
    finally:
        src.release()
    ss.vid_running = False  # the file ended or the camera stopped delivering frames
    if not got_frame:
        ss.vid_error = f"{name} opened but delivered no frames." + (" Another app may be using the camera." if live else "")
    st.rerun()
