# app.py — SafeCut Pro (Ultra-Precise Blood & Violence Detection | Clean Build)

import os, json, tempfile, subprocess, shutil
from pathlib import Path
from collections import deque

import numpy as np
import pandas as pd
import cv2
from PIL import Image

import streamlit as st
import matplotlib.pyplot as plt

# ========================= Optional HF model =========================
try:
    from transformers import pipeline
    HF_AVAILABLE = True
except Exception:
    HF_AVAILABLE = False
    pipeline = None

@st.cache_resource
def load_violence_model():
    """Optional classifier; we won’t depend on it."""
    if not HF_AVAILABLE or pipeline is None:
        return None
    try:
        # lightweight fallbacks (not really "violence", just signal help)
        for m in ["dima806/facial_emotions_image_detection", "nateraw/vit-age-classifier"]:
            try:
                return pipeline("image-classification", model=m)
            except Exception:
                continue
        return None
    except Exception:
        return None

# ========================= FFmpeg utils =========================
def ensure_ffmpeg():
    return shutil.which("ffmpeg") is not None

def ffprobe_duration(p):
    try:
        out = subprocess.check_output(
            ["ffprobe","-v","error","-show_entries","format=duration","-of","default=nw=1:nk=1", p],
            timeout=10
        ).decode().strip()
        return float(out)
    except Exception:
        return None

def process_video_with_filters(input_path, output_path, vf, af, cuts):
    """
    Two-stage:
      1) optional blur/mute pass
      2) real cuts via trim/concat
    """
    temp_dir = Path(output_path).parent
    current_input = input_path

    # Pass-1: blur/mute if needed
    if vf or af:
        temp_filtered = str(temp_dir / "temp_filtered.mp4")
        cmd = ["ffmpeg","-y","-i",current_input,"-loglevel","error"]
        if vf: cmd += ["-vf", vf]
        if af: cmd += ["-af", af]
        cmd += ["-c:v","libx264","-preset","medium","-crf","23","-c:a","aac","-b:a","192k", temp_filtered]
        r = subprocess.run(cmd, capture_output=True)
        if r.returncode != 0: return r
        current_input = temp_filtered

    # Pass-2: real cuts (optional)
    if cuts:
        dur = ffprobe_duration(current_input)
        if not dur:
            return type("R",(object,),{"returncode":1,"stderr":b"ffprobe failed"})()
        keeps = calculate_keep_intervals(dur, cuts)
        if not keeps:
            return type("R",(object,),{"returncode":1,"stderr":b"No keep intervals"})()
        fcx = build_trim_concat_filter(keeps)
        cmd = ["ffmpeg","-y","-i",current_input,"-filter_complex",fcx,
               "-map","[outv]","-map","[outa]","-c:v","libx264","-preset","medium",
               "-crf","23","-c:a","aac","-b:a","192k","-movflags","+faststart","-loglevel","error",output_path]
        return subprocess.run(cmd, capture_output=True)

    # No cuts: copy filtered as output (or original if no vf/af)
    shutil.copy2(current_input, output_path)
    return type("R",(object,),{"returncode":0,"stderr":b""})()

def calculate_keep_intervals(total_duration, cuts):
    cuts = sorted([(max(0,c["start"]), max(0,c["end"])) for c in cuts], key=lambda x:x[0])
    keeps, cursor = [], 0.0
    for s,e in cuts:
        s = max(0.0, min(s, total_duration))
        e = max(0.0, min(e, total_duration))
        if s > cursor: keeps.append((cursor, s))
        cursor = max(cursor, e)
    if cursor < total_duration:
        keeps.append((cursor, total_duration))
    return [(a,b) for (a,b) in keeps if (b-a) > 0.05]

def build_trim_concat_filter(keep_intervals):
    parts=[]
    for i,(s,e) in enumerate(keep_intervals):
        parts.append(
            f"[0:v]trim=start={s:.3f}:end={e:.3f},setpts=PTS-STARTPTS[v{i}];"
            f"[0:a]atrim=start={s:.3f}:end={e:.3f},asetpts=PTS-STARTPTS[a{i}];"
        )
    cat="".join([f"[v{i}][a{i}]" for i in range(len(keep_intervals))])
    parts.append(f"{cat}concat=n={len(keep_intervals)}:v=1:a=1[outv][outa]")
    return "".join(parts)

# ========================= I/O & frames =========================
def save_uploaded_file(uploaded_file, out_path):
    with open(out_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return out_path

def extract_frames(video_path, out_dir, every_sec=1):
    """
    every_sec may be float (e.g., 0.5). We still index by int(second) for timeline.
    """
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    stride = max(1, int(round(fps*float(every_sec))))
    frame_idx, saved = 0, 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        if frame_idx % stride == 0:
            sec = int(frame_idx / fps)  # coarse second bucket
            cv2.imwrite(str(Path(out_dir)/f"frame_{sec:05d}.jpg"), frame)
            saved += 1
        frame_idx += 1
    cap.release()
    return saved, int(fps)

# ========================= Blood / violence scoring =========================
def detect_red_colors_advanced(img_bgr):
    if img_bgr is None or img_bgr.size == 0:
        return None, 0.0
    h, w = img_bgr.shape[:2]
    hsv  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    lab  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    ycc  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)

    # HSV ranges (wider V)
    mask_hsv1 = cv2.inRange(hsv, np.array([ 0,  50,  30],np.uint8), np.array([15,255,255],np.uint8))
    mask_hsv2 = cv2.inRange(hsv, np.array([ 0, 120,  80],np.uint8), np.array([10,255,255],np.uint8))
    mask_hsv3 = cv2.inRange(hsv, np.array([165,120, 80],np.uint8), np.array([180,255,255],np.uint8))

    L,A,B = cv2.split(lab)
    mask_lab = cv2.bitwise_and(cv2.inRange(A,140,255), cv2.inRange(L,10,170))

    Y,Cr,Cb = cv2.split(ycc)
    mask_ycc = cv2.bitwise_and(cv2.inRange(Cr,150,255), cv2.inRange(Cb,65,155))

    b,g,r = cv2.split(img_bgr)
    mask_bgr = cv2.bitwise_and(cv2.compare(r,g,cv2.CMP_GT), cv2.compare(r,b,cv2.CMP_GT))
    mask_bgr = cv2.bitwise_and(mask_bgr, cv2.inRange(r,70,255))

    combined = (
        0.28*mask_hsv1.astype(np.float32) +
        0.22*mask_hsv2.astype(np.float32) +
        0.15*mask_hsv3.astype(np.float32) +
        0.20*mask_lab.astype(np.float32)  +
        0.10*mask_ycc.astype(np.float32)  +
        0.05*mask_bgr.astype(np.float32)
    )
    combined = np.clip(combined, 0, 255).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7))
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel, iterations=1)
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE,kernel, iterations=2)

    red_ratio = float(np.count_nonzero(combined>80)) / (h*w + 1e-6)
    S = hsv[:,:,1]
    high_sat_in_red = float(np.count_nonzero((combined>80) & (S>90))) / (np.count_nonzero(combined>80)+1e-6)
    confidence = min(1.0, red_ratio * 5.0 * (0.5 + 0.5*high_sat_in_red))
    return combined, confidence

def filter_skin_tones(img_bgr, red_mask):
    if img_bgr is None or red_mask is None: return red_mask
    ycc = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
    skin = cv2.inRange(ycc, np.array([0,130,75],np.uint8), np.array([255,180,140],np.uint8))
    hsv  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    keep_red = ((red_mask>0) & (hsv[:,:,1]>120)).astype(np.uint8)*255  # keep high-sat reds
    skin = cv2.bitwise_and(skin, cv2.bitwise_not(keep_red))
    return cv2.bitwise_and(red_mask, red_mask, mask=cv2.bitwise_not(skin))

def analyze_texture_features(img_bgr, mask=None):
    if img_bgr is None: return 0.0
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    if mask is not None: gray = cv2.bitwise_and(gray, gray, mask=mask)
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges>0)/edges.size
    variance = np.var(gray[gray>0]) if np.any(gray>0) else 0.0
    variance_score = min(variance/1000.0, 1.0)
    texture_score = 0.40*edge_density*5.0 + 0.35*variance_score
    return float(np.clip(texture_score, 0, 1))

def analyze_spatial_distribution(mask):
    if mask is None or np.sum(mask)==0: return 0.0
    contours, _ = cv2.findContours((mask>100).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return 0.0
    areas = [cv2.contourArea(c) for c in contours]
    sig = len([a for a in areas if a>100])
    region_score = min(sig/8.0, 1.0)
    compact = []
    for c in contours:
        a = cv2.contourArea(c)
        if a>50:
            p = cv2.arcLength(c, True)
            if p>0: compact.append(4*np.pi*a/(p*p))
    irregularity = 1.0 - (np.mean(compact) if compact else 1.0)
    return float(np.clip(0.6*region_score + 0.4*irregularity, 0, 1))

def ultra_precise_blood_detection(img_bgr):
    if img_bgr is None or img_bgr.size==0: return 0.0, 0.0, {}
    h,w = img_bgr.shape[:2]
    red_mask, color_conf = detect_red_colors_advanced(img_bgr)
    if red_mask is None: return 0.0, 0.0, {}
    filtered = filter_skin_tones(img_bgr, red_mask)
    txt = analyze_texture_features(img_bgr, filtered)
    sp  = analyze_spatial_distribution(filtered)

    blood_px = float(np.count_nonzero(filtered>80))
    total_px = float(h*w)
    blood_ratio = blood_px/(total_px+1e-6)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    S,V = hsv[:,:,1], hsv[:,:,2]
    high_sat_ratio = float(np.count_nonzero((filtered>80) & (S>100))) / (blood_px+1e-6)
    appropriate_v  = float(np.count_nonzero((filtered>80) & (V>25) & (V<245))) / (blood_px+1e-6)

    confidence_bits = [
        blood_ratio > 0.010,
        color_conf  > 0.25,
        txt         > 0.18,
        sp          > 0.15,
        high_sat_ratio > 0.30,
        appropriate_v  > 0.35
    ]
    conf = sum(confidence_bits)/len(confidence_bits)

    blood_score = (
        0.45*min(blood_ratio*12.0,1.0) +
        0.25*color_conf +
        0.18*txt +
        0.10*sp  +
        0.02*high_sat_ratio
    ) * (0.30 + 0.70*conf)

    details = {
        "blood_ratio": float(blood_ratio),
        "color_confidence": float(color_conf),
        "texture_score": float(txt),
        "spatial_score": float(sp),
        "saturation_ratio": float(high_sat_ratio),
        "confidence_factors": int(sum(confidence_bits)),
    }
    return float(np.clip(blood_score,0,1)), float(conf), details

def detect_violence_motion(img_bgr, prev_frames):
    if img_bgr is None or len(prev_frames)==0: return 0.0
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    prev = cv2.cvtColor(prev_frames[-1], cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(gray, prev)
    motion = min(np.mean(diff)/255.0*3.0, 1.0)
    edges = cv2.Canny(gray,100,200)
    chaos = min(np.sum(edges>0)/edges.size*8.0, 1.0)
    return float(np.clip(0.6*motion + 0.4*chaos, 0, 1))

# ========================= Temporal processing =========================
def apply_temporal_smoothing(scores, window=7):
    if len(scores)<window: return scores
    half = window//2
    out=[]
    for i in range(len(scores)):
        s=max(0,i-half); e=min(len(scores),i+half+1)
        local = scores[s:e]
        lmean = float(np.mean(local)); lmax = float(np.max(local))
        if scores[i] >= lmax*0.8: out.append(0.65*scores[i]+0.35*lmean)
        else:                     out.append(0.35*scores[i]+0.65*lmean)
    return out

def validate_temporal_consistency(scores, window=5, threshold=0.4):
    if len(scores)<window: return scores
    half=window//2; out=[]
    for i in range(len(scores)):
        if scores[i] < threshold: out.append(scores[i]); continue
        s=max(0,i-half); e=min(len(scores),i+half+1)
        neigh = scores[s:e]
        support = sum(1 for v in neigh if v > threshold*0.6)
        out.append(scores[i] if support >= len(neigh)*0.4 else scores[i]*0.3)
    return out

def calculate_adaptive_thresholds(scores, strict_mode=True):
    if len(scores)==0: return 0.65,0.40,0.35,0.25
    arr = np.array(scores, dtype=float)
    mean,std = float(np.mean(arr)), float(np.std(arr))
    median   = float(np.median(arr))
    q75,q85,q95 = [float(np.percentile(arr,p)) for p in (75,85,95)]
    maxv = float(np.max(arr))
    if strict_mode:
        HIGH_ON = min(max(max(q95, mean+3.0*std, median+0.35, 0.60), 0.55), 0.85)
        MED_ON  = min(max(max(q85, mean+1.8*std, median+0.20, 0.35), 0.30), HIGH_ON*0.85)
    else:
        HIGH_ON = min(max(max(q85, mean+2.0*std, median+0.28, 0.50), 0.45), 0.80)
        MED_ON  = min(max(max(q75, mean+1.2*std, median+0.16, 0.28), 0.25), HIGH_ON*0.80)
    if maxv > 0.55:
        HIGH_ON = min(HIGH_ON, maxv*0.95)
        MED_ON  = min(MED_ON,  max(0.30, maxv*0.70))
    HIGH_OFF = max(0.15, min(HIGH_ON*0.65, HIGH_ON-0.05))
    MED_OFF  = max(0.10, min(MED_ON *0.65,  MED_ON -0.04))
    return HIGH_ON, HIGH_OFF, MED_ON, MED_OFF

def hysteresis_state_machine(scores, on_thr, off_thr):
    states=[]; active=False
    for s in scores:
        if not active and s>=on_thr: active=True
        elif active and s<=off_thr:  active=False
        states.append(1 if active else 0)
    return states

def clean_binary_sequence(sequence, min_on_length=3, min_off_gap=2):
    seq = sequence[:]; n=len(seq)
    i=0
    while i<n:
        if seq[i]==1:
            j=i
            while j<n and seq[j]==1: j+=1
            if (j-i)<min_on_length: seq[i:j]=[0]*(j-i)
            i=j
        else: i+=1
    i=0
    while i<n:
        if seq[i]==0:
            j=i
            while j<n and seq[j]==0: j+=1
            before=(i>0 and seq[i-1]==1); after=(j<n and seq[j]==1)
            if before and after and (j-i)<min_off_gap: seq[i:j]=[1]*(j-i)
            i=j
        else: i+=1
    return seq

# ========================= Segments & filters =========================
def merge_detection_segments(decisions, sec_per_frame=1.0, min_duration=2.0):
    segs=[]; cur=None
    for sec in sorted(decisions.keys()):
        dec = decisions[sec]
        if dec["action"]=="keep":
            if cur: segs.append(cur); cur=None
            continue
        if not cur:
            cur={"start":sec, "end":sec+sec_per_frame,
                 "action":dec["action"], "severity":dec["severity"],
                 "max_score":dec["score"]}
        else:
            if dec["action"]==cur["action"]:
                cur["end"]=sec+sec_per_frame
                cur["max_score"]=max(cur["max_score"], dec["score"])
                if dec["severity"]=="high": cur["severity"]="high"
            else:
                segs.append(cur)
                cur={"start":sec, "end":sec+sec_per_frame,
                     "action":dec["action"], "severity":dec["severity"],
                     "max_score":dec["score"]}
    if cur: segs.append(cur)
    return [s for s in segs if (s["end"]-s["start"])>=min_duration]

def build_ffmpeg_filter_string(segments, margin_sec=0.5):
    vf,af,cut=[],[],[]
    for seg in segments:
        t0=max(0, seg["start"]-margin_sec); t1=seg["end"]
        action=seg["action"]; severity=seg["severity"]
        if action in ("blur","blur_mute"):
            blur_strength = 35 if severity=="high" else 25
            vf.append(f"boxblur={blur_strength}:enable='between(t,{t0:.2f},{t1:.2f})'")
            if action=="blur_mute":
                af.append(f"volume=enable='between(t,{t0:.2f},{t1:.2f})':volume=0")
        elif action in ("cut","cut_blur"):
            if action=="cut_blur":
                vf.append(f"boxblur=40:enable='between(t,{t0:.2f},{t1:.2f})'")
            cut.append({"start":t0,"end":t1})
    return (",".join(vf) if vf else None,
            ",".join(af) if af else None,
            cut)

# ========================= Streamlit UI =========================
st.set_page_config(page_title="SafeCut Pro", page_icon="🎬", layout="centered")

st.markdown("""
<style>
.main {max-width: 1100px; margin: 0 auto;}
:root { --bg:#0a0e1a; --card:#111827; --accent:#6366f1; --success:#10b981; --warning:#f59e0b; --danger:#dc2626;}
html, body, .stApp {background: var(--bg); color: #e5e7eb;}
section[data-testid="stSidebar"] {background: #0d1117;}
.sc-card {background: var(--card); border:1px solid #1f2937; border-radius:12px; padding:24px; margin:20px 0; box-shadow:0 4px 16px rgba(0,0,0,0.4);}
.sc-badge {display:inline-block; padding:8px 16px; border-radius:20px; font-size:14px; font-weight:600; margin:6px;}
.sc-badge.red {background: var(--danger); color: white;}
.sc-badge.yellow {background: var(--warning); color: white;}
.sc-badge.green {background: var(--success); color: white;}
.sc-badge.blue {background: #3b82f6; color: white;}
.metric-card {background: linear-gradient(135deg,#1f2937 0%,#111827 100%); border:1px solid #374151; border-radius:10px; padding:20px; text-align:center;}
.metric-value {font-size:32px; font-weight:700; margin:10px 0;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="sc-card">
  <h1 style="color:#6366f1; text-shadow:0 0 20px rgba(99,102,241,.3)">🎬 SafeCut Pro — Ultra-Precise Detection</h1>
  <p style="color:#9ca3af">Detect & remove/blur only bloody/violent shots. No manual thresholds.</p>
  <span class="sc-badge yellow">Violence/Blood (Auto)</span>
</div>
""", unsafe_allow_html=True)

if not ensure_ffmpeg():
    st.error("⚠️ FFmpeg not found. Install it and ensure it's on PATH.")
    st.stop()

st.markdown("### 📤 Upload Video")
with st.container():
    st.markdown("<div class='sc-card'>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Choose a video", type=["mp4","mov","avi","mkv"])
    c1,c2 = st.columns(2)
    with c1:
        sample_rate = st.selectbox("Frame sampling", options=[0.5,1,2], index=1, format_func=lambda x: f"Every {x} sec")
        strict_mode = st.checkbox("🎯 Strict Mode (recommended)", value=True)
    with c2:
        medium_action = st.selectbox("Action for MEDIUM", options=["Blur + Mute","Cut Completely"], index=0)
        high_action   = st.selectbox("Action for HIGH",   options=["Cut Completely","Cut + Blur Margins"], index=0)
    st.markdown("</div>", unsafe_allow_html=True)

# ========================= Main pipeline =========================
if uploaded_file is not None:
    tmpdir = tempfile.mkdtemp(prefix="safecut_")
    in_path = str(Path(tmpdir)/"input.mp4")
    with st.spinner("📥 Loading video..."):
        save_uploaded_file(uploaded_file, in_path)

    st.markdown("### 🎥 Original")
    st.video(uploaded_file)

    st.markdown("### 🎞️ Extracting Frames")
    with st.spinner(f"Extracting frames (1 per {sample_rate}s)..."):
        frames_dir = str(Path(tmpdir)/"frames")
        saved, fps = extract_frames(in_path, frames_dir, every_sec=sample_rate)
    if saved==0:
        st.error("❌ Failed to extract frames.")
        st.stop()
    st.success(f"✅ Extracted {saved} frames (FPS: {fps})")

    violence_model = load_violence_model()

    # --------- Analyze frames ---------
    st.markdown("### 🔬 Analyzing Content")
    progress = st.progress(0); status = st.empty()
    results=[]; hist=deque(maxlen=3)
    frame_paths = sorted(Path(frames_dir).glob("*.jpg"))

    for i, fp in enumerate(frame_paths):
        progress.progress((i+1)/len(frame_paths))
        status.text(f"Analyzing {i+1}/{len(frame_paths)}")
        sec = int(Path(fp).stem.split("_")[-1])
        img = cv2.imread(str(fp))
        if img is None: continue

        blood_score, blood_conf, blood_det = ultra_precise_blood_detection(img)
        v_motion = detect_violence_motion(img, list(hist))

        ai_score = 0.0
        if violence_model:
            try:
                preds = violence_model(str(fp))
                for p in preds:
                    label = p.get("label","").lower()
                    if any(k in label for k in ["angry","fear","sad"]):
                        ai_score = max(ai_score, float(p.get("score",0.0))*0.5)
            except Exception:
                pass

        combined = float(np.clip(0.55*blood_score + 0.25*v_motion + 0.20*ai_score, 0, 1))

        row = {
            "second": int(sec),
            "blood_score": float(blood_score),
            "blood_confidence": float(blood_conf),
            "violence_motion": float(v_motion),
            "ai_score": float(ai_score),
            "combined_score": combined,
            **blood_det
        }
        results.append(row)
        hist.append(img)

    progress.empty(); status.empty()
    if not results:
        st.error("❌ No frames analyzed.")
        st.stop()

    df = pd.DataFrame(results).sort_values("second").reset_index(drop=True)

    # --------- Temporal pipeline ---------
    st.markdown("### 🔄 Temporal Analysis")
    raw_scores = df["combined_score"].tolist()
    smooth = apply_temporal_smoothing(raw_scores, window=7)
    validated = validate_temporal_consistency(smooth, window=5, threshold=0.4)
    df["final_score"] = validated

    HIGH_ON, HIGH_OFF, MED_ON, MED_OFF = calculate_adaptive_thresholds(validated, strict_mode=strict_mode)
    st.success("✅ Temporal analysis complete")

    # Hysteresis masks
    high_mask = hysteresis_state_machine(validated, HIGH_ON, HIGH_OFF)
    med_mask  = hysteresis_state_machine(validated, MED_ON,  MED_OFF)
    high_mask = clean_binary_sequence(high_mask, min_on_length=3, min_off_gap=2)
    med_mask  = clean_binary_sequence(med_mask,  min_on_length=3, min_off_gap=2)
    # High overrides medium
    for i in range(len(high_mask)):
        if high_mask[i]==1: med_mask[i]=0

    # --------- Build decisions ---------
    action_map_med  = {"Blur + Mute":"blur_mute", "Cut Completely":"cut"}
    action_map_high = {"Cut Completely":"cut", "Cut + Blur Margins":"cut_blur"}

    decisions={}
    secs = df["second"].astype(int).tolist()
    fs   = df["final_score"].to_numpy()
    for idx, sec in enumerate(secs):
        if high_mask[idx]==1:
            severity="high"; action="detected_high"
        elif med_mask[idx]==1:
            severity="medium"; action="detected_medium"
        else:
            severity="none"; action="keep"
        if action=="detected_high":   action = action_map_high[high_action]
        elif action=="detected_medium": action = action_map_med[medium_action]
        decisions[sec]={"severity":severity, "action":action, "score":float(fs[idx])}

    # --------- Merge to segments ---------
    segments = merge_detection_segments(decisions, sec_per_frame=float(sample_rate), min_duration=2.0)

    # Fallback: if nothing, take local peaks as short medium blur (very conservative)
    if not segments:
        vals = df["final_score"].to_numpy()
        medv = float(np.median(vals))
        candidates = [s for s,v in zip(secs, vals) if v >= medv+0.15]
        groups=[]
        if candidates:
            a=candidates[0]; prev=candidates[0]
            for s in candidates[1:]:
                if s-prev <= float(sample_rate): prev=s
                else: groups.append((a,prev)); a=prev=s
            groups.append((a,prev))
        fallback=[]
        for (a,b) in groups:
            dur = (b-a)+float(sample_rate)
            if dur >= max(1.0, 2*float(sample_rate)):
                mask = (df["second"]>=a) & (df["second"]<=b)
                fallback.append({
                    "start": float(a),
                    "end": float(b + float(sample_rate)),
                    "action":"blur_mute",
                    "severity":"medium",
                    "max_score": float(df.loc[mask,"final_score"].max() if mask.any() else (medv+0.15))
                })
        if fallback: segments = fallback

    # --------- Build FFmpeg filters ---------
    vf, af, cuts = build_ffmpeg_filter_string(segments, margin_sec=0.5)

    # --------- Visualization ---------
    st.markdown("### 📊 Detection Thresholds")
    c1,c2,c3,c4 = st.columns(4)
    c1.markdown(f"<div class='metric-card'><div style='color:#dc2626'>HIGH ON</div><div class='metric-value' style='color:#dc2626'>{HIGH_ON:.3f}</div></div>", unsafe_allow_html=True)
    c2.markdown(f"<div class='metric-card'><div style='color:#f59e0b'>MED ON</div><div class='metric-value' style='color:#f59e0b'>{MED_ON:.3f}</div></div>", unsafe_allow_html=True)
    c3.markdown(f"<div class='metric-card'><div style='color:#9ca3af'>HIGH OFF</div><div class='metric-value' style='color:#9ca3af'>{HIGH_OFF:.3f}</div></div>", unsafe_allow_html=True)
    c4.markdown(f"<div class='metric-card'><div style='color:#9ca3af'>MED OFF</div><div class='metric-value' style='color:#9ca3af'>{MED_OFF:.3f}</div></div>", unsafe_allow_html=True)

    st.markdown("### 📈 Detection Analysis")
    fig,(ax1,ax2)=plt.subplots(2,1,figsize=(14,8))
    ax1.plot(df["second"], df["blood_score"], label="Blood", lw=2)
    ax1.plot(df["second"], df["violence_motion"], label="Motion", lw=2)
    ax1.plot(df["second"], df["final_score"], label="Final", lw=3)
    ax1.axhline(HIGH_ON, ls="--", label=f"HIGH {HIGH_ON:.2f}")
    ax1.axhline(MED_ON,  ls="--", label=f"MED  {MED_ON:.2f}")
    ax1.set_xlabel("sec"); ax1.set_ylabel("score"); ax1.grid(alpha=.2); ax1.legend()

    ax2.fill_between(df["second"], 0, high_mask, alpha=.6, label="HIGH")
    ax2.fill_between(df["second"], 0, med_mask,  alpha=.6, label="MED")
    ax2.set_xlabel("sec"); ax2.set_ylabel("active"); ax2.set_ylim(-.1,1.3); ax2.grid(alpha=.2); ax2.legend()
    plt.tight_layout()
    st.pyplot(fig)

    st.markdown("### 🎯 Detected Segments")
    if segments:
        seg_df = pd.DataFrame(segments)
        seg_df["duration"] = seg_df["end"] - seg_df["start"]
        show = seg_df[["start","end","duration","severity","action","max_score"]].copy()
        show.columns = ["Start (s)","End (s)","Duration (s)","Severity","Action","Max Score"]
        st.dataframe(show.style.format({"Start (s)":"{:.2f}","End (s)":"{:.2f}","Duration (s)":"{:.2f}","Max Score":"{:.3f}"}), use_container_width=True)
        st.info(f"📌 Total segments: {len(segments)}")
    else:
        st.markdown("""
        <div class="sc-card" style="text-align:center;">
          <span class="sc-badge green">✅ CLEAN VIDEO</span>
          <h3 style="margin:12px 0; color:#10b981;">No violent segments detected</h3>
        </div>
        """, unsafe_allow_html=True)

    # --------- Stats ---------
    kept_count   = sum(1 for d in decisions.values() if d["action"]=="keep")
    blur_count   = sum(1 for d in decisions.values() if d["action"]=="blur_mute")
    cut_count    = sum(1 for d in decisions.values() if d["action"] in ("cut","cut_blur"))
    total_frames = len(decisions) if len(decisions)>0 else 1
    kept_pct = kept_count/total_frames*100.0
    s1,s2,s3,s4 = st.columns(4)
    s1.markdown(f"<div class='metric-card'><span class='sc-badge green'>KEPT</span><div class='metric-value' style='color:#10b981'>{kept_count}</div><p style='color:#6b7280'>{kept_pct:.1f}%</p></div>", unsafe_allow_html=True)
    s2.markdown(f"<div class='metric-card'><span class='sc-badge yellow'>BLURRED</span><div class='metric-value' style='color:#f59e0b'>{blur_count}</div></div>", unsafe_allow_html=True)
    s3.markdown(f"<div class='metric-card'><span class='sc-badge red'>CUT</span><div class='metric-value' style='color:#dc2626'>{cut_count}</div></div>", unsafe_allow_html=True)
    s4.markdown(f"<div class='metric-card'><span class='sc-badge blue'>SEGMENTS</span><div class='metric-value' style='color:#3b82f6'>{len(segments)}</div></div>", unsafe_allow_html=True)

    # --------- Technical details ---------
    vf_str = vf if vf else "None"
    af_str = af if af else "None"
    with st.expander("🔧 Technical details"):
        st.code(f"""
Thresholds:
  HIGH_ON={HIGH_ON:.4f}, HIGH_OFF={HIGH_OFF:.4f}
  MED_ON ={MED_ON:.4f},  MED_OFF ={MED_OFF:.4f}

Video Filter: {vf_str}
Audio Filter: {af_str}

Cut Segments: {len(cuts)}
{json.dumps(cuts, indent=2) if cuts else "[]"}
""", language="yaml")

    # --------- Export ---------
    st.markdown("### 💾 Export Processed Video")
    if st.button("🎬 Process & Export", type="primary", use_container_width=True):
        out_path = str(Path(tmpdir)/"safecut_output.mp4")
        with st.spinner("⚙️ Processing..."):
            r = process_video_with_filters(in_path, out_path, vf, af, cuts)
        if getattr(r, "returncode", 1)==0 and os.path.exists(out_path):
            st.success("✅ Video processed.")
            st.video(out_path)
            with open(out_path, "rb") as f:
                st.download_button("⬇️ Download Processed", f.read(), file_name="safecut_processed.mp4", mime="video/mp4", use_container_width=True)
            in_dur  = ffprobe_duration(in_path)  or 0
            out_dur = ffprobe_duration(out_path) or 0
            in_mb   = os.path.getsize(in_path)/(1024*1024)
            out_mb  = os.path.getsize(out_path)/(1024*1024)
            c1,c2 = st.columns(2)
            c1.metric("Input",  f"{in_dur:.1f}s",  f"{in_mb:.1f} MB")
            c2.metric("Output", f"{out_dur:.1f}s", f"{out_mb:.1f} MB")
        else:
            st.error("❌ FFmpeg processing failed.")
            if hasattr(r,"stderr") and r.stderr:
                st.code(r.stderr.decode("utf-8","ignore"))

else:
    st.markdown("""
    <div class="sc-card">
      <h2>🚀 How SafeCut Pro Works</h2>
      <ul style="color:#9ca3af; line-height:1.8">
        <li>Advanced red/blood detection (HSV/LAB/YCrCb + texture + spatial)</li>
        <li>Skin-tone filtering to avoid faces/skin false positives</li>
        <li>Temporal smoothing + hysteresis to avoid flicker</li>
        <li>Auto thresholds per-video (no manual sliders)</li>
      </ul>
    </div>
    """, unsafe_allow_html=True)
