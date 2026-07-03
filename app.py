# =====================================================================
# ASISTEN NAVIGASI TUNANETRA - STREAMLIT DASHBOARD v4.4 (STABLE)
# =====================================================================
# Basis: v3.9 (stable) + perbaikan:
# 1. Webcam: st.camera_input (works di browser/cloud/local)
# 2. OCR: fuzzy dedup (typo tidak bikin dibaca 2x)
# 3. Deteksi: confidence & area threshold dinaikkan (kurangi false positive)
# 4. Obstacle M2: pakai conf dari model, tidak diturunkan paksa
# 5. detection_history aktif
# 6. Alert: sekali per objek + instruksi jelas
# =====================================================================

import streamlit as st
import cv2, os, re, time, base64, tempfile
import numpy as np
import pandas as pd
from gtts import gTTS
from datetime import datetime
from pathlib import Path
from io import BytesIO
from collections import defaultdict
import logging
import random

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="Asisten Tunanetra",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
html, body, [data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #f5f7ff 0%, #faf8ff 100%) !important;
    font-family: 'Inter', sans-serif;
}
.header-logo {
    width: 50px; height: 50px;
    background: linear-gradient(135deg, #6c3fff 0%, #3f8bff 100%);
    border-radius: 14px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.4rem;
    box-shadow: 0 4px 15px rgba(108, 63, 255, 0.3);
}
.header-text h1 { font-size: 1.8rem; color: #1a1a2e; margin: 0; }
.header-text p { font-size: 0.85rem; color: #888; margin: 0; }
.pills { display: flex; gap: 0.6rem; flex-wrap: wrap; margin-top: 0.8rem; }
.pill { padding: 0.35rem 1rem; border-radius: 20px; font-size: 0.75rem; font-weight: 600; border: 1px solid transparent; }
.pill-run { background: #ede8ff; color: #6c3fff; border-color: #c4b0ff; }
.pill-ok { background: #e6fff5; color: #00955a; border-color: #80ecc0; }
.pill-danger { background: #fff0f0; color: #cc2222; border-color: #ffaaaa; }
.pill-ocr { background: #e8f4ff; color: #0066cc; border-color: #90c8ff; }
.pill-model { background: #f0e8ff; color: #6c3fff; border-color: #c4b0ff; }
.alert-danger { background: #fff0f0; border: 1px solid #ffaaaa; border-left: 4px solid #ff4444; border-radius: 8px; padding: 1rem; color: #cc2222; font-weight: 600; margin: 0.8rem 0; }
.alert-warning { background: #fffbe6; border: 1px solid #ffe58f; border-left: 4px solid #faad14; border-radius: 8px; padding: 1rem; color: #ad6800; font-weight: 600; margin: 0.8rem 0; }
.alert-info { background: #f0f6ff; border: 1px solid #90c8ff; border-left: 4px solid #3f8bff; border-radius: 8px; padding: 1rem; color: #0066cc; margin: 0.8rem 0; }
.alert-success { background: #f0fff8; border: 1px solid #80ecc0; border-left: 4px solid #00c073; border-radius: 8px; padding: 1rem; color: #00955a; margin: 0.8rem 0; }
.ocr-result { background: #f8fbff; border: 2px solid #3f8bff; border-radius: 10px; padding: 1.4rem; color: #0066cc; font-size: 1.05rem; min-height: 60px; }
.stat-box { background: linear-gradient(135deg, #f5f7ff 0%, #faf8ff 100%); border-radius: 10px; padding: 1.2rem; text-align: center; border: 1px solid #e0e4f0; }
.stat-value { font-size: 2rem; font-weight: 700; color: #6c3fff; }
.stat-label { font-size: 0.8rem; color: #999; text-transform: uppercase; letter-spacing: 1px; font-weight: 600; }
.stButton > button { border-radius: 10px; border: none; font-weight: 600; padding: 0.8rem 1.4rem; transition: all 0.2s ease; text-transform: uppercase; letter-spacing: 0.5px; }
.stButton > button:hover { transform: translateY(-2px); box-shadow: 0 4px 12px rgba(108, 63, 255, 0.3); }
#MainMenu, footer { visibility: hidden; display: none; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────────────
session_defaults = {
    'model1': None, 'model2': None, 'model3': None, 'ocr_engine': None,
    'last_frame': None, 'log': [], 'detection_history': [],
    'last_alert_time': defaultdict(lambda: -99.0),
    'alerted_objects': set(),
    'ocr_triggered': False, 'ocr_triggered_cam': False,
    'ocr_frame_count': 0,
    'last_uploaded_name': None,
    'last_ocr_text': '',
}
for key, val in session_defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val

def add_log(msg):
    st.session_state.log.insert(0, (time.strftime("%H:%M:%S"), msg))
    st.session_state.log = st.session_state.log[:30]

# ============================================================
# AUDIO
# ============================================================
def play_audio_safe(placeholder, audio_bytes):
    if audio_bytes:
        b64 = base64.b64encode(audio_bytes).decode()
        uid = f"audio_{int(time.time()*1000)}_{random.randint(1000,9999)}"
        placeholder.markdown(f"""
            <audio autoplay="true" style="display:none;" id="{uid}">
                <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
            </audio>
            <script>setTimeout(function(){{ var e=document.getElementById("{uid}"); if(e) e.play(); }}, 100);</script>
        """, unsafe_allow_html=True)

def get_audio_bytes(text, lang='id'):
    try:
        buf = BytesIO()
        gTTS(text=text, lang=lang, slow=False).write_to_fp(buf)
        buf.seek(0)
        return buf.read()
    except Exception as e:
        logger.error(f"TTS error: {e}")
        return None

# ============================================================
# TERJEMAHAN
# ============================================================
def get_indo_name(name):
    t = {
        'person':'orang', 'car':'mobil', 'bus':'bus', 'truck':'truk',
        'motorcycle':'motor', 'bicycle':'sepeda', 'dog':'anjing', 'cat':'kucing',
        'pothole':'lubang', 'stairs':'tangga', 'obstacle':'rintangan',
        'road-barrier':'pembatas jalan', 'pole':'tiang', 'train':'kereta',
        'stop sign':'rambu stop', 'traffic light':'lampu lalu lintas',
        'sidewalk':'trotoar', 'crosswalk':'jalur penyeberangan', 'tree':'pohon',
        'animal':'hewan', 'vehicle':'kendaraan',
        'dilarang masuk':'dilarang masuk', 'dilarang parkir':'dilarang parkir',
        'dilarang berhenti':'dilarang berhenti', 'hati-hati':'hati-hati',
        'rumah sakit':'rumah sakit', 'masjid':'masjid', 'gereja':'gereja',
        'pom bensin':'pom bensin', 'tempat parkir':'tempat parkir',
        'jalur sepeda':'jalur sepeda', 'batas kecepatan':'batas kecepatan',
        'persimpangan':'persimpangan', 'ikuti arah bundaran':'ikuti arah bundaran'
    }
    n = name.lower()
    if n in t: return t[n]
    for k, v in t.items():
        if k in n or n in k: return v
    return n

def is_rambu(name):
    n = name.lower()
    kw = {'rambu', 'lampu lalu lintas', 'traffic light', 'stop sign', 'dilarang',
          'hati-hati', 'pemberhentian', 'rumah sakit', 'masjid', 'gereja', 'spbu',
          'pom bensin', 'parkir', 'persimpangan', 'bundaran', 'jalur', 'lajur',
          'kecepatan', 'belok', 'berhenti', 'masuk', 'putar balik', 'sepeda', 'kiri'}
    return any(k in n for k in kw)

def generate_rambu_alert(name):
    n = name.lower()
    suara = {
        'stop sign':'Ada rambu stop. Berhenti!',
        'rambu stop':'Ada rambu stop. Berhenti!',
        'dilarang masuk':'Ada rambu dilarang masuk. Jangan masuk!',
        'dilarang parkir':'Ada rambu dilarang parkir',
        'dilarang berhenti':'Ada rambu dilarang berhenti',
        'hati-hati':'Ada rambu hati-hati. Waspada!',
        'lampu lalu lintas':'Ada lampu lalu lintas',
        'rumah sakit':'Ada rumah sakit',
        'masjid':'Ada masjid', 'gereja':'Ada gereja',
        'pom bensin':'Ada pom bensin', 'tempat parkir':'Ada tempat parkir',
        'jalur sepeda':'Ada jalur sepeda', 'batas kecepatan':'Ada batas kecepatan',
        'persimpangan':'Ada persimpangan di depan',
    }
    for k, v in suara.items():
        if k in n or n in k: return v
    return f"Ada {get_indo_name(n)}"

def generate_alert(name, pos_x, area):
    nama = get_indo_name(name)
    if pos_x < 0.3:
        pos, arah = "di sebelah kiri", "Berjalanlah ke kanan"
    elif pos_x > 0.7:
        pos, arah = "di sebelah kanan", "Berjalanlah ke kiri"
    else:
        pos, arah = "di depan Anda", "Berhenti dan belok ke samping"
    if area > 0.15:
        return f"Awas! Ada {nama} sangat dekat {pos}. {arah} sekarang!"
    elif area > 0.08:
        return f"Hati-hati! Ada {nama} {pos}. {arah}."
    else:
        return f"Ada {nama} {pos}. {arah} pelan-pelan."

# ============================================================
# OCR — DARI v3.9 (STABIL) + FUZZY DEDUP
# ============================================================
def perform_ocr_on_frame(frame, ocr_engine, min_conf=0.20):
    if ocr_engine is None: return "OCR engine tidak tersedia"
    try:
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
        h, w = gray.shape
        if w < 400:
            scale = 600 / w
            gray = cv2.resize(gray, (600, int(h * scale)))
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        sharpened = cv2.filter2D(enhanced, -1, kernel)

        results = None
        try: results = ocr_engine.readtext(sharpened, detail=1, paragraph=False)
        except: pass
        if not results or len(results) == 0:
            try: results = ocr_engine.readtext(enhanced, detail=1, paragraph=False)
            except: pass
        if not results or len(results) == 0:
            try: results = ocr_engine.readtext(frame, detail=1, paragraph=False)
            except: pass

        if results and len(results) > 0:
            texts = []
            for (bbox, text, conf) in results:
                text = text.strip()
                text = re.sub(r'[^a-zA-Z0-9\s\.,!?\-:/()]', '', text)
                if len(text) > 1 and conf > min_conf:
                    texts.append(text)
            if texts:
                result_text = ' '.join(texts)
                result_text = ' '.join(result_text.split())
                if len(result_text) > 2:
                    return result_text
        return "Tidak ada teks terdeteksi"
    except Exception as e:
        return f"Error: {str(e)[:30]}"

def texts_are_similar(text1, text2, threshold=0.7):
    """Fuzzy comparison — teks mirip (beda typo) dianggap SAMA"""
    if not text1 or not text2: return False
    t1 = ' '.join(text1.lower().split())
    t2 = ' '.join(text2.lower().split())
    if t1 == t2: return True
    if len(t1) < 3 or len(t2) < 3: return t1 == t2
    def bigrams(s):
        return set(s[i:i+2] for i in range(len(s)-1))
    b1, b2 = bigrams(t1), bigrams(t2)
    if not b1 or not b2: return t1 == t2
    similarity = len(b1 & b2) / len(b1 | b2)
    return similarity >= threshold

# ============================================================
# DETEKSI — THRESHOLD KETAT (KURANGI FALSE POSITIVE)
# ============================================================
def process_frame_detection(frame, model, conf=0.4):
    if model is None: return frame, []
    try:
        results = model.predict(frame, conf=conf, iou=0.45, verbose=False)
        detections = []
        if results and len(results) > 0:
            result = results[0]
            if result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes.xyxy.cpu().numpy()
                confidences = result.boxes.conf.cpu().numpy()
                classes = result.boxes.cls.cpu().numpy().astype(int)
                frame_h, frame_w = frame.shape[:2]
                for box, conf_score, cls_idx in zip(boxes, confidences, classes):
                    x1, y1, x2, y2 = map(int, box)
                    cls_name = result.names[cls_idx]
                    area = (x2 - x1) * (y2 - y1)
                    area_ratio = area / (frame_w * frame_h)
                    pos_x = ((x1 + x2) / 2) / frame_w
                    cls_lower = cls_name.lower()

                    obstacle_kw = ['pothole', 'lubang', 'stairs', 'tangga', 'obstacle',
                                   'rintangan', 'road-barrier', 'pembatas jalan', 'pole', 'tiang']
                    vehicle_kw = ['car', 'mobil', 'bus', 'truck', 'truk', 'vehicle',
                                  'kendaraan', 'motorcycle', 'motor', 'bicycle', 'sepeda',
                                  'train', 'kereta']
                    person_kw = ['person', 'orang']

                    obj_obstacle = any(kw in cls_lower for kw in obstacle_kw)
                    obj_vehicle = any(kw in cls_lower for kw in vehicle_kw)
                    obj_person = any(kw in cls_lower for kw in person_kw)

                    # FILTER KETAT — buang false positive
                    if obj_obstacle:
                        if conf_score < 0.35 or area_ratio < 0.01: continue
                    elif obj_vehicle:
                        if conf_score < 0.45 or area_ratio < 0.02: continue
                    elif obj_person:
                        if conf_score < 0.45 or area_ratio < 0.015: continue
                    else:
                        if area_ratio < 0.015: continue

                    # RISK LEVEL — threshold tinggi
                    if obj_obstacle:
                        if conf_score > 0.50 and area_ratio > 0.08: risk_level = 'BAHAYA'
                        elif conf_score > 0.40 and area_ratio > 0.03: risk_level = 'WASPADA'
                        else: risk_level = 'AMAN'
                    elif obj_vehicle:
                        if conf_score > 0.55 and area_ratio > 0.15: risk_level = 'BAHAYA'
                        elif conf_score > 0.50 and area_ratio > 0.08: risk_level = 'WASPADA'
                        else: risk_level = 'AMAN'
                    elif obj_person:
                        if conf_score > 0.55 and area_ratio > 0.10: risk_level = 'WASPADA'
                        else: risk_level = 'AMAN'
                    else:
                        risk_level = 'AMAN'

                    detections.append({
                        'class': cls_name, 'confidence': float(conf_score),
                        'area_ratio': area_ratio, 'position_x': pos_x,
                        'risk_level': risk_level, 'bbox': (x1, y1, x2, y2),
                        'timestamp': datetime.now()
                    })

                    if risk_level == 'BAHAYA': color = (0, 0, 255)
                    elif risk_level == 'WASPADA': color = (0, 165, 255)
                    else: color = (0, 255, 0)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    label = f"{get_indo_name(cls_name)} {conf_score:.0%}"
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    fs, th = 0.6, 2
                    (tw, tht), _ = cv2.getTextSize(label, font, fs, th)
                    cv2.rectangle(frame, (x1, y1-tht-10), (x1+tw+4, y1), color, -1)
                    cv2.putText(frame, label, (x1+2, y1-5), font, fs, (255,255,255), th)
        return frame, detections
    except Exception as e:
        logger.error(f"Detection error: {e}")
        return frame, []

def process_frame_detection_multi(frame, model1, model2, model3, conf=0.4):
    frame_out = frame.copy()
    all_dets = []
    for mdl in [model1, model2, model3]:
        if mdl is not None:
            frame_out, dets = process_frame_detection(frame_out, mdl, conf)
            all_dets.extend(dets)
    return frame_out, all_dets

# ============================================================
# ALERT — SEKALI PER OBJEK + INSTRUKSI
# ============================================================
def handle_alerts(dets, now_time, enable_audio, alert_cooldown,
                  warn_ph, status_ph, audio_danger_ph, audio_rambu_ph):
    danger = [d for d in dets if d['risk_level'] == 'BAHAYA']
    waspada = [d for d in dets if d['risk_level'] == 'WASPADA']
    rambu_dets = [d for d in dets if is_rambu(d['class'])]

    current = {d['class'] for d in dets}
    gone = st.session_state.alerted_objects - current
    for obj in gone:
        st.session_state.alerted_objects.discard(obj)

    if danger and enable_audio:
        d = danger[0]
        key = d['class']
        if key not in st.session_state.alerted_objects:
            msg = generate_alert(d['class'], d['position_x'], d['area_ratio'])
            warn_ph.markdown(f'<div class="alert-danger">🚨 {msg}</div>', unsafe_allow_html=True)
            audio = get_audio_bytes(msg)
            if audio:
                audio_danger_ph.empty()
                play_audio_safe(audio_danger_ph, audio)
            st.session_state.alerted_objects.add(key)
            add_log(f"BAHAYA: {get_indo_name(key)}")
        else:
            warn_ph.markdown(
                f'<div class="alert-danger">🚨 Masih ada {get_indo_name(d["class"])}. Tetap waspada!</div>',
                unsafe_allow_html=True)
        status_ph.markdown('<div class="pills"><span class="pill pill-run">● AKTIF</span><span class="pill pill-danger">● BAHAYA</span></div>', unsafe_allow_html=True)

    elif waspada and enable_audio:
        d = waspada[0]
        key = f"w_{d['class']}"
        if (now_time - st.session_state.last_alert_time[key]) >= alert_cooldown:
            msg = generate_alert(d['class'], d['position_x'], d['area_ratio'])
            warn_ph.markdown(f'<div class="alert-warning">⚡ {msg}</div>', unsafe_allow_html=True)
            audio = get_audio_bytes(msg)
            if audio:
                audio_danger_ph.empty()
                play_audio_safe(audio_danger_ph, audio)
            st.session_state.last_alert_time[key] = now_time
            add_log(f"WASPADA: {get_indo_name(d['class'])}")
        status_ph.markdown('<div class="pills"><span class="pill pill-run">● AKTIF</span><span class="pill pill-danger" style="background:#fffbe6;color:#ad6800;border-color:#ffe58f;">● WASPADA</span></div>', unsafe_allow_html=True)

    elif rambu_dets and enable_audio:
        d = rambu_dets[0]
        key = f"r_{d['class']}"
        if (now_time - st.session_state.last_alert_time[key]) >= alert_cooldown:
            msg = generate_rambu_alert(d['class'])
            warn_ph.markdown(f'<div class="alert-info">ℹ️ {msg}</div>', unsafe_allow_html=True)
            audio = get_audio_bytes(msg)
            if audio:
                audio_rambu_ph.empty()
                play_audio_safe(audio_rambu_ph, audio)
            st.session_state.last_alert_time[key] = now_time
            add_log(f"RAMBU: {msg}")
        status_ph.markdown('<div class="pills"><span class="pill pill-run">● AKTIF</span><span class="pill pill-ocr">● RAMBU</span></div>', unsafe_allow_html=True)
    else:
        warn_ph.markdown('<div class="alert-success">✅ Aman</div>', unsafe_allow_html=True)
        status_ph.markdown('<div class="pills"><span class="pill pill-run">● AKTIF</span><span class="pill pill-ok">● AMAN</span></div>', unsafe_allow_html=True)
    return danger, rambu_dets

def handle_ocr_dedup(frame, ocr_engine, ocr_min_conf, enable_tts, ocr_ph, audio_ocr_ph):
    text = perform_ocr_on_frame(frame, ocr_engine, ocr_min_conf)
    if text and text != "Tidak ada teks terdeteksi" and len(text) > 5:
        if not texts_are_similar(text, st.session_state.last_ocr_text):
            st.session_state.last_ocr_text = text
            ocr_ph.markdown(f'<div class="ocr-result">📝 {text}</div>', unsafe_allow_html=True)
            add_log(f"OCR: {text[:40]}...")
            if enable_tts:
                audio = get_audio_bytes(f"Ada tulisan: {text}")
                if audio:
                    audio_ocr_ph.empty()
                    play_audio_safe(audio_ocr_ph, audio)
            return True
        else:
            ocr_ph.markdown(
                f'<div class="ocr-result" style="opacity:0.6;">📝 {text}'
                f'<br><small>(Teks serupa, tidak dibaca ulang)</small></div>',
                unsafe_allow_html=True)
    return False

# ─────────────────────────────────────────────────────────────────────
# LOAD MODELS
# ─────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_yolo_models(path1='yolo11s.pt', path2=None, path3=None):
    try:
        from ultralytics import YOLO
        models = {}
        try: models['m1'] = YOLO(path1)
        except: models['m1'] = None
        if path2:
            try: models['m2'] = YOLO(path2)
            except: models['m2'] = None
        else: models['m2'] = None
        if path3:
            try: models['m3'] = YOLO(path3)
            except: models['m3'] = None
        else: models['m3'] = None
        return models
    except: return {}

@st.cache_resource(show_spinner=False)
def load_ocr():
    try:
        import easyocr
        return easyocr.Reader(['id', 'en'], gpu=False)
    except: return None

# ─────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────
c1, c2 = st.columns([0.1, 0.9])
with c1:
    st.markdown('<div class="header-logo">👁️</div>', unsafe_allow_html=True)
with c2:
    st.markdown("""
    <div class="header-text">
        <h1>Asisten Navigasi Tunanetra</h1>
        <p>Deteksi Objek + Baca Teks On-Demand</p>
    </div>
    """, unsafe_allow_html=True)
st.divider()

# ─────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Pengaturan")
    if st.button("📥 Load YOLO", use_container_width=True):
        with st.spinner("Loading..."):
            models = load_yolo_models()
            st.session_state.model1 = models.get('m1')
            add_log("YOLO loaded")
            st.success("✅ YOLO Dimuat!")
    if st.button("📥 Load OCR", use_container_width=True):
        with st.spinner("Loading..."):
            st.session_state.ocr_engine = load_ocr()
            add_log("OCR loaded")
            st.success("✅ OCR Dimuat!")
    st.markdown("---")
    st.info("💡 Upload model (.pt) bisa 1-5 menit.")
    with st.expander("🏔️ Model 2: Tangga/Lubang"):
        up2 = st.file_uploader("Upload best.pt (M2)", type=['pt'], key='m2up')
        if up2 and st.button("Load M2"):
            with st.spinner("Memuat..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as tmp:
                    tmp.write(up2.read())
                from ultralytics import YOLO
                st.session_state.model2 = YOLO(tmp.name)
                add_log("M2 loaded")
                st.success("✅ M2 Dimuat!")
    with st.expander("🚦 Model 3: Rambu"):
        up3 = st.file_uploader("Upload best_rambu.pt (M3)", type=['pt'], key='m3up')
        if up3 and st.button("Load M3"):
            with st.spinner("Memuat..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as tmp:
                    tmp.write(up3.read())
                from ultralytics import YOLO
                st.session_state.model3 = YOLO(tmp.name)
                add_log("M3 loaded")
                st.success("✅ M3 Dimuat!")
    st.markdown("---")
    s1 = "✅" if st.session_state.model1 else "⚠️"
    s2 = "✅" if st.session_state.model2 else "⚠️"
    s3 = "✅" if st.session_state.model3 else "⚠️"
    so = "✅" if st.session_state.ocr_engine else "⚠️"
    st.markdown(f'<div class="pills"><span class="pill pill-model">{s1} M1 Umum</span><span class="pill pill-model">{s2} M2 Rintangan</span><span class="pill pill-model">{s3} M3 Rambu</span><span class="pill pill-model">{so} OCR</span></div>', unsafe_allow_html=True)
    conf_threshold = st.slider("Confidence Deteksi", 0.1, 0.9, 0.4, 0.05)
    enable_audio = st.checkbox("🔊 Audio Alert", value=True)
    alert_cooldown = st.slider("Cooldown Alert (s)", 2, 10, 5)
    ocr_min_conf = st.slider("OCR Confidence", 0.1, 0.9, 0.2, 0.05)
    ocr_scan_interval = st.slider("OCR Interval (frame)", 1, 10, 3)
    enable_tts = st.checkbox("🔊 TTS Baca Teks", value=True)
    show_logs = st.checkbox("📋 Show Logs", value=True)

# ─────────────────────────────────────────────────────────────────────
# MAIN TABS
# ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🎯 Detection", "📖 Text Reading", "📊 Statistics"])

# ═════════════════════════════════════════════════════════════════════
# TAB 1
# ═════════════════════════════════════════════════════════════════════
with tab1:
    mode = st.radio("Mode:", ["📹 Webcam", "📤 Upload Video"], horizontal=True)
    st.divider()

    if mode == "📹 Webcam":
        st.markdown('<div class="alert-info">📸 Ambil foto dari kamera lalu otomatis dianalisis. Klik "Clear photo" untuk scan ulang.</div>', unsafe_allow_html=True)
        ocr_cam_on = st.checkbox("📖 Baca Teks (OCR)", value=False)
        frame_ph = st.empty()
        status_ph = st.empty()
        warn_ph = st.empty()
        ocr_ph = st.empty()
        audio_danger_ph = st.empty()
        audio_rambu_ph = st.empty()
        audio_ocr_ph = st.empty()
        c1, c2, c3 = st.columns(3)
        with c1: m_det = st.empty()
        with c2: m_danger = st.empty()
        with c3: m_mode = st.empty()
        if show_logs: log_ph = st.expander("📋 Logs").empty()

        cam_photo = st.camera_input("📸 Ambil foto")
        if cam_photo is not None:
            if st.session_state.model1 is None and st.session_state.model2 is None:
                st.error("⚠️ Load minimal satu model YOLO!")
            else:
                from PIL import Image
                img_pil = Image.open(cam_photo)
                frame = np.array(img_pil)
                if len(frame.shape) == 3 and frame.shape[2] == 3:
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                else:
                    frame_bgr = frame
                frame_bgr = cv2.resize(frame_bgr, (640, 480))
                orig = frame_bgr.copy()
                m1, m2, m3 = st.session_state.model1, st.session_state.model2, st.session_state.model3
                frame_ann, dets = process_frame_detection_multi(frame_bgr, m1, m2, m3, conf_threshold)
                frame_ph.image(cv2.cvtColor(frame_ann, cv2.COLOR_BGR2RGB), caption=f"Terdeteksi: {len(dets)} objek", use_container_width=True)
                for d in dets:
                    st.session_state.detection_history.append(d)
                st.session_state.detection_history = st.session_state.detection_history[-500:]
                now_time = time.time()
                danger_list, rambu_list = handle_alerts(dets, now_time, enable_audio, alert_cooldown, warn_ph, status_ph, audio_danger_ph, audio_rambu_ph)
                m_det.metric("Deteksi", len(dets))
                m_danger.metric("⚠️ Bahaya", len(danger_list))
                m_mode.metric("Mode", "📸 Foto")
                if ocr_cam_on and st.session_state.ocr_engine is not None:
                    handle_ocr_dedup(orig, st.session_state.ocr_engine, ocr_min_conf, enable_tts, ocr_ph, audio_ocr_ph)
                if show_logs:
                    log_ph.markdown('<br>'.join([f'[{ts}] {msg}' for ts, msg in st.session_state.log[:10]]), unsafe_allow_html=True)

    else:
        uploaded = st.file_uploader("Upload video", type=['mp4','avi','mov','mkv'])
        frame_ph = st.empty()
        status_ph = st.empty()
        warn_ph = st.empty()
        ocr_ph = st.empty()
        audio_danger_ph = st.empty()
        audio_rambu_ph = st.empty()
        audio_ocr_ph = st.empty()
        c1, c2, c3 = st.columns(3)
        with c1: m_det = st.empty()
        with c2: m_danger = st.empty()
        with c3: m_fps = st.empty()
        if show_logs: log_ph = st.expander("📋 Logs").empty()

        if uploaded is not None:
            current_name = uploaded.name
            if st.session_state.last_uploaded_name != current_name:
                st.session_state.ocr_triggered = False
                st.session_state.ocr_frame_count = 0
                st.session_state.last_uploaded_name = current_name
                st.session_state.last_ocr_text = ''
                st.session_state.alerted_objects.clear()
                ocr_ph.empty()
                st.info("🔄 Video baru. Klik 'Baca Teks' untuk scan OCR.")

        col1, col2 = st.columns(2)
        with col1: btn_start = st.button("▶️ Start Detection", use_container_width=True)
        with col2: btn_baca = st.button("📖 Baca Teks", use_container_width=True)
        if btn_baca:
            st.session_state.ocr_triggered = True
            st.session_state.ocr_frame_count = 0
            st.session_state.last_ocr_text = ''
            st.info("🔍 OCR aktif sepanjang video.")

        if uploaded and btn_start:
            if st.session_state.model1 is None and st.session_state.model2 is None:
                st.error("⚠️ Load minimal satu model YOLO!")
                st.stop()
            st.session_state.alerted_objects.clear()
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
                tmp.write(uploaded.read())
                vid_path = tmp.name
            try:
                cap = cv2.VideoCapture(vid_path)
                total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps_vid = cap.get(cv2.CAP_PROP_FPS) or 30.0
                m1, m2, m3 = st.session_state.model1, st.session_state.model2, st.session_state.model3
                ocr = st.session_state.ocr_engine
                cnt, start = 0, time.time()
                prog = st.progress(0)
                while True:
                    ok, frame = cap.read()
                    if not ok: break
                    cnt += 1
                    if cnt % 2 != 0: continue
                    frame = cv2.resize(frame, (640, 480))
                    orig = frame.copy()
                    now_sec = cnt / fps_vid
                    prog.progress(min(cnt/max(total,1), 1.0), text=f"Frame {cnt}/{total} ({now_sec:.1f}s)")
                    frame_ann, dets = process_frame_detection_multi(frame, m1, m2, m3, conf_threshold)
                    frame_ph.image(cv2.cvtColor(frame_ann, cv2.COLOR_BGR2RGB), use_container_width=True)
                    st.session_state.last_frame = orig
                    for d in dets:
                        st.session_state.detection_history.append(d)
                    st.session_state.detection_history = st.session_state.detection_history[-500:]
                    now_time = time.time()
                    danger_list, rambu_list = handle_alerts(dets, now_time, enable_audio, alert_cooldown, warn_ph, status_ph, audio_danger_ph, audio_rambu_ph)
                    if st.session_state.ocr_triggered and ocr is not None:
                        st.session_state.ocr_frame_count += 1
                        if st.session_state.ocr_frame_count % ocr_scan_interval == 0:
                            handle_ocr_dedup(orig, ocr, ocr_min_conf, enable_tts, ocr_ph, audio_ocr_ph)
                    m_det.metric("Deteksi", len(dets))
                    m_danger.metric("⚠️ Bahaya", len(danger_list))
                    elapsed = time.time() - start
                    m_fps.metric("FPS", f"{cnt/elapsed:.1f}" if elapsed > 0 else "0")
                    if show_logs:
                        log_ph.markdown('<br>'.join([f'[{ts}] {msg}' for ts,msg in st.session_state.log[:10]]), unsafe_allow_html=True)
                    time.sleep(0.01)
                cap.release()
                prog.empty()
                st.success(f"✅ Selesai! {cnt} frame.")
                st.session_state.ocr_triggered = False
            finally:
                try: os.unlink(vid_path)
                except: pass

# ═════════════════════════════════════════════════════════════════════
# TAB 2: TEXT READING
# ═════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### 📖 Text Reading")
    mode2 = st.radio("Input:", ["📷 Dari Kamera", "📤 Upload Gambar"], horizontal=True)
    img_ph2, res_ph2, aud_ph2 = st.empty(), st.empty(), st.empty()
    if mode2 == "📷 Dari Kamera":
        cam_ocr = st.camera_input("📸 Ambil foto untuk baca teks")
        if cam_ocr is not None:
            if st.session_state.ocr_engine is None:
                st.error("⚠️ Load OCR dulu!")
            else:
                from PIL import Image
                img = Image.open(cam_ocr)
                arr = np.array(img)
                img_ph2.image(img, use_container_width=True)
                text = perform_ocr_on_frame(arr, st.session_state.ocr_engine, ocr_min_conf)
                res_ph2.markdown(f'<div class="ocr-result">📝 {text}</div>', unsafe_allow_html=True)
                if text and text != "Tidak ada teks terdeteksi" and len(text) > 5 and enable_tts:
                    audio = get_audio_bytes(f"Ada tulisan: {text}")
                    if audio:
                        aud_ph2.empty()
                        play_audio_safe(aud_ph2, audio)
                        st.success("🔊 Audio diputar")
    else:
        up_img = st.file_uploader("Upload gambar", type=['jpg','jpeg','png','bmp'])
        if up_img:
            from PIL import Image
            img = Image.open(up_img)
            arr = np.array(img)
            img_ph2.image(img, use_container_width=True)
            if st.session_state.ocr_engine is None:
                st.error("⚠️ Load OCR dulu!")
            else:
                text = perform_ocr_on_frame(arr, st.session_state.ocr_engine, ocr_min_conf)
                res_ph2.markdown(f'<div class="ocr-result">📝 {text}</div>', unsafe_allow_html=True)
                if text and text != "Tidak ada teks terdeteksi" and len(text) > 5 and enable_tts:
                    audio = get_audio_bytes(f"Ada tulisan: {text}")
                    if audio:
                        aud_ph2.empty()
                        play_audio_safe(aud_ph2, audio)
                        st.success("🔊 Audio diputar")

# ═════════════════════════════════════════════════════════════════════
# TAB 3: STATISTICS
# ═════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("### 📊 Statistics")
    if st.session_state.detection_history:
        hist = st.session_state.detection_history
        total = len(hist)
        danger = len([d for d in hist if d['risk_level'] == 'BAHAYA'])
        warn = len([d for d in hist if d['risk_level'] == 'WASPADA'])
        aman = len([d for d in hist if d['risk_level'] == 'AMAN'])
        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(f'<div class="stat-box"><div class="stat-value">{total}</div><div class="stat-label">Total</div></div>', unsafe_allow_html=True)
        c2.markdown(f'<div class="stat-box"><div class="stat-value" style="color:#ff4444;">{danger}</div><div class="stat-label">Bahaya</div></div>', unsafe_allow_html=True)
        c3.markdown(f'<div class="stat-box"><div class="stat-value" style="color:#faad14;">{warn}</div><div class="stat-label">Waspada</div></div>', unsafe_allow_html=True)
        c4.markdown(f'<div class="stat-box"><div class="stat-value" style="color:#00c073;">{aman}</div><div class="stat-label">Aman</div></div>', unsafe_allow_html=True)
        df = pd.DataFrame([{
            'Waktu': d['timestamp'].strftime("%H:%M:%S"),
            'Objek': get_indo_name(d['class']),
            'Confidence': f"{d['confidence']:.0%}",
            'Risiko': d['risk_level'],
            'Posisi': 'Kiri' if d['position_x'] < 0.3 else ('Kanan' if d['position_x'] > 0.7 else 'Depan'),
        } for d in hist[-100:]])
        st.dataframe(df, use_container_width=True)
        st.markdown("#### 📈 Ringkasan per Objek")
        class_counts = defaultdict(lambda: {'total':0,'bahaya':0,'waspada':0})
        for d in hist:
            name = get_indo_name(d['class'])
            class_counts[name]['total'] += 1
            if d['risk_level'] == 'BAHAYA': class_counts[name]['bahaya'] += 1
            elif d['risk_level'] == 'WASPADA': class_counts[name]['waspada'] += 1
        summary_df = pd.DataFrame([
            {'Objek':n,'Total':v['total'],'Bahaya':v['bahaya'],'Waspada':v['waspada']}
            for n,v in sorted(class_counts.items(), key=lambda x:x[1]['total'], reverse=True)
        ])
        st.dataframe(summary_df, use_container_width=True)
        if st.button("🗑️ Hapus Data"):
            st.session_state.detection_history = []
            st.session_state.alerted_objects.clear()
            st.rerun()
    else:
        st.info("📊 Belum ada data. Mulai deteksi di tab Detection.")

st.divider()
st.markdown('<div style="text-align:center; color:#999; font-size:0.8rem; padding:1rem 0;"><strong>Asisten Navigasi Tunanetra v4.4</strong> • YOLOv11 • EasyOCR • gTTS</div>', unsafe_allow_html=True)
