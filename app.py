# =====================================================================
# ASISTEN NAVIGASI TUNANETRA - STREAMLIT DASHBOARD v4.0 (FINAL)
# =====================================================================

import streamlit as st
import cv2, os, re, time, base64, tempfile, json
import numpy as np
import pandas as pd
from gtts import gTTS
from datetime import datetime
from pathlib import Path
from io import BytesIO
from collections import defaultdict
import logging
import random

# ─────────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────
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
.alert-info { background: #f0f6ff; border: 1px solid #90c8ff; border-left: 4px solid #3f8bff; border-radius: 8px; padding: 1rem; color: #0066cc; margin: 0.8rem 0; }
.alert-success { background: #f0fff8; border: 1px solid #80ecc0; border-left: 4px solid #00c073; border-radius: 8px; padding: 1rem; color: #00955a; margin: 0.8rem 0; }

.ocr-result {
    background: #f8fbff;
    border: 2px solid #3f8bff;
    border-radius: 10px;
    padding: 1.4rem;
    color: #0066cc;
    font-size: 1.05rem;
    min-height: 60px;
}

.stat-box {
    background: linear-gradient(135deg, #f5f7ff 0%, #faf8ff 100%);
    border-radius: 10px;
    padding: 1.2rem;
    text-align: center;
    border: 1px solid #e0e4f0;
}
.stat-value { font-size: 2rem; font-weight: 700; color: #6c3fff; }
.stat-label { font-size: 0.8rem; color: #999; text-transform: uppercase; letter-spacing: 1px; font-weight: 600; }

.stButton > button {
    border-radius: 10px;
    border: none;
    font-weight: 600;
    padding: 0.8rem 1.4rem;
    transition: all 0.2s ease;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
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
    'ocr_triggered': False, 'ocr_triggered_cam': False,
    'ocr_frame_count': 0,
    'last_uploaded_name': None,
    'last_ocr_text': '',          # Track teks OCR terakhir agar tidak duplikat
}
for key, default_value in session_defaults.items():
    if key not in st.session_state:
        st.session_state[key] = default_value

def add_log(msg):
    st.session_state.log.insert(0, (time.strftime("%H:%M:%S"), msg))
    st.session_state.log = st.session_state.log[:30]

# ============================================================
# FUNGSI AUDIO (ANTI ERROR / DUPLICATE ID)
# ============================================================
def play_audio_safe(placeholder, audio_bytes):
    """Memutar audio - SETIAP KALI BUAT ELEMEN BARU"""
    if audio_bytes:
        b64 = base64.b64encode(audio_bytes).decode()
        unique_id = f"audio_{int(time.time()*1000)}_{random.randint(1000,9999)}"
        html_code = f"""
            <audio autoplay="true" style="display:none;" id="{unique_id}">
                <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
            </audio>
            <script>
                setTimeout(function() {{
                    document.getElementById("{unique_id}").play();
                }}, 100);
            </script>
        """
        placeholder.markdown(html_code, unsafe_allow_html=True)

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
# FUNGSI TERJEMAHAN
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
    for k in kw:
        if k in n: return True
    return False

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
        'masjid':'Ada masjid',
        'gereja':'Ada gereja',
        'pom bensin':'Ada pom bensin',
        'tempat parkir':'Ada tempat parkir',
        'jalur sepeda':'Ada jalur sepeda',
        'batas kecepatan':'Ada batas kecepatan',
        'persimpangan':'Ada persimpangan di depan',
    }
    for k, v in suara.items():
        if k in n or n in k: return v
    return f"Ada {get_indo_name(n)}"

def generate_alert(name, pos_x, area):
    nama = get_indo_name(name)
    if pos_x < 0.3:
        pos, arah = "di kiri", "Geser ke kanan"
    elif pos_x > 0.7:
        pos, arah = "di kanan", "Geser ke kiri"
    else:
        pos, arah = "di depan", "Berhenti"
    
    if area > 0.15:
        return f"Awas! Ada {nama} sangat dekat {pos}. {arah} sekarang!"
    elif area > 0.08:
        return f"Awas! Ada {nama} {pos}. {arah}!"
    else:
        return f"Hati-hati, ada {nama} {pos}. {arah} pelan-pelan."

# ============================================================
# DETEKSI FRAME
# ============================================================
def process_frame_detection(frame, model, conf=0.4):
    if model is None: return frame, []
    try:
        results = model.predict(frame, conf=conf, verbose=False)
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
                    
                    if area_ratio < 0.015: continue
                    
                    pos_x = ((x1 + x2) / 2) / frame_w
                    cls_lower = cls_name.lower()
                    
                    obstacle_objects = ['pothole', 'lubang', 'stairs', 'tangga', 'obstacle', 'rintangan', 'road-barrier', 'pembatas jalan', 'pole', 'tiang']
                    vehicle_objects = ['car', 'mobil', 'bus', 'truck', 'truk', 'vehicle', 'kendaraan', 'motorcycle', 'motor', 'bicycle', 'sepeda', 'train', 'kereta']
                    person_objects = ['person', 'orang']
                    
                    is_obstacle = any(obs in cls_lower for obs in obstacle_objects)
                    is_vehicle = any(veh in cls_lower for veh in vehicle_objects)
                    is_person = any(per in cls_lower for per in person_objects)
                    
                    if is_obstacle:
                        if conf_score > 0.25:
                            if area_ratio > 0.08: risk_level = 'BAHAYA'
                            elif area_ratio > 0.03: risk_level = 'WASPADA'
                            else: risk_level = 'AMAN'
                        else: risk_level = 'AMAN'
                    elif is_vehicle:
                        if conf_score > 0.35 and area_ratio > 0.12: risk_level = 'BAHAYA'
                        elif conf_score > 0.30 and area_ratio > 0.06: risk_level = 'WASPADA'
                        else: risk_level = 'AMAN'
                    elif is_person:
                        if conf_score > 0.40 and area_ratio > 0.08: risk_level = 'WASPADA'
                        else: risk_level = 'AMAN'
                    else:
                        risk_level = 'AMAN'
                    
                    detections.append({
                        'class': cls_name, 'confidence': float(conf_score), 'area_ratio': area_ratio,
                        'position_x': pos_x, 'risk_level': risk_level, 'bbox': (x1, y1, x2, y2),
                        'timestamp': datetime.now()
                    })
                    
                    if risk_level == 'BAHAYA': color = (0, 0, 255)
                    elif risk_level == 'WASPADA': color = (0, 165, 255)
                    else: color = (0, 255, 0)
                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    label = f"{cls_name} {conf_score:.2f}"
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return frame, detections
    except Exception as e:
        logger.error(f"Detection error: {e}")
        return frame, []

def process_frame_detection_multi(frame, model1, model2, model3, conf=0.4):
    """Memproses frame secara berurutan pada M1, M2, dan M3"""
    frame_annotated = frame.copy()
    all_detections = []
    
    for mdl in [model1, model2, model3]:
        if mdl is not None:
            frame_annotated, dets = process_frame_detection(frame_annotated, mdl, conf)
            all_detections.extend(dets)
            
    return frame_annotated, all_detections

# ============================================================
# FUNGSI OCR
# ============================================================
def perform_ocr_on_frame(frame, ocr_engine, min_conf=0.20):
    if ocr_engine is None: return "OCR engine tidak tersedia"
    try:
        if len(frame.shape) == 3: gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else: gray = frame
        
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
                text = re.sub(r'[^a-zA-Z0-9\s\.,!?\-]', '', text)
                if len(text) > 1 and conf > min_conf: texts.append(text)
            if texts:
                result_text = ' '.join(texts)
                result_text = ' '.join(result_text.split())
                if len(result_text) > 2: return result_text
        return "Tidak ada teks terdeteksi"
    except Exception as e:
        return f"Error: {str(e)[:30]}"

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
    
    # ----------------------------------------------------
    # INFO PENTING UPLOAD FILE
    # ----------------------------------------------------
    st.info("💡 Info: Mengunggah file model (.pt) memakan waktu tergantung pada ukuran file dan kecepatan internet Anda (Bisa 1-5 menit).")
    
    with st.expander("🏔️ Model 2: Tangga/Lubang"):
        up2 = st.file_uploader("Upload best.pt (M2)", type=['pt'], key='m2up')
        if up2 and st.button("Load M2"):
            with st.spinner("Sedang memuat Model 2, mohon tunggu..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as tmp:
                    tmp.write(up2.read())
                from ultralytics import YOLO
                st.session_state.model2 = YOLO(tmp.name)
                st.success("✅ M2 Dimuat!")
            
    with st.expander("🚦 Model 3: Rambu"):
        up3 = st.file_uploader("Upload best_rambu.pt (M3)", type=['pt'], key='m3up')
        if up3 and st.button("Load M3"):
            with st.spinner("Sedang memuat Model 3, mohon tunggu..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as tmp:
                    tmp.write(up3.read())
                from ultralytics import YOLO
                st.session_state.model3 = YOLO(tmp.name)
                st.success("✅ M3 Dimuat!")
    
    st.markdown("---")
    s1 = "✅" if st.session_state.model1 else "⚠️"
    s2 = "✅" if st.session_state.model2 else "⚠️"
    s3 = "✅" if st.session_state.model3 else "⚠️"
    so = "✅" if st.session_state.ocr_engine else "⚠️"
    st.markdown(f'<div class="pills"><span class="pill pill-model">{s1} M1</span><span class="pill pill-model">{s2} M2</span><span class="pill pill-model">{s3} M3</span><span class="pill pill-model">{so} OCR</span></div>', unsafe_allow_html=True)
    
    conf_threshold = st.slider("Confidence", 0.1, 0.9, 0.4, 0.05)
    enable_audio = st.checkbox("🔊 Audio", value=True)
    alert_cooldown = st.slider("Cooldown (s)", 2, 10, 5)
    ocr_min_conf = st.slider("OCR Confidence", 0.1, 0.9, 0.2, 0.05)
    ocr_scan_interval = st.slider("OCR Scan Interval", 1, 10, 3, help="Scan OCR setiap N frame")
    enable_tts = st.checkbox("🔊 TTS", value=True)
    show_logs = st.checkbox("Show Logs", value=True)

# ─────────────────────────────────────────────────────────────────────
# MAIN TABS
# ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🎯 Detection", "📖 Text Reading", "📊 Statistics"])

# ═════════════════════════════════════════════════════════════════════
# TAB 1: DETECTION
# ═════════════════════════════════════════════════════════════════════
with tab1:
    mode = st.radio("Mode:", ["📹 Webcam", "📤 Upload Video"], horizontal=True)
    st.divider()
    
    frame_ph = st.empty()
    status_ph = st.empty()
    warn_ph = st.empty()
    ocr_ph = st.empty()
    
    # Placeholder Audio
    audio_danger_ph = st.empty()
    audio_rambu_ph = st.empty()
    audio_ocr_ph = st.empty()
    
    c1, c2, c3 = st.columns(3)
    with c1: m_det = st.empty()
    with c2: m_danger = st.empty()
    with c3: m_fps = st.empty()
    
    if show_logs: log_ph = st.expander("📋 Logs").empty()

    # ============================================================
    # WEBCAM
    # ============================================================
    if mode == "📹 Webcam":
        run = st.toggle("🎥 Aktifkan Webcam")
        btn_baca = st.button("📖 Baca Teks")
        
        if btn_baca:
            st.session_state.ocr_triggered_cam = True
            st.session_state.last_ocr_text = '' # Reset agar dibaca lagi jika ditekan manual
            st.info("🔍 OCR akan membaca teks dari frame webcam...")
        
        if run:
            if st.session_state.model1 is None:
                st.error("⚠️ Load YOLO dulu!")
            else:
                # GITHUB CAMERA FIX
                cap = None
                backends_to_try = [
                    cv2.CAP_V4L2, cv2.CAP_DSHOW, cv2.CAP_ANY, 0,
                ]
                for backend in backends_to_try:
                    try:
                        test_cap = cv2.VideoCapture(0, backend)
                        if test_cap.isOpened():
                            cap = test_cap
                            break
                        test_cap.release()
                    except:
                        continue
                if cap is None:
                    cap = cv2.VideoCapture(0)
                
                if not cap.isOpened():
                    st.error("❌ Webcam error! Pastikan kamera tersedia.")
                else:
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    m1, m2, m3 = st.session_state.model1, st.session_state.model2, st.session_state.model3
                    ocr = st.session_state.ocr_engine
                    cnt, start = 0, time.time()
                    
                    while run:
                        ok, frame = cap.read()
                        if not ok: 
                            cap.release()
                            cap = cv2.VideoCapture(0)
                            if not cap.isOpened():
                                st.error("❌ Kamera terputus!")
                                break
                            continue
                        cnt += 1
                        if cnt % 3 != 0: continue
                        
                        now = time.time()
                        orig = frame.copy()
                        
                        frame_ann, dets = process_frame_detection_multi(frame, m1, m2, m3, conf_threshold)
                        frame_ph.image(cv2.cvtColor(frame_ann, cv2.COLOR_BGR2RGB), use_container_width=True)
                        st.session_state.last_frame = orig
                        
                        # ============================================================
                        # OCR - BACA TEKS (HANYA JIKA TEKS BERBEDA)
                        # ============================================================
                        if st.session_state.ocr_triggered_cam and ocr is not None:
                            if cnt % ocr_scan_interval == 0:
                                with st.spinner("🔍 Reading..."):
                                    text = perform_ocr_on_frame(orig, ocr, ocr_min_conf)
                                if text and text != "Tidak ada teks terdeteksi" and len(text) > 5:
                                    
                                    # Normalisasi Teks untuk mencegah duplikat berulang
                                    norm_text = re.sub(r'[^a-zA-Z0-9]', '', text).lower()
                                    norm_last = re.sub(r'[^a-zA-Z0-9]', '', st.session_state.last_ocr_text).lower()
                                    
                                    if norm_text != norm_last:
                                        st.session_state.last_ocr_text = text
                                        ocr_ph.markdown(f'<div class="ocr-result">{text}</div>', unsafe_allow_html=True)
                                        if enable_tts:
                                            audio = get_audio_bytes(f"Ada tulisan: {text}")
                                            if audio:
                                                audio_ocr_ph.empty()
                                                play_audio_safe(audio_ocr_ph, audio)
                                                st.success(f"🔊 Suara diputar: {text[:30]}...")
                                    else:
                                        ocr_ph.markdown(f'<div class="ocr-result">📝 {text} <br><small>(Teks sama, tidak dibaca ulang)</small></div>', unsafe_allow_html=True)
                        
                        # ============================================================
                        # DETEKSI BAHAYA & RAMBU
                        # ============================================================
                        danger = [d for d in dets if d['risk_level'] == 'BAHAYA']
                        rambu = [d for d in dets if is_rambu(d['class'])]
                        
                        if danger and enable_audio:
                            d = danger[0]
                            key = d['class']
                            if (now - st.session_state.last_alert_time[key]) >= alert_cooldown:
                                msg = generate_alert(d['class'], d['position_x'], d['area_ratio'])
                                warn_ph.markdown(f'<div class="alert-danger">⚠️ {msg}</div>', unsafe_allow_html=True)
                                audio = get_audio_bytes(msg)
                                if audio:
                                    audio_danger_ph.empty()
                                    play_audio_safe(audio_danger_ph, audio)
                                st.session_state.last_alert_time[key] = now
                                status_ph.markdown('<div class="pills"><span class="pill pill-run">● YOLO</span><span class="pill pill-danger">● DANGER</span></div>', unsafe_allow_html=True)
                        elif rambu and enable_audio:
                            d = rambu[0]
                            key = f"rambu_{d['class']}"
                            if (now - st.session_state.last_alert_time[key]) >= alert_cooldown:
                                msg = generate_rambu_alert(d['class'])
                                warn_ph.markdown(f'<div class="alert-info">ℹ️ {msg}</div>', unsafe_allow_html=True)
                                audio = get_audio_bytes(msg)
                                if audio:
                                    audio_rambu_ph.empty()
                                    play_audio_safe(audio_rambu_ph, audio)
                                st.session_state.last_alert_time[key] = now
                                status_ph.markdown('<div class="pills"><span class="pill pill-run">● YOLO</span><span class="pill pill-ocr">● RAMBU</span></div>', unsafe_allow_html=True)
                        else:
                            warn_ph.markdown('<div class="alert-success">✅ Aman</div>', unsafe_allow_html=True)
                            status_ph.markdown('<div class="pills"><span class="pill pill-run">● YOLO</span><span class="pill pill-ok">● Safe</span></div>', unsafe_allow_html=True)
                        
                        m_det.metric("Detections", len(dets))
                        m_danger.metric("⚠️ Danger", len(danger))
                        elapsed = time.time() - start
                        m_fps.metric("FPS", f"{cnt/elapsed:.1f}" if elapsed>0 else "0")
                        if show_logs:
                            log_ph.markdown('<br>'.join([f'[{ts}] {msg}' for ts,msg in st.session_state.log[:10]]), unsafe_allow_html=True)
                        
                        if btn_baca and not st.session_state.ocr_triggered_cam:
                            st.session_state.ocr_triggered_cam = True
                        
                        time.sleep(0.01)
                    
                    cap.release()
                    # Reset OCR text saat webcam mati
                    st.session_state.last_ocr_text = ''
                    st.session_state.ocr_triggered_cam = False
                    st.success("Webcam dihentikan.")

    # ============================================================
    # UPLOAD VIDEO
    # ============================================================
    else:
        uploaded = st.file_uploader("Upload video", type=['mp4','avi','mov','mkv'])
        
        # ──────────────────────────────────────────────────────────
        # RESET OCR TRIGGER SAAT UPLOAD VIDEO BARU
        # ──────────────────────────────────────────────────────────
        if uploaded is not None:
            current_name = uploaded.name
            if st.session_state.last_uploaded_name != current_name:
                st.session_state.ocr_triggered = False
                st.session_state.ocr_frame_count = 0
                st.session_state.last_uploaded_name = current_name
                st.session_state.last_ocr_text = ''  # Reset teks terakhir
                ocr_ph.empty()
                st.info("🔄 Video baru diupload. OCR di-reset. Klik 'Baca Teks' lagi jika ingin scan.")
        
        col1, col2 = st.columns(2)
        with col1: btn_start = st.button("▶️ Start Detection", use_container_width=True)
        with col2: btn_baca = st.button("📖 Baca Teks", use_container_width=True)
        
        if btn_baca:
            st.session_state.ocr_triggered = True
            st.session_state.ocr_frame_count = 0
            st.session_state.last_ocr_text = '' # Reset text agar terbaca jika di-klik manual
            st.info("🔍 OCR akan membaca teks sepanjang video...")
        
        if uploaded and btn_start:
            if st.session_state.model1 is None:
                st.error("⚠️ Load YOLO dulu!")
                st.stop()
            
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
                tmp.write(uploaded.read())
                vid_path = tmp.name
            
            try:
                cap = cv2.VideoCapture(vid_path)
                total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                
                m1, m2, m3 = st.session_state.model1, st.session_state.model2, st.session_state.model3
                ocr = st.session_state.ocr_engine
                
                cnt, start = 0, time.time()
                prog = st.progress(0)
                
                while True:
                    ok, frame = cap.read()
                    if not ok: break
                    cnt += 1
                    if cnt % 3 != 0: continue
                    
                    now = cnt / fps
                    orig = frame.copy()
                    
                    prog.progress(min(cnt/max(total,1), 1.0), text=f"Frame {cnt}/{total} ({now:.1f}s)")
                    
                    frame_ann, dets = process_frame_detection_multi(frame, m1, m2, m3, conf_threshold)
                    frame_ph.image(cv2.cvtColor(frame_ann, cv2.COLOR_BGR2RGB), use_container_width=True)
                    st.session_state.last_frame = orig
                    
                    # ============================================================
                    # OCR - BACA TEKS (HANYA JIKA TEKS BERBEDA)
                    # ============================================================
                    if st.session_state.ocr_triggered and ocr is not None:
                        st.session_state.ocr_frame_count += 1
                        if st.session_state.ocr_frame_count % ocr_scan_interval == 0:
                            with st.spinner("🔍 Reading..."):
                                text = perform_ocr_on_frame(orig, ocr, ocr_min_conf)
                            if text and text != "Tidak ada teks terdeteksi" and len(text) > 5:
                                
                                # Normalisasi Teks untuk mencegah duplikat
                                norm_text = re.sub(r'[^a-zA-Z0-9]', '', text).lower()
                                norm_last = re.sub(r'[^a-zA-Z0-9]', '', st.session_state.last_ocr_text).lower()
                                
                                if norm_text != norm_last:
                                    st.session_state.last_ocr_text = text
                                    ocr_ph.markdown(f'<div class="ocr-result">{text}</div>', unsafe_allow_html=True)
                                    if enable_tts:
                                        audio = get_audio_bytes(f"Ada tulisan: {text}")
                                        if audio:
                                            audio_ocr_ph.empty()
                                            play_audio_safe(audio_ocr_ph, audio)
                                            st.success(f"🔊 Suara diputar: {text[:30]}...")
                                else:
                                    ocr_ph.markdown(f'<div class="ocr-result">📝 {text} <br><small>(Teks sama, tidak dibaca ulang)</small></div>', unsafe_allow_html=True)
                    
                    # ============================================================
                    # DETEKSI BAHAYA & RAMBU
                    # ============================================================
                    danger = [d for d in dets if d['risk_level'] == 'BAHAYA']
                    rambu = [d for d in dets if is_rambu(d['class'])]
                    
                    if danger and enable_audio:
                        d = danger[0]
                        key = d['class']
                        if (now - st.session_state.last_alert_time[key]) >= alert_cooldown:
                            msg = generate_alert(d['class'], d['position_x'], d['area_ratio'])
                            warn_ph.markdown(f'<div class="alert-danger">⚠️ {msg}</div>', unsafe_allow_html=True)
                            audio = get_audio_bytes(msg)
                            if audio:
                                audio_danger_ph.empty()
                                play_audio_safe(audio_danger_ph, audio)
                            st.session_state.last_alert_time[key] = now
                            status_ph.markdown('<div class="pills"><span class="pill pill-run">● YOLO</span><span class="pill pill-danger">● DANGER</span></div>', unsafe_allow_html=True)
                    elif rambu and enable_audio:
                        d = rambu[0]
                        key = f"rambu_{d['class']}"
                        if (now - st.session_state.last_alert_time[key]) >= alert_cooldown:
                            msg = generate_rambu_alert(d['class'])
                            warn_ph.markdown(f'<div class="alert-info">ℹ️ {msg}</div>', unsafe_allow_html=True)
                            audio = get_audio_bytes(msg)
                            if audio:
                                audio_rambu_ph.empty()
                                play_audio_safe(audio_rambu_ph, audio)
                            st.session_state.last_alert_time[key] = now
                            status_ph.markdown('<div class="pills"><span class="pill pill-run">● YOLO</span><span class="pill pill-ocr">● RAMBU</span></div>', unsafe_allow_html=True)
                    else:
                        warn_ph.markdown('<div class="alert-success">✅ Aman</div>', unsafe_allow_html=True)
                        status_ph.markdown('<div class="pills"><span class="pill pill-run">● YOLO</span><span class="pill pill-ok">● Safe</span></div>', unsafe_allow_html=True)
                    
                    m_det.metric("Detections", len(dets))
                    m_danger.metric("⚠️ Danger", len(danger))
                    elapsed = time.time() - start
                    m_fps.metric("FPS", f"{cnt/elapsed:.1f}" if elapsed>0 else "0")
                    if show_logs:
                        log_ph.markdown('<br>'.join([f'[{ts}] {msg}' for ts,msg in st.session_state.log[:10]]), unsafe_allow_html=True)
                    
                    time.sleep(0.01)
                
                cap.release()
                prog.empty()
                st.success("✅ Selesai!")
                st.metric("Total Frames", cnt)
                
                # Reset flag setelah selesai
                st.session_state.ocr_triggered = False
                
            finally:
                try: os.unlink(vid_path)
                except: pass

# ═════════════════════════════════════════════════════════════════════
# TAB 2: TEXT READING
# ═════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### 📖 Text Reading")
    mode2 = st.radio("Input:", ["📷 Capture", "📤 Upload"], horizontal=True)
    
    img_ph, res_ph, aud_ph = st.empty(), st.empty(), st.empty()
    
    if mode2 == "📷 Capture":
        if st.button("📸 Capture & Read"):
            if st.session_state.ocr_engine is None:
                st.error("⚠️ Load OCR dulu!")
            else:
                try:
                    cap = cv2.VideoCapture(0)
                    ret, frame = cap.read()
                    if ret:
                        frame = cv2.resize(frame, (640,480))
                        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        img_ph.image(rgb, use_container_width=True)
                        text = perform_ocr_on_frame(rgb, st.session_state.ocr_engine, ocr_min_conf)
                        res_ph.markdown(f'<div class="ocr-result">{text}</div>', unsafe_allow_html=True)
                        if text and text != "Tidak ada teks terdeteksi" and len(text) > 5 and enable_tts:
                            audio = get_audio_bytes(f"Ada tulisan: {text}")
                            if audio:
                                aud_ph.empty()
                                play_audio_safe(aud_ph, audio)
                                st.success("🔊 Audio playing...")
                    cap.release()
                except Exception as e:
                    st.error(f"Error: {e}")
    else:
        up_img = st.file_uploader("Upload image", type=['jpg','jpeg','png','bmp'])
        if up_img:
            from PIL import Image
            img = Image.open(up_img)
            arr = np.array(img)
            img_ph.image(img, use_container_width=True)
            text = perform_ocr_on_frame(arr, st.session_state.ocr_engine, ocr_min_conf)
            res_ph.markdown(f'<div class="ocr-result">{text}</div>', unsafe_allow_html=True)
            if text and text != "Tidak ada teks terdeteksi" and len(text) > 5 and enable_tts:
                audio = get_audio_bytes(f"Ada tulisan: {text}")
                if audio:
                    aud_ph.empty()
                    play_audio_safe(aud_ph, audio)
                    st.success("🔊 Audio playing...")

# ═════════════════════════════════════════════════════════════════════
# TAB 3: STATISTICS
# ═════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("### 📊 Statistics")
    if st.session_state.detection_history:
        total = len(st.session_state.detection_history)
        danger = len([d for d in st.session_state.detection_history if d['risk_level'] == 'BAHAYA'])
        warn = len([d for d in st.session_state.detection_history if d['risk_level'] == 'WASPADA'])
        
        c1, c2, c3 = st.columns(3)
        c1.markdown(f'<div class="stat-box"><div class="stat-value">{total}</div><div class="stat-label">Total</div></div>', unsafe_allow_html=True)
        c2.markdown(f'<div class="stat-box"><div class="stat-value" style="color:#ff4444;">{danger}</div><div class="stat-label">Danger</div></div>', unsafe_allow_html=True)
        c3.markdown(f'<div class="stat-box"><div class="stat-value" style="color:#ffaa00;">{warn}</div><div class="stat-label">Warning</div></div>', unsafe_allow_html=True)
        
        df = pd.DataFrame([{
            'Time': d['timestamp'].strftime("%H:%M:%S"),
            'Object': d['class'],
            'Confidence': f"{d['confidence']:.1%}",
            'Risk': d['risk_level'],
        } for d in st.session_state.detection_history[-100:]])
        st.dataframe(df, use_container_width=True)
        
        if st.button("🗑️ Clear"):
            st.session_state.detection_history = []
            st.rerun()
    else:
        st.info("📊 Belum ada data")

# ─────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────
st.divider()
st.markdown("""
<div style="text-align:center; color:#999; font-size:0.8rem; padding:1rem 0;">
    <strong>Asisten Navigasi Tunanetra v4.0 (FINAL)</strong> • YOLOv11 • EasyOCR • gTTS
</div>
""", unsafe_allow_html=True)
