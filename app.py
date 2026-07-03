# =====================================================================
# ASISTEN NAVIGASI TUNANETRA - STREAMLIT DASHBOARD v5.1 (FINAL FIX)
# =====================================================================
# PERBAIKAN:
# 1. Lubang/tangga: conf M2 = 0.10, area filter = 0.002
# 2. Orang: TIDAK PERNAH BAHAYA, hanya WASPADA
# 3. OCR: similarity threshold turun ke 0.5 agar lebih sensitif bedakan teks
# 4. Audio OCR: fix supaya suara keluar setiap teks berbeda
# =====================================================================

import streamlit as st
import cv2, os, re, time, base64, tempfile
import numpy as np
import pandas as pd
from gtts import gTTS
from datetime import datetime
from io import BytesIO
from collections import defaultdict
import logging
import random

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────
# PAGE CONFIG & CSS
# ─────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Asisten Tunanetra", page_icon="👁️", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
html, body, [data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #f5f7ff 0%, #faf8ff 100%) !important;
    font-family: 'Inter', sans-serif;
}
.header-logo { width:50px;height:50px;background:linear-gradient(135deg,#6c3fff,#3f8bff);border-radius:14px;display:flex;align-items:center;justify-content:center;font-size:1.4rem;box-shadow:0 4px 15px rgba(108,63,255,.3); color:white;}
.header-text h1 { font-size:1.8rem;color:#1a1a2e;margin:0; }
.header-text p { font-size:.85rem;color:#888;margin:0; }
.pills { display:flex;gap:.6rem;flex-wrap:wrap;margin-top:.8rem; }
.pill { padding:.35rem 1rem;border-radius:20px;font-size:.75rem;font-weight:600;border:1px solid transparent; }
.pill-run { background:#ede8ff;color:#6c3fff;border-color:#c4b0ff; }
.pill-ok { background:#e6fff5;color:#00955a;border-color:#80ecc0; }
.pill-danger { background:#fff0f0;color:#cc2222;border-color:#ffaaaa; }
.pill-ocr { background:#e8f4ff;color:#0066cc;border-color:#90c8ff; }
.pill-model { background:#f0e8ff;color:#6c3fff;border-color:#c4b0ff; }
.alert-danger { background:#fff0f0;border:1px solid #ffaaaa;border-left:4px solid #ff4444;border-radius:8px;padding:1rem;color:#cc2222;font-weight:600;margin:.8rem 0; }
.alert-warning { background:#fffbe6;border:1px solid #ffe58f;border-left:4px solid #faad14;border-radius:8px;padding:1rem;color:#ad6800;font-weight:600;margin:.8rem 0; }
.alert-info { background:#f0f6ff;border:1px solid #90c8ff;border-left:4px solid #3f8bff;border-radius:8px;padding:1rem;color:#0066cc;margin:.8rem 0; }
.alert-success { background:#f0fff8;border:1px solid #80ecc0;border-left:4px solid #00c073;border-radius:8px;padding:1rem;color:#00955a;margin:.8rem 0; }
.ocr-result { background:#f8fbff;border:2px solid #3f8bff;border-radius:10px;padding:1.4rem;color:#0066cc;font-size:1.05rem;min-height:60px; }
.stat-box { background:linear-gradient(135deg,#f5f7ff,#faf8ff);border-radius:10px;padding:1.2rem;text-align:center;border:1px solid #e0e4f0; }
.stat-value { font-size:2rem;font-weight:700;color:#6c3fff; }
.stat-label { font-size:.8rem;color:#999;text-transform:uppercase;letter-spacing:1px;font-weight:600; }
.stButton > button { border-radius:10px;border:none;font-weight:600;padding:.8rem 1.4rem;transition:all .2s;text-transform:uppercase;letter-spacing:.5px; }
.stButton > button:hover { transform:translateY(-2px);box-shadow:0 4px 12px rgba(108,63,255,.3); }
#MainMenu, footer { visibility:hidden;display:none; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────────────
session_defaults = {
    'model1': None, 'model2': None, 'model3': None, 'ocr_engine': None,
    'last_frame': None, 'log': [], 'detection_history': [],
    'last_alert_time': defaultdict(lambda: -99.0),
    'danger_announced': set(),
    'rambu_announced': set(),
    'ocr_triggered_cam': False,
    'ocr_triggered_vid': False,
    'ocr_frame_count': 0,
    'last_uploaded_name': None,
    'last_ocr_text': '',
    'ocr_silence_count': 0,
}
for key, value in session_defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

def add_log(msg):
    st.session_state.log.insert(0, (time.strftime("%H:%M:%S"), msg))
    st.session_state.log = st.session_state.log[:30]

# ─────────────────────────────────────────────────────────────────────
# FUNGSI AUDIO STABIL
# ─────────────────────────────────────────────────────────────────────
def play_audio_safe(placeholder, audio_bytes):
    if audio_bytes:
        b64 = base64.b64encode(audio_bytes).decode()
        uid = f"a_{int(time.time()*1000)}_{random.randint(1000,9999)}"
        html_code = f"""
            <audio autoplay="true" style="display:none;" id="{uid}">
                <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
            </audio>
            <script>
                setTimeout(function() {{
                    var el = document.getElementById("{uid}");
                    if(el) el.play();
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

# ─────────────────────────────────────────────────────────────────────
# TERJEMAHAN & LOGIKA PERINTAH ARAH
# ─────────────────────────────────────────────────────────────────────
INDO_NAMES = {
    'person':'orang', 'car':'mobil', 'bus':'bus', 'truck':'truk',
    'motorcycle':'motor', 'bicycle':'sepeda', 'dog':'anjing', 'cat':'kucing',
    'pothole':'lubang', 'stairs':'tangga', 'obstacle':'rintangan',
    'road-barrier':'pembatas jalan', 'pole':'tiang', 'train':'kereta',
    'stop sign':'rambu stop', 'traffic light':'lampu lalu lintas',
    'sidewalk':'trotoar', 'crosswalk':'zebra cross', 'tree':'pohon',
    'animal':'hewan', 'vehicle':'kendaraan', 'hole':'lubang', 'step':'tangga',
    'dilarang masuk':'dilarang masuk', 'dilarang parkir':'dilarang parkir',
    'dilarang berhenti':'dilarang berhenti', 'hati-hati':'hati-hati',
    'rumah sakit':'rumah sakit', 'masjid':'masjid', 'gereja':'gereja',
    'pom bensin':'pom bensin', 'tempat parkir':'tempat parkir',
    'jalur sepeda':'jalur sepeda', 'batas kecepatan':'batas kecepatan',
    'persimpangan':'persimpangan', 'ikuti arah bundaran':'ikuti arah bundaran',
}

def get_indo_name(name):
    n = name.lower()
    if n in INDO_NAMES: return INDO_NAMES[n]
    for k, v in INDO_NAMES.items():
        if k in n or n in k: return v
    return n

RAMBU_KW = {'rambu', 'lampu lalu lintas', 'traffic light', 'stop sign', 'dilarang',
            'hati-hati', 'rumah sakit', 'masjid', 'gereja', 'pom bensin', 'parkir',
            'persimpangan', 'bundaran', 'jalur', 'kecepatan', 'berhenti', 'masuk'}

def is_rambu(name):
    return any(k in name.lower() for k in RAMBU_KW)

def generate_rambu_alert(name):
    n = name.lower()
    m = {
        'stop sign':'Ada rambu stop. Berhenti!', 'rambu stop':'Ada rambu stop. Berhenti!',
        'dilarang masuk':'Ada rambu dilarang masuk. Jangan masuk!',
        'dilarang parkir':'Ada rambu dilarang parkir', 'dilarang berhenti':'Ada rambu dilarang berhenti',
        'hati-hati':'Ada rambu hati-hati. Waspada!', 'lampu lalu lintas':'Ada lampu lalu lintas',
        'rumah sakit':'Ada rumah sakit', 'masjid':'Ada masjid', 'gereja':'Ada gereja',
        'pom bensin':'Ada pom bensin', 'tempat parkir':'Ada tempat parkir',
        'jalur sepeda':'Ada jalur sepeda', 'batas kecepatan':'Ada batas kecepatan',
        'persimpangan':'Ada persimpangan di depan'
    }
    for k, v in m.items():
        if k in n: return v
    return f"Ada {get_indo_name(n)}"

def generate_alert(name, pos_x, area):
    nama = get_indo_name(name)
    if pos_x < 0.35: pos, arah = "di kiri", "Geser ke kanan"
    elif pos_x > 0.65: pos, arah = "di kanan", "Geser ke kiri"
    else: pos, arah = "di depan", "Berhenti"
    
    if area > 0.15: return f"Awas! {nama} sangat dekat {pos}. {arah} sekarang!"
    elif area > 0.05: return f"Awas! Ada {nama} {pos}. {arah}!"
    else: return f"Hati-hati, ada {nama} {pos}. {arah} pelan-pelan."

# ============================================================
# OCR - AUTO CORRECT + FILTER KARAKTER ANEH
# ============================================================
COMMON_WORDS = {
    'jln':'jalan','jl':'jalan','dlarang':'dilarang','dlrang':'dilarang',
    'dilarag':'dilarang','dilarng':'dilarang','dilrang':'dilarang',
    'parkr':'parkir','prkir':'parkir','pakir':'parkir','msuk':'masuk',
    'mausk':'masuk','brhenti':'berhenti','brheti':'berhenti','berheti':'berhenti',
    'hti-hti':'hati-hati','hati-hti':'hati-hati','hti-hati':'hati-hati',
    'rmah':'rumah','rumh':'rumah','sakt':'sakit','msjid':'masjid',
    'masjd':'masjid','grja':'gereja','greja':'gereja','skolah':'sekolah',
    'sekolh':'sekolah','bhaya':'bahaya','kluar':'keluar','kluaar':'keluar',
    'blok':'belok','belk':'belok','knan':'kanan','kri':'kiri','dpan':'depan',
    'belkang':'belakang','stp':'stop','stopp':'stop','lmpu':'lampu',
    'lampo':'lampu','mrah':'merah','hjau':'hijau','kunng':'kuning',
    'pnyebrangan':'penyeberangan','penyebrangn':'penyeberangan','zebr':'zebra',
    'crss':'cross','mtr':'meter','kec':'kecepatan','maks':'maksimal',
    'min':'minimal','spb':'SPBU','phtobox':'photobox','z0na':'zona','zon':'zona',
    'inspirasi':'inspirasi','foto':'foto','box':'box',
}

def autocorrect_word(word):
    if len(word) <= 1: return word
    low = word.lower()
    if low in COMMON_WORDS: return COMMON_WORDS[low]
    
    best_match, best_sim = None, 0
    for k, v in COMMON_WORDS.items():
        if abs(len(k) - len(low)) > 2: continue
        common = sum(1 for c in low if c in k)
        sim = common / max(len(k), len(low))
        if sim > 0.75 and sim > best_sim:
            best_sim, best_match = sim, v
            
    return best_match if best_match else word

def fix_spelled_text(text):
    if not text: return text
    words = text.split()
    result_words = []
    i = 0
    while i < len(words):
        if len(words[i]) == 1 and words[i].isalpha():
            single_chars = [words[i]]
            j = i + 1
            while j < len(words) and len(words[j]) == 1 and words[j].isalpha():
                single_chars.append(words[j])
                j += 1
            if len(single_chars) >= 3: result_words.append(''.join(single_chars))
            else: result_words.extend(single_chars)
            i = j
        else:
            result_words.append(words[i])
            i += 1
    return ' '.join(result_words)

def clean_ocr_text(raw_text):
    if not raw_text: return "Tidak ada teks terdeteksi"
    text = fix_spelled_text(raw_text)
    text = re.sub(r'[^a-zA-Z0-9\s\.,!?\-:/()%]', '', text)
    
    corrected = []
    for w in text.split():
        if len(w) <= 1 and not w.isdigit(): continue
        corrected.append(autocorrect_word(w))
        
    text = ' '.join(' '.join(corrected).split())
    return text.title() if len(text) >= 3 else "Tidak ada teks terdeteksi"

def perform_ocr_on_frame(frame, ocr_engine, min_conf=0.30):
    if ocr_engine is None: return "OCR engine tidak tersedia"
    try:
        if len(frame.shape) == 3: gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else: gray = frame

        h, w = gray.shape
        if w < 800:
            scale = 800 / w
            gray = cv2.resize(gray, (800, int(h * scale)), interpolation=cv2.INTER_CUBIC)

        denoised = cv2.fastNlMeansDenoising(gray, None, h=10, templateWindowSize=7, searchWindowSize=21)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)

        results = None
        try:
            results = ocr_engine.readtext(enhanced, detail=1, paragraph=False, text_threshold=0.6)
        except: pass
        
        if not results:
            try: results = ocr_engine.readtext(gray, detail=1, paragraph=False)
            except: pass

        if results:
            texts = []
            for (bbox, text, conf) in results:
                text = text.strip()
                if len(text) >= 2 and conf >= min_conf:
                    texts.append(text)
            if texts:
                raw_result = ' '.join(' '.join(texts).split())
                return clean_ocr_text(raw_result)
                
        return "Tidak ada teks terdeteksi"
    except Exception as e: 
        return f"Error: {str(e)[:30]}"

def texts_are_similar(t1, t2, threshold=0.5):
    """Fuzzy match - threshold 0.5 agar lebih sensitif bedakan teks"""
    if not t1 or not t2: return False
    a, b = ' '.join(t1.lower().split()), ' '.join(t2.lower().split())
    if a == b: return True
    if len(a) < 2 or len(b) < 2: return a == b
    # Bigram overlap
    s1 = set(a[i:i+2] for i in range(len(a)-1))
    s2 = set(b[i:i+2] for i in range(len(b)-1))
    if not s1 or not s2: return a == b
    return len(s1 & s2) / len(s1 | s2) >= threshold

# ============================================================
# DETEKSI YOLO - LUBANG & TANGGA PRIORITAS
# ============================================================
OBSTACLE_KW = ['pothole', 'lubang', 'hole', 'stairs', 'stair', 'step', 'tangga', 'obstacle', 'rintangan', 'road-barrier', 'barrier', 'pembatas', 'pole', 'tiang']
VEHICLE_KW = ['car', 'mobil', 'bus', 'truck', 'truk', 'vehicle', 'kendaraan', 'motorcycle', 'motor', 'bicycle', 'sepeda', 'train', 'kereta']
PERSON_KW = ['person', 'orang']

def process_frame_detection(frame, model, conf=0.4, is_m2=False):
    if model is None: return frame, []
    try:
        # PERBAIKAN 1: M2 conf sangat rendah agar lubang/tangga kedetek
        effective_conf = 0.10 if is_m2 else conf
        results = model.predict(frame, conf=effective_conf, iou=0.45, verbose=False)
        detections = []
        
        if results and len(results) > 0:
            result = results[0]
            if result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes.xyxy.cpu().numpy()
                confidences = result.boxes.conf.cpu().numpy()
                classes = result.boxes.cls.cpu().numpy().astype(int)
                fh, fw = frame.shape[:2]

                for box, conf_score, cls_idx in zip(boxes, confidences, classes):
                    x1, y1, x2, y2 = map(int, box)
                    cls_name = result.names[cls_idx]
                    cls_lower = cls_name.lower()

                    is_obstacle = any(k in cls_lower for k in OBSTACLE_KW)
                    is_vehicle = any(k in cls_lower for k in VEHICLE_KW)
                    is_person = any(k in cls_lower for k in PERSON_KW)

                    area = (x2 - x1) * (y2 - y1)
                    area_ratio = area / (fw * fh)

                    # PERBAIKAN 2: Lubang kecil tetap masuk (area 0.002)
                    effective_min_area = 0.002 if is_obstacle else 0.015
                    if area_ratio < effective_min_area: continue

                    pos_x = ((x1 + x2) / 2) / fw

                    # PERBAIKAN 3: ORANG TIDAK PERNAH BAHAYA
                    if is_obstacle:
                        if area_ratio > 0.04: risk_level = 'BAHAYA'
                        elif area_ratio > 0.01: risk_level = 'WASPADA'
                        else: risk_level = 'AMAN'
                    elif is_vehicle:
                        if conf_score > 0.45 and area_ratio > 0.15: risk_level = 'BAHAYA'
                        elif conf_score > 0.35 and area_ratio > 0.05: risk_level = 'WASPADA'
                        else: risk_level = 'AMAN'
                    elif is_person:
                        # ORANG TIDAK PERNAH BAHAYA, HANYA WASPADA
                        if conf_score > 0.40 and area_ratio > 0.10: risk_level = 'WASPADA'
                        else: risk_level = 'AMAN'
                    else:
                        risk_level = 'AMAN'

                    detections.append({
                        'class': cls_name, 'confidence': float(conf_score),
                        'area_ratio': area_ratio, 'position_x': pos_x,
                        'risk_level': risk_level, 'bbox': (x1, y1, x2, y2),
                        'timestamp': datetime.now()
                    })

                    color = (0, 0, 255) if risk_level == 'BAHAYA' else (0, 165, 255) if risk_level == 'WASPADA' else (0, 255, 0)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    label = f"{get_indo_name(cls_name)} {conf_score:.2f}"
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return frame, detections
    except Exception as e:
        logger.error(f"Detection error: {e}")
        return frame, []

def process_frame_detection_multi(frame, m1, m2, m3, conf=0.4):
    out = frame.copy()
    all_d = []
    # PRIORITAS: M2 dulu (lubang/tangga)
    if m2:
        out, d = process_frame_detection(out, m2, conf, is_m2=True)
        all_d.extend(d)
    if m1:
        out, d = process_frame_detection(out, m1, conf, is_m2=False)
        all_d.extend(d)
    if m3:
        out, d = process_frame_detection(out, m3, conf, is_m2=False)
        all_d.extend(d)
    return out, all_d

# ─────────────────────────────────────────────────────────────────────
# PENANGANAN ALERT SEKALI & DEDUP OCR
# ─────────────────────────────────────────────────────────────────────
REAPPEAR_WINDOW = 15.0  

def handle_alerts_once(dets, now_time, enable_audio, warn_ph, status_ph, audio_ph):
    danger = [d for d in dets if d['risk_level'] == 'BAHAYA']
    waspada = [d for d in dets if d['risk_level'] == 'WASPADA']
    rambu = [d for d in dets if is_rambu(d['class'])]

    if danger:
        d = danger[0]
        key = d['class']
        msg = generate_alert(d['class'], d['position_x'], d['area_ratio'])
        warn_ph.markdown(f'<div class="alert-danger">⚠️ {msg}</div>', unsafe_allow_html=True)
        
        if key not in st.session_state.danger_announced:
            st.session_state.danger_announced.add(key)
            add_log(f"BAHAYA: {msg}")
            if enable_audio:
                audio = get_audio_bytes(msg)
                if audio: play_audio_safe(audio_ph, audio)
                
        status_ph.markdown('<div class="pills"><span class="pill pill-run">● YOLO</span><span class="pill pill-danger">● BAHAYA</span></div>', unsafe_allow_html=True)

    elif waspada:
        d = waspada[0]
        key = f"w_{d['class']}"
        msg = generate_alert(d['class'], d['position_x'], d['area_ratio'])
        warn_ph.markdown(f'<div class="alert-warning">⚡ {msg}</div>', unsafe_allow_html=True)
        
        if key not in st.session_state.danger_announced:
            st.session_state.danger_announced.add(key)
            add_log(f"WASPADA: {msg}")
            if enable_audio:
                audio = get_audio_bytes(msg)
                if audio: play_audio_safe(audio_ph, audio)
                
        status_ph.markdown('<div class="pills"><span class="pill pill-run">● YOLO</span><span class="pill pill-warning" style="background:#fff5e0;color:#cc8800;border-color:#ffcc66;">● WASPADA</span></div>', unsafe_allow_html=True)

    elif rambu:
        d = rambu[0]
        key = d['class']
        msg = generate_rambu_alert(d['class'])
        warn_ph.markdown(f'<div class="alert-info">ℹ️ {msg}</div>', unsafe_allow_html=True)
        
        if key not in st.session_state.rambu_announced:
            st.session_state.rambu_announced.add(key)
            add_log(f"RAMBU: {msg}")
            if enable_audio:
                audio = get_audio_bytes(msg)
                if audio: play_audio_safe(audio_ph, audio)
                
        status_ph.markdown('<div class="pills"><span class="pill pill-run">● YOLO</span><span class="pill pill-ocr">● RAMBU</span></div>', unsafe_allow_html=True)

    else:
        warn_ph.markdown('<div class="alert-success">✅ Area Terpantau Aman</div>', unsafe_allow_html=True)
        status_ph.markdown('<div class="pills"><span class="pill pill-run">● YOLO</span><span class="pill pill-ok">● Aman</span></div>', unsafe_allow_html=True)

    return danger, rambu

def handle_ocr_dedup(frame, ocr_eng, min_conf, en_tts, ocr_ph, aocr_ph):
    text = perform_ocr_on_frame(frame, ocr_eng, min_conf)
    if text and text != "Tidak ada teks terdeteksi" and len(text) > 2:
        # PERBAIKAN 4: threshold 0.5 agar lebih sensitif bedakan teks
        if not texts_are_similar(text, st.session_state.last_ocr_text, threshold=0.5):
            st.session_state.last_ocr_text = text
            ocr_ph.markdown(f'<div class="ocr-result">📝 {text}</div>', unsafe_allow_html=True)
            add_log(f"OCR: {text[:40]}")
            if en_tts:
                a = get_audio_bytes(f"Ada tulisan: {text}")
                if a: play_audio_safe(aocr_ph, a)
            return True
        else:
            ocr_ph.markdown(f'<div class="ocr-result" style="opacity:0.6;">📝 {text}<br><small>(Teks sama, tidak dibaca ulang)</small></div>', unsafe_allow_html=True)
    return False

# ─────────────────────────────────────────────────────────────────────
# LOAD MODELS
# ─────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_yolo_models(path1='yolo11s.pt'):
    try:
        from ultralytics import YOLO
        try: return YOLO(path1)
        except: return None
    except: return None

@st.cache_resource(show_spinner=False)
def load_ocr():
    try:
        import easyocr
        return easyocr.Reader(['id', 'en'], gpu=False)
    except: return None

# ─────────────────────────────────────────────────────────────────────
# UI LAYOUT
# ─────────────────────────────────────────────────────────────────────
c1, c2 = st.columns([0.1, 0.9])
with c1: st.markdown('<div class="header-logo">👁️</div>', unsafe_allow_html=True)
with c2: st.markdown('<div class="header-text"><h1>Asisten Navigasi Tunanetra</h1><p>Deteksi Objek & Teks Terpadu v5.1</p></div>', unsafe_allow_html=True)
st.divider()

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("### ⚙️ Pengaturan")
    if st.button("📥 Load YOLO (M1)", use_container_width=True):
        with st.spinner("Loading YOLO..."): 
            st.session_state.model1 = load_yolo_models()
            st.success("✅ YOLO Dimuat!")
            
    if st.button("📥 Load OCR", use_container_width=True):
        with st.spinner("Loading OCR..."): 
            st.session_state.ocr_engine = load_ocr()
            st.success("✅ OCR Dimuat!")
            
    st.markdown("---")
    with st.expander("🏔️ Model 2: Tangga/Lubang"):
        up2 = st.file_uploader("Upload best.pt", type=['pt'], key='m2up')
        if up2 and st.button("Load M2"):
            with st.spinner("Memuat..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as tmp: tmp.write(up2.read())
                from ultralytics import YOLO
                st.session_state.model2 = YOLO(tmp.name)
                st.success("✅ M2 Dimuat!")
                
    with st.expander("🚦 Model 3: Rambu"):
        up3 = st.file_uploader("Upload best_rambu.pt", type=['pt'], key='m3up')
        if up3 and st.button("Load M3"):
            with st.spinner("Memuat..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as tmp: tmp.write(up3.read())
                from ultralytics import YOLO
                st.session_state.model3 = YOLO(tmp.name)
                st.success("✅ M3 Dimuat!")

    st.markdown("---")
    s1 = "✅" if st.session_state.model1 else "⚠️"
    s2 = "✅" if st.session_state.model2 else "⚠️"
    s3 = "✅" if st.session_state.model3 else "⚠️"
    so = "✅" if st.session_state.ocr_engine else "⚠️"
    st.markdown(f'<div class="pills"><span class="pill pill-model">{s1} M1</span><span class="pill pill-model">{s2} M2</span><span class="pill pill-model">{s3} M3</span><span class="pill pill-model">{so} OCR</span></div>', unsafe_allow_html=True)
    
    conf_threshold = st.slider("Confidence Deteksi", 0.1, 0.9, 0.30, 0.05, help="M2 (lubang) otomatis conf=0.10")
    enable_audio = st.checkbox("🔊 Audio Alert", value=True)
    ocr_min_conf = st.slider("OCR Confidence", 0.1, 0.9, 0.25, 0.05)
    ocr_scan_interval = st.slider("OCR Scan Tiap N Frame", 1, 30, 10)
    frame_skip = st.slider("Frame Skip (Biar Ringan)", 1, 10, 2)
    enable_tts = st.checkbox("🔊 TTS Baca Teks", value=True)
    show_logs = st.checkbox("📋 Show Logs", value=True)
    
    if st.button("🔁 Reset Daftar Suara"):
        st.session_state.danger_announced = set()
        st.session_state.rambu_announced = set()
        st.session_state.last_ocr_text = ''
        st.success("Daftar memori suara di-reset!")

# ─────────────────────────────────────────────────────────────────────
# TABS UTAMA
# ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🎯 Detection", "📖 Text Reading", "📊 Statistics"])

with tab1:
    mode = st.radio("Mode:", ["📹 Webcam LOKAL (Lancar)", "📤 Upload Video"], horizontal=True)
    st.divider()

    frame_ph = st.empty()
    status_ph = st.empty()
    warn_ph = st.empty()
    ocr_ph = st.empty()
    audio_alert_ph = st.empty()
    audio_ocr_ph = st.empty()

    c1, c2, c3 = st.columns(3)
    with c1: m_det = st.empty()
    with c2: m_danger = st.empty()
    with c3: m_fps = st.empty()

    if show_logs: log_ph = st.expander("📋 Logs").empty()

    # ============================================================
    # MODE WEBCAM LOKAL
    # ============================================================
    if mode == "📹 Webcam LOKAL (Lancar)":
        st.info("💡 Mode ini menggunakan kamera bawaan laptop secara langsung.")
        
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1: run_cam = st.toggle("🔴 AKTIFKAN KAMERA LIVE")
        with col_btn2: btn_ocr_cam = st.button("📖 Baca Teks")
        
        if btn_ocr_cam:
            st.session_state.ocr_triggered_cam = True
            st.info("🔍 OCR diaktifkan pada webcam.")

        if run_cam:
            if st.session_state.model1 is None and st.session_state.model2 is None:
                st.error("⚠️ Load YOLO dulu di pengaturan!")
                st.stop()
                
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                st.error("❌ Kamera tidak terdeteksi.")
                st.stop()
                
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            cnt, start = 0, time.time()
            m1, m2, m3 = st.session_state.model1, st.session_state.model2, st.session_state.model3
            ocr_eng = st.session_state.ocr_engine

            while run_cam:
                ret, frame = cap.read()
                if not ret: break
                cnt += 1
                
                if cnt % frame_skip != 0: continue
                orig = frame.copy()
                now_t = time.time()
                
                frame_ann, dets = process_frame_detection_multi(frame, m1, m2, m3, conf_threshold)
                frame_ph.image(cv2.cvtColor(frame_ann, cv2.COLOR_BGR2RGB), use_container_width=True)
                
                for d in dets: st.session_state.detection_history.append(d)
                st.session_state.detection_history = st.session_state.detection_history[-500:]

                danger_list, rambu_list = handle_alerts_once(dets, now_t, enable_audio, warn_ph, status_ph, audio_alert_ph)
                
                if st.session_state.ocr_triggered_cam and ocr_eng is not None:
                    if cnt % ocr_scan_interval == 0:
                        handle_ocr_dedup(orig, ocr_eng, ocr_min_conf, enable_tts, ocr_ph, audio_ocr_ph)

                m_det.metric("Deteksi", len(dets))
                m_danger.metric("⚠️ Bahaya", len(danger_list))
                elapsed = now_t - start
                m_fps.metric("FPS", f"{cnt/elapsed:.1f}" if elapsed > 0 else "0")
                
                if show_logs:
                    log_ph.markdown('<br>'.join([f'[{ts}] {msg}' for ts, msg in st.session_state.log[:10]]), unsafe_allow_html=True)
            
            cap.release()
            st.session_state.ocr_triggered_cam = False

    # ============================================================
    # MODE UPLOAD VIDEO
    # ============================================================
    else:
        uploaded = st.file_uploader("Upload video", type=['mp4','avi','mov','mkv'])
        
        if uploaded is not None:
            current_name = uploaded.name
            if st.session_state.last_uploaded_name != current_name:
                st.session_state.ocr_triggered_vid = False
                st.session_state.ocr_frame_count = 0
                st.session_state.last_uploaded_name = current_name
                st.session_state.last_ocr_text = ''
                st.session_state.danger_announced = set()
                st.session_state.rambu_announced = set()
                ocr_ph.empty()
                st.info("🔄 Video baru diupload.")

        col1, col2 = st.columns(2)
        with col1: btn_start = st.button("▶️ Start Detection", use_container_width=True)
        with col2: btn_baca = st.button("📖 Baca Teks", use_container_width=True)

        if btn_baca:
            st.session_state.ocr_triggered_vid = True
            st.session_state.ocr_frame_count = 0
            st.info("🔍 OCR aktif — teks akan dibaca.")

        if uploaded and btn_start:
            if st.session_state.model1 is None and st.session_state.model2 is None:
                st.error("⚠️ Load YOLO dulu di sidebar!")
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
                    
                    if cnt % frame_skip != 0: continue
                    frame = cv2.resize(frame, (640, 480))
                    orig = frame.copy()

                    prog.progress(min(cnt / max(total, 1), 1.0), text=f"Frame {cnt}/{total} ({cnt/fps:.1f}s)")

                    frame_ann, dets = process_frame_detection_multi(frame, m1, m2, m3, conf_threshold)
                    frame_ph.image(cv2.cvtColor(frame_ann, cv2.COLOR_BGR2RGB), use_container_width=True)
                    st.session_state.last_frame = orig

                    for d in dets: st.session_state.detection_history.append(d)
                    st.session_state.detection_history = st.session_state.detection_history[-500:]

                    now_time = time.time()
                    danger_list, rambu_list = handle_alerts_once(dets, now_time, enable_audio, warn_ph, status_ph, audio_alert_ph)

                    if st.session_state.ocr_triggered_vid and ocr is not None:
                        st.session_state.ocr_frame_count += 1
                        if st.session_state.ocr_frame_count % ocr_scan_interval == 0:
                            handle_ocr_dedup(orig, ocr, ocr_min_conf, enable_tts, ocr_ph, audio_ocr_ph)

                    m_det.metric("Deteksi", len(dets))
                    m_danger.metric("⚠️ Bahaya", len(danger_list))
                    elapsed = time.time() - start
                    m_fps.metric("FPS", f"{cnt/elapsed:.1f}" if elapsed > 0 else "0")

                    if show_logs:
                        log_ph.markdown('<br>'.join([f'[{ts}] {msg}' for ts, msg in st.session_state.log[:10]]), unsafe_allow_html=True)
                    time.sleep(0.001)

                cap.release()
                prog.empty()
                st.success(f"✅ Selesai! Total: {cnt} frame")
                st.session_state.ocr_triggered_vid = False

            finally:
                try: os.unlink(vid_path)
                except: pass

# ═════════════════════════════════════════════════════════════════════
# TAB 2: TEXT READING
# ═════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### 📖 Text Reading")
    mode2 = st.radio("Input:", ["📷 Capture dari Kamera", "📤 Upload Gambar"], horizontal=True)

    img_ph2, res_ph2, aud_ph2 = st.empty(), st.empty(), st.empty()

    def run_ocr_image(pil_img):
        arr = np.array(pil_img)
        if len(arr.shape) == 3 and arr.shape[2] >= 3:
            arr = cv2.cvtColor(arr[:, :, :3], cv2.COLOR_RGB2BGR)
        img_ph2.image(pil_img, use_container_width=True)
        text = perform_ocr_on_frame(arr, st.session_state.ocr_engine, ocr_min_conf)
        res_ph2.markdown(f'<div class="ocr-result">📝 {text}</div>', unsafe_allow_html=True)
        if text and text != "Tidak ada teks terdeteksi" and len(text) > 3 and enable_tts:
            audio = get_audio_bytes(f"Ada tulisan: {text}")
            if audio:
                aud_ph2.empty()
                play_audio_safe(aud_ph2, audio)
                st.success("🔊 Audio diputar")

    if mode2 == "📷 Capture dari Kamera":
        cam_ocr = st.camera_input("📸 Ambil foto untuk baca teks")
        if cam_ocr is not None:
            if st.session_state.ocr_engine is None: st.error("⚠️ Load OCR dulu di sidebar!")
            else:
                from PIL import Image
                run_ocr_image(Image.open(cam_ocr))
    else:
        up_img = st.file_uploader("Upload gambar", type=['jpg','jpeg','png','bmp'])
        if up_img:
            if st.session_state.ocr_engine is None: st.error("⚠️ Load OCR dulu di sidebar!")
            else:
                from PIL import Image
                run_ocr_image(Image.open(up_img))

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
        c3.markdown(f'<div class="stat-box"><div class="stat-value" style="color:#ffaa00;">{warn}</div><div class="stat-label">Waspada</div></div>', unsafe_allow_html=True)
        c4.markdown(f'<div class="stat-box"><div class="stat-value" style="color:#00c073;">{aman}</div><div class="stat-label">Aman</div></div>', unsafe_allow_html=True)

        df = pd.DataFrame([{
            'Waktu': d['timestamp'].strftime("%H:%M:%S"),
            'Objek': get_indo_name(d['class']),
            'Confidence': f"{d['confidence']:.1%}",
            'Risiko': d['risk_level'],
            'Posisi': 'Kiri' if d['position_x'] < 0.35 else ('Kanan' if d['position_x'] > 0.65 else 'Depan'),
        } for d in hist[-100:]])
        st.dataframe(df, use_container_width=True)

        if st.button("🗑️ Hapus Semua Data"):
            st.session_state.detection_history = []
            st.rerun()
    else:
        st.info("📊 Belum ada data deteksi. Mulai deteksi di tab Detection.")

st.divider()
st.markdown("""
<div style="text-align:center; color:#999; font-size:0.8rem; padding:1rem 0;">
    <strong>Asisten Navigasi Tunanetra v5.1</strong> • YOLOv11 • EasyOCR • gTTS
</div>
""", unsafe_allow_html=True)
