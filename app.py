# =====================================================================
# ASISTEN NAVIGASI TUNANETRA - STREAMLIT v6.0 (COLAB LOGIC)
# =====================================================================
# Logika deteksi diambil dari Fiks_Indo_Video_2.ipynb
# - Model2 (best.pt) conf=0.25, stream=True
# - KELAS_BEST_DIPAKAI dari Colab
# - AHP + Entropy bobot
# - Instruksi GESER/BERHENTI
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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="Asisten Tunanetra",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

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
.pill-warn { background: #fffbe6; color: #ad6800; border-color: #ffe58f; }
.pill-ocr { background: #e8f4ff; color: #0066cc; border-color: #90c8ff; }
.pill-model { background: #f0e8ff; color: #6c3fff; border-color: #c4b0ff; }

.alert-danger { background: #fff0f0; border: 1px solid #ffaaaa; border-left: 4px solid #ff4444; border-radius: 8px; padding: 1rem; color: #cc2222; font-weight: 600; margin: 0.8rem 0; }
.alert-warning { background: #fffbe6; border: 1px solid #ffe58f; border-left: 4px solid #faad14; border-radius: 8px; padding: 1rem; color: #ad6800; font-weight: 600; margin: 0.8rem 0; }
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
    white-space: pre-wrap;
    word-wrap: break-word;
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
# ID_MAP DARI COLAB - LENGKAP
# ─────────────────────────────────────────────────────────────────────
ID_MAP = {
    # COCO classes
    "person": "orang", "car": "mobil", "bus": "bus", "truck": "truck",
    "motorcycle": "motor", "bicycle": "sepeda", "train": "kereta",
    "dog": "anjing", "cat": "kucing", "fire hydrant": "hidran",
    "stop sign": "rambu stop", "traffic light": "lampu lalu lintas",
    
    # Model2 (SafeWalkBD)
    "Animal": "hewan", "Person": "orang", "Vehicle": "kendaraan",
    "Train": "kereta", "Traffic- light": "lampu lalu lintas",
    "Stairs": "tangga", "Pothole": "lubang di jalan",
    "Over- bridge": "jembatan penyeberangan", "Railway": "rel kereta",
    "Road- barrier": "pembatas jalan", "Sidewalk": "trotoar",
    "Crosswalk": "jalur penyeberangan", "Obstacle": "rintangan",
    "Pole": "tiang", "Tree": "pohon", "Traffic-sign": "rambu",
    
    # Model3 (Rambu Indonesia)
    "Balai Pertolongan Pertama": "balai pertolongan pertama",
    "Banyak Anak-Anak": "area banyak anak-anak",
    "Dilarang Belok Kanan": "dilarang belok kanan",
    "Dilarang Berhenti": "dilarang berhenti",
    "Dilarang Berjalan Terus": "dilarang berjalan terus",
    "Dilarang Masuk": "dilarang masuk",
    "Dilarang Mendahului": "dilarang mendahului",
    "Dilarang Parkir": "dilarang parkir",
    "Dilarang Putar Balik": "dilarang putar balik",
    "Gereja": "gereja",
    "Hati-Hati": "rambu hati-hati",
    "Jalur Penyebrangan": "jalur penyeberangan",
    "Lampu Lalu Lintas": "lampu lalu lintas",
    "Larangan Kecepatan - 30km-jam": "batas kecepatan 30 km per jam",
    "Larangan Kecepatan - 40km-jam": "batas kecepatan 40 km per jam",
    "Larangan Kendaraan MST - 10 Ton": "batas muatan 10 ton",
    "Masjid": "masjid",
    "Pemberhentian Bus": "pemberhentian bus",
    "Perintah Ikuti Bundaran": "ikuti arah bundaran",
    "Perintah Jalur Sepeda": "jalur sepeda",
    "Perintah Lajur Kiri": "gunakan lajur kiri",
    "Perintah Pilih Satu Jalur": "pilih satu jalur",
    "Persimpangan 3 Prioritas": "persimpangan tiga",
    "Persimpangan 3 Sisi Kanan Prioritas": "persimpangan tiga kanan prioritas",
    "Persimpangan 3 Sisi Kiri Prioritas": "persimpangan tiga kiri prioritas",
    "Persimpangan Empat": "persimpangan empat",
    "Putar Balik": "area putar balik",
    "Rumah Sakit": "rumah sakit",
    "SPBU": "pom bensin",
    "Tempat Parkir": "tempat parkir",
}

def id_nama(n):
    if not isinstance(n, str): return str(n)
    n_clean = n.strip()
    for key, value in ID_MAP.items():
        if key.lower() == n_clean.lower():
            return value
    return n_clean.lower().replace("-", " ")

# ─────────────────────────────────────────────────────────────────────
# KELAS_BEST_DIPAKAI DARI COLAB
# ─────────────────────────────────────────────────────────────────────
KELAS_BEST_DIPAKAI = {
    "Stairs", "Pothole", "Over-bridge", "Railway",
    "Road-barrier", "Sidewalk", "Crosswalk", "Obstacle", "Pole",
    "Vehicle", "Animal",
}

# ─────────────────────────────────────────────────────────────────────
# AHP BOBOT DARI COLAB
# ─────────────────────────────────────────────────────────────────────
TINGKAT = {
    "sangat_tinggi": ["car", "bus", "truck", "motorcycle", "train",
                      "Vehicle", "Train", "Pothole", "Stairs"],
    "tinggi": ["bicycle", "Road- barrier", "Obstacle", "Over- bridge", "Railway"],
    "sedang": ["person", "dog", "cat", "fire hydrant", "stop sign",
               "Person", "Animal", "Pole"],
    "rendah": ["bench", "backpack", "handbag", "suitcase", "potted plant",
               "chair", "umbrella", "Tree", "Sidewalk", "Crosswalk"],
    "informasi": ["traffic light", "Traffic- light", "Traffic- sign"],
}
URUT = ["sangat_tinggi", "tinggi", "sedang", "rendah", "informasi"]

# Matriks AHP
A = np.array([
    [1, 2, 3, 5, 7],
    [1/2, 1, 2, 4, 6],
    [1/3, 1/2, 1, 3, 5],
    [1/5, 1/4, 1/3, 1, 3],
    [1/7, 1/6, 1/5, 1/3, 1],
], dtype=float)

kol = A / A.sum(axis=0)
bobot_tingkat = kol.mean(axis=1)
bobot_tingkat /= bobot_tingkat.sum()

skala = bobot_tingkat / bobot_tingkat.max()
peta = {t: s for t, s in zip(URUT, skala)}
BOBOT_OBJEK = {o: round(float(peta[t]), 3) for t, items in TINGKAT.items() for o in items}
BOBOT_DEFAULT = 0.5
BOBOT_RAMBU = round(float(peta["informasi"]), 3)

print("Bobot AHP:", {t: round(float(w), 3) for t, w in zip(URUT, bobot_tingkat)})

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
for key, default_value in session_defaults.items():
    if key not in st.session_state:
        st.session_state[key] = default_value

def add_log(msg):
    st.session_state.log.insert(0, (time.strftime("%H:%M:%S"), msg))
    st.session_state.log = st.session_state.log[:30]

# ─────────────────────────────────────────────────────────────────────
# FUNGSI AUDIO
# ─────────────────────────────────────────────────────────────────────
def play_audio_safe(placeholder, audio_bytes):
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

# ─────────────────────────────────────────────────────────────────────
# FUNGSI GET_INDO_NAME (pakai ID_MAP)
# ─────────────────────────────────────────────────────────────────────
def get_indo_name(name):
    return id_nama(name)

def is_rambu(name):
    n = name.lower()
    kw = {'rambu', 'lampu lalu lintas', 'traffic light', 'stop sign', 'dilarang',
          'hati-hati', 'rumah sakit', 'masjid', 'gereja', 'spbu',
          'pom bensin', 'parkir', 'persimpangan', 'bundaran', 'jalur',
          'kecepatan', 'belok', 'berhenti', 'masuk', 'putar balik', 'sepeda'}
    return any(k in n for k in kw)

# ─────────────────────────────────────────────────────────────────────
# DETEKSI - PAKAI LOGIKA COLAB
# ─────────────────────────────────────────────────────────────────────
def process_frame_detection(frame, model, conf=0.25, model_type='best'):
    """Deteksi frame - conf=0.25 untuk best.pt seperti Colab"""
    if model is None: return frame, []
    try:
        # COLAB: conf=0.25 untuk model2, conf=0.7 untuk model1
        if model_type == 'best':
            results = model.predict(frame, conf=0.25, verbose=False)
        else:
            results = model.predict(frame, conf=conf, verbose=False)
        
        detections = []
        if results and len(results) > 0:
            result = results[0]
            if result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes.xyxy.cpu().numpy()
                confidences = result.boxes.conf.cpu().numpy()
                classes = result.boxes.cls.cpu().numpy().astype(int)
                fh, fw = frame.shape[:2]
                
                for box, cs, ci in zip(boxes, confidences, classes):
                    x1, y1, x2, y2 = map(int, box)
                    cn = result.names[ci]
                    
                    area = (x2-x1)*(y2-y1)
                    ar = area/(fw*fh)
                    px = ((x1+x2)/2)/fw
                    
                    # Filter area terlalu kecil
                    if ar < 0.005:
                        continue
                    
                    # COLAB: hanya pakai KELAS_BEST_DIPAKAI untuk model2
                    if model_type == 'best' and cn not in KELAS_BEST_DIPAKAI:
                        continue
                    
                    # Risk level berdasarkan AHP
                    bobot = BOBOT_OBJEK.get(cn, BOBOT_DEFAULT)
                    
                    # Komponen risiko
                    comp_dekat = ar
                    comp_jalur = max(0.0, 1 - abs(px - 0.5)/0.5)
                    
                    # Skor risiko
                    risiko = bobot * (comp_dekat * 0.7 + comp_jalur * 0.3)
                    
                    # Klasifikasi
                    if cn in ["Pothole", "Obstacle", "Stairs", "Road-barrier"] or \
                       cn.lower() in ["pothole", "obstacle", "stairs", "road-barrier"]:
                        if ar > 0.03:
                            risk_level = 'BAHAYA'
                        else:
                            risk_level = 'WASPADA'
                    elif cn in ["Vehicle", "Car", "Bus", "Truck"] or \
                         cn.lower() in ["vehicle", "car", "bus", "truck"]:
                        if ar > 0.08:
                            risk_level = 'BAHAYA'
                        else:
                            risk_level = 'WASPADA'
                    elif cn.lower() in ["person", "orang"]:
                        risk_level = 'WASPADA'
                    else:
                        risk_level = 'AMAN'
                    
                    detections.append({
                        'class': cn,
                        'confidence': float(cs),
                        'area_ratio': ar,
                        'position_x': px,
                        'risk_level': risk_level,
                        'bobot': bobot,
                        'risiko': risiko,
                        'bbox': (x1, y1, x2, y2),
                        'timestamp': datetime.now()
                    })
                    
                    color = (0,0,255) if risk_level=='BAHAYA' else (0,165,255) if risk_level=='WASPADA' else (0,255,0)
                    cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)
                    label = f"{get_indo_name(cn)} {cs:.0%}"
                    cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        return frame, detections
    except Exception as e:
        logger.error(f"Detection error: {e}")
        return frame, []

def process_frame_detection_multi(frame, model1, model2, model3, conf=0.5):
    out = frame.copy()
    all_d = []
    
    # Model1: General (conf=0.7 seperti Colab)
    if model1:
        out, d = process_frame_detection(out, model1, conf=0.7, model_type='general')
        all_d.extend(d)
    
    # Model2: best.pt (conf=0.25 seperti Colab)
    if model2:
        out, d = process_frame_detection(out, model2, conf=0.25, model_type='best')
        all_d.extend(d)
    
    # Model3: Rambu
    if model3:
        out, d = process_frame_detection(out, model3, conf=0.25, model_type='best')
        all_d.extend(d)
    
    return out, all_d

# ─────────────────────────────────────────────────────────────────────
# FUNGSI INSTRUKSI - DARI COLAB
# ─────────────────────────────────────────────────────────────────────
def get_instruksi(dets):
    """Instruksi berdasarkan posisi objek - dari Colab"""
    if not dets:
        return "JALAN TERUS"
    
    # Filter risiko >= 0.05
    filtered = [d for d in dets if d.get('risiko', 0) >= 0.05]
    if not filtered:
        return "JALAN TERUS"
    
    # Klasifikasi
    FATAL = {"pothole", "obstacle", "stairs", "road-barrier", "pole"}
    fisik = [d for d in filtered if d['class'].lower() in FATAL]
    orang = [d for d in filtered if d['class'].lower() == "person"]
    
    # Hitung risiko per zona
    kr = sum(d['risiko'] for d in fisik if d['position_x'] < 0.3)
    tg = sum(d['risiko'] for d in fisik if 0.3 <= d['position_x'] <= 0.7)
    kn = sum(d['risiko'] for d in fisik if d['position_x'] > 0.7)
    
    AMBANG = 0.1  # threshold dari Colab
    
    # Logika Fatal
    if fisik and tg > AMBANG:
        if kr > AMBANG and kn > AMBANG:
            max_area = max((d['area_ratio'] for d in fisik), default=0)
            if max_area >= 0.08:
                return "BERHENTI SEKARANG"
            return "BERHENTI"
        return "GESER KE KANAN" if kr <= kn else "GESER KE KIRI"
    
    # Logika Orang
    if orang:
        orang_dekat = [o for o in orang if o.get('area_ratio', 0) >= 0.04]
        if orang_dekat:
            o = orang_dekat[0]
            if o['position_x'] > 0.5:
                return "GESER SEDIKIT KE KIRI"
            else:
                return "GESER SEDIKIT KE KANAN"
    
    # Panduan Jalan
    if tg < AMBANG:
        if kr > AMBANG and kr > kn:
            return "JALAN TERUS, AGAK KE KANAN"
        if kn > AMBANG and kn > kr:
            return "JALAN TERUS, AGAK KE KIRI"
        return "JALAN TERUS"
    
    return "JALAN TERUS"

# ─────────────────────────────────────────────────────────────────────
# OCR - DARI COLAB
# ─────────────────────────────────────────────────────────────────────
def clean_ocr_text(raw_text):
    if not raw_text: return "Tidak ada teks terdeteksi"
    # Hapus karakter aneh
    text = re.sub(r'[^a-zA-Z0-9\s\.,!?\-]', ' ', raw_text)
    text = ' '.join(text.split())
    if len(text) < 3: return "Tidak ada teks terdeteksi"
    return text

def perform_ocr_on_frame(frame, ocr_engine, min_conf=0.20):
    if ocr_engine is None: return "OCR tidak tersedia"
    try:
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
        h, w = gray.shape
        if w < 400:
            scale = 600/w
            gray = cv2.resize(gray, (600, int(h*scale)), interpolation=cv2.INTER_CUBIC)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
        sharp = cv2.filter2D(enhanced, -1, kernel)
        
        results = None
        try:
            results = ocr_engine.readtext(sharp, detail=1, paragraph=False)
        except: pass
        if not results:
            try:
                results = ocr_engine.readtext(enhanced, detail=1, paragraph=False)
            except: pass
        if not results:
            try:
                results = ocr_engine.readtext(frame, detail=1, paragraph=False)
            except: pass
        
        if results:
            texts = []
            for (bbox, text, conf) in results:
                text = text.strip()
                if len(text) > 1 and conf > min_conf:
                    texts.append(text)
            if texts:
                raw = ' '.join(texts)
                cleaned = clean_ocr_text(raw)
                return cleaned
        return "Tidak ada teks terdeteksi"
    except Exception as e:
        return f"Error: {str(e)[:30]}"

def texts_are_similar(t1, t2, threshold=0.6):
    if not t1 or not t2: return False
    a = ' '.join(t1.lower().split())
    b = ' '.join(t2.lower().split())
    if a == b: return True
    if len(a) < 3 or len(b) < 3: return a == b
    def bg(s): return set(s[i:i+2] for i in range(len(s)-1))
    s1, s2 = bg(a), bg(b)
    if not s1 or not s2: return a == b
    return len(s1 & s2) / len(s1 | s2) >= threshold

# ─────────────────────────────────────────────────────────────────────
# ALERT FUNCTIONS
# ─────────────────────────────────────────────────────────────────────
def generate_rambu_alert(name):
    n = name.lower()
    m = {'stop sign':'Ada rambu stop. Berhenti!',
         'rambu stop':'Ada rambu stop. Berhenti!',
         'dilarang masuk':'Dilarang masuk!',
         'dilarang parkir':'Dilarang parkir',
         'dilarang berhenti':'Dilarang berhenti',
         'hati-hati':'Hati-hati!',
         'lampu lalu lintas':'Ada lampu lalu lintas',
         'rumah sakit':'Ada rumah sakit',
         'masjid':'Ada masjid',
         'gereja':'Ada gereja',
         'pom bensin':'Ada pom bensin',
         'persimpangan':'Ada persimpangan'}
    for k,v in m.items():
        if k in n: return v
    return f"Ada {get_indo_name(n)}"

def generate_alert(name, pos_x, area):
    nama = get_indo_name(name)
    if pos_x < 0.3: pos,arah = "di kiri", "Geser ke kanan"
    elif pos_x > 0.7: pos,arah = "di kanan", "Geser ke kiri"
    else: pos,arah = "di depan", "Berhenti"
    if area > 0.08:
        return f"Awas! {nama} sangat dekat {pos}! {arah} sekarang!"
    elif area > 0.03:
        return f"Awas! Ada {nama} {pos}. {arah}!"
    else:
        return f"Hati-hati, ada {nama} {pos}. {arah} pelan-pelan."

def handle_alerts(dets, now_t, en_audio, cooldown, warn_ph, status_ph, ad_ph, ar_ph):
    # Hanya objek dengan risiko >= 0.05
    valid = [d for d in dets if d.get('risiko', 0) >= 0.05]
    danger = [d for d in valid if d['risk_level'] == 'BAHAYA']
    waspada = [d for d in valid if d['risk_level'] == 'WASPADA']
    rambu = [d for d in valid if is_rambu(d['class'])]
    
    current = {d['class'] for d in valid}
    st.session_state.alerted_objects &= current

    if danger and en_audio:
        d = danger[0]
        k = d['class']
        if k not in st.session_state.alerted_objects:
            msg = generate_alert(d['class'], d['position_x'], d['area_ratio'])
            warn_ph.markdown(f'<div class="alert-danger">🚨 {msg}</div>', unsafe_allow_html=True)
            a = get_audio_bytes(msg)
            if a: ad_ph.empty(); play_audio_safe(ad_ph, a)
            st.session_state.alerted_objects.add(k)
            add_log(f"BAHAYA: {get_indo_name(k)}")
        else:
            warn_ph.markdown(f'<div class="alert-danger">🚨 Masih ada {get_indo_name(d["class"])}!</div>', unsafe_allow_html=True)
        status_ph.markdown('<div class="pills"><span class="pill pill-run">● AKTIF</span><span class="pill pill-danger">● BAHAYA</span></div>', unsafe_allow_html=True)
    elif waspada and en_audio:
        d = waspada[0]
        k = f"w_{d['class']}"
        if (now_t - st.session_state.last_alert_time[k]) >= cooldown:
            msg = generate_alert(d['class'], d['position_x'], d['area_ratio'])
            warn_ph.markdown(f'<div class="alert-warning">⚡ {msg}</div>', unsafe_allow_html=True)
            a = get_audio_bytes(msg)
            if a: ad_ph.empty(); play_audio_safe(ad_ph, a)
            st.session_state.last_alert_time[k] = now_t
        status_ph.markdown('<div class="pills"><span class="pill pill-run">● AKTIF</span><span class="pill pill-warn">● WASPADA</span></div>', unsafe_allow_html=True)
    elif rambu and en_audio:
        d = rambu[0]
        k = f"r_{d['class']}"
        if (now_t - st.session_state.last_alert_time[k]) >= cooldown:
            msg = generate_rambu_alert(d['class'])
            warn_ph.markdown(f'<div class="alert-info">ℹ️ {msg}</div>', unsafe_allow_html=True)
            a = get_audio_bytes(msg)
            if a: ar_ph.empty(); play_audio_safe(ar_ph, a)
            st.session_state.last_alert_time[k] = now_t
        status_ph.markdown('<div class="pills"><span class="pill pill-run">● AKTIF</span><span class="pill pill-ocr">● RAMBU</span></div>', unsafe_allow_html=True)
    else:
        # Tampilkan instruksi
        instruksi = get_instruksi(valid)
        warn_ph.markdown(f'<div class="alert-success">✅ {instruksi}</div>', unsafe_allow_html=True)
        status_ph.markdown('<div class="pills"><span class="pill pill-run">● AKTIF</span><span class="pill pill-ok">● AMAN</span></div>', unsafe_allow_html=True)
    return danger, rambu

def handle_ocr_dedup(frame, ocr_eng, min_conf, en_tts, ocr_ph, aocr_ph):
    text = perform_ocr_on_frame(frame, ocr_eng, min_conf)
    if text and text != "Tidak ada teks terdeteksi" and len(text) > 3:
        if not texts_are_similar(text, st.session_state.last_ocr_text):
            st.session_state.last_ocr_text = text
            ocr_ph.markdown(f'<div class="ocr-result">📝 {text}</div>', unsafe_allow_html=True)
            add_log(f"OCR: {text[:40]}")
            if en_tts:
                a = get_audio_bytes(f"Ada tulisan: {text}")
                if a: aocr_ph.empty(); play_audio_safe(aocr_ph, a)
            return True
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
with c1: st.markdown('<div class="header-logo">👁️</div>', unsafe_allow_html=True)
with c2: st.markdown('<div class="header-text"><h1>Asisten Navigasi Tunanetra</h1><p>Deteksi Objek + Baca Teks — v6.0 (Colab Logic)</p></div>', unsafe_allow_html=True)
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
    with st.expander("🏔️ Model 2: Tangga/Lubang"):
        up2 = st.file_uploader("Upload best.pt", type=['pt'], key='m2up')
        if up2 and st.button("Load M2"):
            with st.spinner("Memuat..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as tmp:
                    tmp.write(up2.read())
                from ultralytics import YOLO
                st.session_state.model2 = YOLO(tmp.name)
                add_log("M2 loaded")
                st.success("✅ M2 Dimuat!")
    with st.expander("🚦 Model 3: Rambu"):
        up3 = st.file_uploader("Upload best_rambu.pt", type=['pt'], key='m3up')
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
    st.markdown(f'<div class="pills"><span class="pill pill-model">{s1} M1</span><span class="pill pill-model">{s2} M2</span><span class="pill pill-model">{s3} M3</span><span class="pill pill-model">{so} OCR</span></div>', unsafe_allow_html=True)
    enable_audio = st.checkbox("🔊 Audio", value=True)
    alert_cooldown = st.slider("Cooldown (s)", 2, 10, 5)
    ocr_min_conf = st.slider("OCR Confidence", 0.1, 0.9, 0.2, 0.05)
    ocr_scan_interval = st.slider("OCR Scan Interval", 1, 10, 3)
    enable_tts = st.checkbox("🔊 TTS", value=True)
    show_logs = st.checkbox("Show Logs", value=True)
    frame_skip = st.slider("Frame Skip", 1, 10, 3, help="Skip N frame. Lebih tinggi = lebih lancar")

# ─────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🎯 Detection", "📖 Text Reading", "📊 Statistics"])

with tab1:
    mode = st.radio("Mode:", ["📹 Webcam", "📤 Upload Video"], horizontal=True)
    st.divider()
    
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

    # ============================================================
    # WEBCAM
    # ============================================================
    if mode == "📹 Webcam":
        st.info("📸 Gunakan webcam untuk deteksi real-time")
        run = st.toggle("🎥 Aktifkan Webcam")
        btn_baca = st.button("📖 Baca Teks")
        
        if btn_baca:
            st.session_state.ocr_triggered_cam = True
            st.info("🔍 OCR akan membaca teks dari frame webcam...")
        
        if run:
            if st.session_state.model1 is None:
                st.error("⚠️ Load YOLO dulu!")
            else:
                cap = None
                backends_to_try = [cv2.CAP_V4L2, cv2.CAP_DSHOW, cv2.CAP_ANY, 0]
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
                        
                        frame_ann, dets = process_frame_detection_multi(frame, m1, m2, m3, conf=0.5)
                        frame_ph.image(cv2.cvtColor(frame_ann, cv2.COLOR_BGR2RGB), use_container_width=True)
                        st.session_state.last_frame = orig
                        
                        if st.session_state.ocr_triggered_cam and ocr is not None:
                            if cnt % ocr_scan_interval == 0:
                                text = perform_ocr_on_frame(orig, ocr, ocr_min_conf)
                                if text and text != "Tidak ada teks terdeteksi" and len(text) > 3:
                                    if text != st.session_state.last_ocr_text:
                                        st.session_state.last_ocr_text = text
                                        ocr_ph.markdown(f'<div class="ocr-result">📝 {text}</div>', unsafe_allow_html=True)
                                        if enable_tts:
                                            a = get_audio_bytes(f"Ada tulisan: {text}")
                                            if a:
                                                audio_ocr_ph.empty()
                                                play_audio_safe(audio_ocr_ph, a)
                                                st.success("🔊 Suara diputar")
                                    else:
                                        ocr_ph.markdown(f'<div class="ocr-result">📝 {text} (sama)</div>', unsafe_allow_html=True)
                        
                        danger, rambu = handle_alerts(dets, now, enable_audio, alert_cooldown, warn_ph, status_ph, audio_danger_ph, audio_rambu_ph)
                        
                        m_det.metric("Detections", len(dets))
                        m_danger.metric("⚠️ Danger", len(danger))
                        elapsed = time.time() - start
                        m_fps.metric("FPS", f"{cnt/elapsed:.1f}" if elapsed>0 else "0")
                        if show_logs:
                            log_ph.markdown('<br>'.join([f'[{t}] {m}' for t,m in st.session_state.log[:10]]), unsafe_allow_html=True)
                        
                        if btn_baca and not st.session_state.ocr_triggered_cam:
                            st.session_state.ocr_triggered_cam = True
                        
                        time.sleep(0.01)
                    
                    cap.release()
                    st.session_state.last_ocr_text = ''
                    st.session_state.ocr_triggered_cam = False
                    st.success("Webcam dihentikan.")

    # ============================================================
    # UPLOAD VIDEO
    # ============================================================
    else:
        uploaded = st.file_uploader("Upload video", type=['mp4','avi','mov','mkv','webm','m4v'])
        
        if uploaded is not None:
            current_name = uploaded.name
            if st.session_state.last_uploaded_name != current_name:
                st.session_state.ocr_triggered = False
                st.session_state.ocr_frame_count = 0
                st.session_state.last_uploaded_name = current_name
                st.session_state.last_ocr_text = ''
                st.session_state.alerted_objects.clear()
                ocr_ph.empty()
                st.info("🔄 Video baru diupload. OCR di-reset.")
        
        col1, col2 = st.columns(2)
        with col1: btn_start = st.button("▶️ Start", use_container_width=True)
        with col2: btn_ocr = st.button("📖 Baca Teks", use_container_width=True)
        
        if btn_ocr:
            st.session_state.ocr_triggered = True
            st.session_state.ocr_frame_count = 0
            st.session_state.last_ocr_text = ''
            st.info("🔍 OCR aktif.")
        
        if uploaded and btn_start:
            if not st.session_state.model1 and not st.session_state.model2:
                st.error("⚠️ Load YOLO!"); st.stop()
            st.session_state.alerted_objects.clear()
            
            suffix = os.path.splitext(uploaded.name)[1] or '.mp4'
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded.read())
                vid_path = tmp.name
            
            try:
                cap = cv2.VideoCapture(vid_path)
                if not cap.isOpened():
                    st.error("❌ Video tidak bisa dibuka! Coba format MP4 H.264.")
                    st.stop()
                
                total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps_v = cap.get(cv2.CAP_PROP_FPS) or 30.0
                m1, m2, m3 = st.session_state.model1, st.session_state.model2, st.session_state.model3
                ocr = st.session_state.ocr_engine
                cnt, start = 0, time.time()
                prog = st.progress(0)
                
                while True:
                    ok, frame = cap.read()
                    if not ok: break
                    cnt += 1
                    if cnt % frame_skip != 0: continue
                    
                    # Resize kecil untuk performa
                    frame = cv2.resize(frame, (320, 240))
                    orig = frame.copy()
                    now_sec = cnt / fps_v
                    prog.progress(min(cnt/max(total,1), 1.0), text=f"{cnt}/{total} ({now_sec:.1f}s)")
                    
                    # Deteksi dengan logika Colab
                    frame_ann, dets = process_frame_detection_multi(frame, m1, m2, m3, conf=0.5)
                    frame_ph.image(cv2.cvtColor(frame_ann, cv2.COLOR_BGR2RGB), use_container_width=True)
                    st.session_state.last_frame = orig
                    
                    for d in dets: st.session_state.detection_history.append(d)
                    st.session_state.detection_history = st.session_state.detection_history[-500:]
                    
                    danger, rambu = handle_alerts(dets, time.time(), enable_audio, alert_cooldown, warn_ph, status_ph, audio_danger_ph, audio_rambu_ph)
                    
                    if st.session_state.ocr_triggered and ocr:
                        st.session_state.ocr_frame_count += 1
                        if st.session_state.ocr_frame_count % ocr_scan_interval == 0:
                            handle_ocr_dedup(orig, ocr, ocr_min_conf, enable_tts, ocr_ph, audio_ocr_ph)
                    
                    m_det.metric("Detections", len(dets))
                    m_danger.metric("⚠️ Danger", len(danger))
                    elapsed = time.time() - start
                    m_fps.metric("FPS", f"{cnt/elapsed:.1f}" if elapsed>0 else "0")
                    if show_logs:
                        log_ph.markdown('<br>'.join([f'[{t}] {m}' for t,m in st.session_state.log[:10]]), unsafe_allow_html=True)
                    
                    time.sleep(0.01)
                
                cap.release()
                prog.empty()
                st.success(f"✅ Selesai! {cnt} frame.")
                st.session_state.ocr_triggered = False
            except Exception as e:
                st.error(f"Error: {e}")
            finally:
                try: os.unlink(vid_path)
                except: pass

# ─────────────────────────────────────────────────────────────────────
# TAB 2: TEXT READING
# ─────────────────────────────────────────────────────────────────────
with tab2:
    st.markdown("### 📖 Text Reading")
    mode2 = st.radio("Input:", ["📷 Kamera", "📤 Upload"], horizontal=True)
    img_ph, res_ph, aud_ph = st.empty(), st.empty(), st.empty()
    if mode2 == "📷 Kamera":
        cam2 = st.camera_input("📸 Foto untuk baca teks")
        if cam2:
            if not st.session_state.ocr_engine: st.error("⚠️ Load OCR!")
            else:
                from PIL import Image
                img = Image.open(cam2)
                arr = np.array(img)
                img_ph.image(img, use_container_width=True)
                text = perform_ocr_on_frame(arr, st.session_state.ocr_engine, ocr_min_conf)
                res_ph.markdown(f'<div class="ocr-result">📝 {text}</div>', unsafe_allow_html=True)
                if text and text != "Tidak ada teks terdeteksi" and len(text) > 3 and enable_tts:
                    a = get_audio_bytes(f"Ada tulisan: {text}")
                    if a: aud_ph.empty(); play_audio_safe(aud_ph, a)
    else:
        ui = st.file_uploader("Upload gambar", type=['jpg','jpeg','png','bmp'])
        if ui:
            from PIL import Image
            img = Image.open(ui)
            arr = np.array(img)
            img_ph.image(img, use_container_width=True)
            if not st.session_state.ocr_engine: st.error("⚠️ Load OCR!")
            else:
                text = perform_ocr_on_frame(arr, st.session_state.ocr_engine, ocr_min_conf)
                res_ph.markdown(f'<div class="ocr-result">📝 {text}</div>', unsafe_allow_html=True)
                if text and text != "Tidak ada teks terdeteksi" and len(text) > 3 and enable_tts:
                    a = get_audio_bytes(f"Ada tulisan: {text}")
                    if a: aud_ph.empty(); play_audio_safe(aud_ph, a)

# ─────────────────────────────────────────────────────────────────────
# TAB 3: STATISTICS
# ─────────────────────────────────────────────────────────────────────
with tab3:
    st.markdown("### 📊 Statistics")
    if st.session_state.detection_history:
        h = st.session_state.detection_history
        total = len(h)
        danger = len([d for d in h if d['risk_level'] == 'BAHAYA'])
        waspada = len([d for d in h if d['risk_level'] == 'WASPADA'])
        aman = len([d for d in h if d['risk_level'] == 'AMAN'])
        c1,c2,c3,c4 = st.columns(4)
        c1.markdown(f'<div class="stat-box"><div class="stat-value">{total}</div><div class="stat-label">Total</div></div>', unsafe_allow_html=True)
        c2.markdown(f'<div class="stat-box"><div class="stat-value" style="color:#ff4444">{danger}</div><div class="stat-label">Bahaya</div></div>', unsafe_allow_html=True)
        c3.markdown(f'<div class="stat-box"><div class="stat-value" style="color:#faad14">{waspada}</div><div class="stat-label">Waspada</div></div>', unsafe_allow_html=True)
        c4.markdown(f'<div class="stat-box"><div class="stat-value" style="color:#00c073">{aman}</div><div class="stat-label">Aman</div></div>', unsafe_allow_html=True)
        df = pd.DataFrame([{
            'Waktu': d['timestamp'].strftime("%H:%M:%S"),
            'Objek': get_indo_name(d['class']),
            'Conf': f"{d['confidence']:.0%}",
            'Risk': d['risk_level'],
            'Pos': 'Kiri' if d['position_x'] < 0.3 else ('Kanan' if d['position_x'] > 0.7 else 'Depan')
        } for d in h[-100:]])
        st.dataframe(df, use_container_width=True)
        if st.button("🗑️ Hapus"):
            st.session_state.detection_history = []
            st.session_state.alerted_objects.clear()
            st.rerun()
    else:
        st.info("📊 Belum ada data.")

st.divider()
st.markdown('<div style="text-align:center;color:#999;font-size:.8rem;padding:1rem 0;"><strong>Asisten Navigasi Tunanetra v6.0</strong> • YOLOv11 • EasyOCR • gTTS</div>', unsafe_allow_html=True)
