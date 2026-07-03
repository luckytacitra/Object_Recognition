# =====================================================================
# ASISTEN NAVIGASI TUNANETRA - STREAMLIT v5.10 (FIX SUARA ALL)
# =====================================================================
# PERBAIKAN:
# 1. Suara rintangan: setiap objek baru (beda class) bersuara
# 2. Suara teks: setiap teks baru (tidak mirip) bersuara
# 3. Cooldown turun ke 2 detik
# 4. OCR threshold turun ke 0.5 (lebih sensitif)
# 5. Reset alert saat video baru
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
# ID_MAP
# ─────────────────────────────────────────────────────────────────────
ID_MAP_VALID = {
    "person":"orang","car":"mobil","bus":"bus","truck":"truk",
    "motorcycle":"motor","bicycle":"sepeda","train":"kereta",
    "stop sign":"rambu stop","traffic light":"lampu lalu lintas",
    "Stairs":"tangga","Pothole":"lubang","Over-bridge":"jembatan penyeberangan",
    "Railway":"rel kereta","Road-barrier":"pembatas jalan","Sidewalk":"trotoar",
    "Crosswalk":"jalur penyeberangan","Obstacle":"rintangan","Pole":"tiang",
    "Vehicle":"kendaraan","Animal":"hewan",
    "Dilarang Masuk":"dilarang masuk",
    "Dilarang Parkir":"dilarang parkir",
    "Dilarang Berhenti":"dilarang berhenti",
    "Dilarang Belok Kanan":"dilarang belok kanan",
    "Dilarang Putar Balik":"dilarang putar balik",
    "Hati-Hati":"hati-hati",
    "Lampu Lalu Lintas":"lampu lalu lintas",
    "Rumah Sakit":"rumah sakit",
    "Masjid":"masjid",
    "Gereja":"gereja",
    "SPBU":"pom bensin",
    "Tempat Parkir":"tempat parkir",
    "Pemberhentian Bus":"pemberhentian bus",
    "Perintah Ikuti Bundaran":"ikuti arah bundaran",
    "Perintah Jalur Sepeda":"jalur sepeda",
    "Perintah Lajur Kiri":"gunakan lajur kiri",
    "Persimpangan":"persimpangan",
}

KELAS_VALID = set(ID_MAP_VALID.keys())
KELAS_VALID_LOWER = {k.lower(): k for k in KELAS_VALID}

def id_nama(n):
    if not isinstance(n, str): return str(n)
    nc = n.strip()
    if nc in ID_MAP_VALID: return ID_MAP_VALID[nc]
    for k, v in ID_MAP_VALID.items():
        if k.lower() == nc.lower(): return v
    return nc

# ─────────────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────────────
session_defaults = {
    'model1': None, 'model2': None, 'model3': None, 'ocr_engine': None,
    'last_frame': None, 'log': [], 'detection_history': [],
    'last_alert_time': defaultdict(lambda: -99.0),
    'ocr_triggered_cam': False,
    'ocr_triggered_vid': False,
    'ocr_frame_count': 0,
    'last_uploaded_name': None,
    'last_ocr_text': '',
    'last_danger_text': '',
    'detected_classes': set(),
}
for key, value in session_defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

def add_log(msg):
    try:
        st.session_state.log.insert(0, (time.strftime("%H:%M:%S"), str(msg)))
        st.session_state.log = st.session_state.log[:30]
    except Exception as e:
        logger.error(f"Add log error: {e}")

def render_log():
    try:
        if not st.session_state.log:
            return '<div style="color: #ccc; text-align: center; padding: 1rem;">Belum ada aktivitas</div>'
        rows = []
        for item in st.session_state.log[:10]:
            try:
                if isinstance(item, (tuple, list)) and len(item) >= 2:
                    rows.append(f'[{item[0]}] {item[1]}')
            except:
                continue
        if not rows:
            return '<div style="color: #ccc; text-align: center; padding: 1rem;">Belum ada aktivitas</div>'
        return '<br>'.join(rows)
    except:
        return '<div style="color: #ccc; text-align: center; padding: 1rem;">Error loading logs</div>'

# ─────────────────────────────────────────────────────────────────────
# FUNGSI AUDIO
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
# TERJEMAHAN
# ─────────────────────────────────────────────────────────────────────
def get_indo_name(name):
    return id_nama(name)

RAMBU_KW = {'rambu', 'lampu lalu lintas', 'stop sign', 'dilarang',
            'hati-hati', 'rumah sakit', 'masjid', 'gereja', 'spbu',
            'pom bensin', 'parkir', 'persimpangan', 'bundaran', 'jalur'}

def is_rambu(name):
    n = name.lower()
    for rk in RAMBU_KW:
        if rk in n:
            return True
    return False

def is_obstacle(name):
    n = name.lower()
    obstacle_kw = ['pothole','lubang','stairs','tangga','obstacle','rintangan',
                   'road-barrier','pembatas','pole','tiang']
    return any(k in n for k in obstacle_kw)

def generate_rambu_alert(name):
    n = name.lower()
    m = {
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
    
    if area > 0.12: return f"Awas! {nama} sangat dekat {pos}. {arah} sekarang!"
    elif area > 0.04: return f"Awas! Ada {nama} {pos}. {arah}!"
    else: return f"Hati-hati, ada {nama} {pos}. {arah} pelan-pelan."

# ============================================================
# OCR
# ============================================================
def perform_ocr_on_frame(frame, ocr_engine, min_conf=0.30):
    if ocr_engine is None: return "OCR engine tidak tersedia"
    try:
        if len(frame.shape) == 3: gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else: gray = frame
        h, w = gray.shape
        if w < 800:
            scale = 800 / w
            gray = cv2.resize(gray, (800, int(h * scale)), interpolation=cv2.INTER_CUBIC)
        denoised = cv2.fastNlMeansDenoising(gray, None, h=10)
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
                raw = ' '.join(texts)
                raw = re.sub(r'[^a-zA-Z0-9\s\.,!?\-:/()%]', '', raw)
                # Simpel autocorrect
                raw = raw.replace('dlarang', 'dilarang').replace('dlrang', 'dilarang')
                raw = raw.replace('parkr', 'parkir').replace('msuk', 'masuk')
                raw = raw.replace('brhenti', 'berhenti').replace('hti-hti', 'hati-hati')
                return raw.title() if len(raw) >= 3 else "Tidak ada teks terdeteksi"
        return "Tidak ada teks terdeteksi"
    except Exception as e: return f"Error: {str(e)[:30]}"

def texts_are_similar(t1, t2, threshold=0.5):
    """FIX: threshold turun ke 0.5 agar lebih sensitif bedakan teks"""
    if not t1 or not t2: return False
    a, b = ' '.join(t1.lower().split()), ' '.join(t2.lower().split())
    if a == b: return True
    if len(a) < 2 or len(b) < 2: return a == b
    s1, s2 = set(a[i:i+2] for i in range(len(a)-1)), set(b[i:i+2] for i in range(len(b)-1))
    if not s1 or not s2: return a == b
    return len(s1 & s2) / len(s1 | s2) >= threshold

# ============================================================
# DETEKSI
# ============================================================
def process_frame_detection(frame, model, conf=0.4, is_m2=False):
    if model is None: return frame, []
    try:
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
                    cls_name = result.names[cls_idx]
                    cls_lower = cls_name.lower()
                    if cls_lower not in KELAS_VALID_LOWER:
                        continue
                    cls_name = KELAS_VALID_LOWER[cls_lower]
                    x1, y1, x2, y2 = map(int, box)
                    area = (x2 - x1) * (y2 - y1)
                    area_ratio = area / (fw * fh)
                    if area_ratio < 0.002:
                        continue
                    pos_x = ((x1 + x2) / 2) / fw
                    is_obs = is_obstacle(cls_name)
                    is_ram = is_rambu(cls_name)
                    if is_obs:
                        if area_ratio > 0.04 and conf_score > 0.20:
                            risk_level = 'BAHAYA'
                        elif area_ratio > 0.01 and conf_score > 0.15:
                            risk_level = 'WASPADA'
                        else:
                            risk_level = 'AMAN'
                    elif is_ram:
                        risk_level = 'RAMBU'
                    else:
                        risk_level = 'AMAN'
                    if risk_level == 'AMAN' and area_ratio < 0.01:
                        continue
                    detections.append({
                        'class': cls_name,
                        'confidence': float(conf_score),
                        'area_ratio': area_ratio,
                        'position_x': pos_x,
                        'risk_level': risk_level,
                        'bbox': (x1, y1, x2, y2),
                        'timestamp': datetime.now()
                    })
                    color = (0, 0, 255) if risk_level == 'BAHAYA' else (0, 165, 255) if risk_level == 'WASPADA' else (0, 255, 0) if risk_level == 'AMAN' else (255, 165, 0)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    label = f"{get_indo_name(cls_name)} {conf_score:.2f}"
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        return frame, detections
    except Exception as e:
        logger.error(f"Detection error: {e}")
        return frame, []

def process_frame_detection_multi(frame, m1, m2, m3):
    out = frame.copy()
    all_d = []
    if m1:
        out, d = process_frame_detection(out, m1, conf=0.7, is_m2=False)
        all_d.extend(d)
    if m2:
        out, d = process_frame_detection(out, m2, conf=0.4, is_m2=True)
        all_d.extend(d)
    if m3:
        out, d = process_frame_detection(out, m3, conf=0.7, is_m2=False)
        all_d.extend(d)
    return out, all_d

# ─────────────────────────────────────────────────────────────────────
# ALERT - SUARA UNTUK SETIAP OBJEK BARU
# ─────────────────────────────────────────────────────────────────────
def handle_alerts(dets, now_time, enable_audio, warn_ph, status_ph, audio_ph, cooldown=2):
    danger = [d for d in dets if d['risk_level'] == 'BAHAYA']
    waspada = [d for d in dets if d['risk_level'] == 'WASPADA']
    rambu = [d for d in dets if d['risk_level'] == 'RAMBU']

    # FIX: RESET detected_classes setiap frame
    current_classes = {d['class'] for d in dets}
    
    # Cek objek baru yang belum pernah bersuara
    new_danger = [d for d in danger if d['class'] not in st.session_state.get('detected_classes', set())]
    new_waspada = [d for d in waspada if d['class'] not in st.session_state.get('detected_classes', set())]
    new_rambu = [d for d in rambu if d['class'] not in st.session_state.get('detected_classes', set())]

    # FIX: SUARA UNTUK OBJEK BARU
    if new_danger and enable_audio:
        d = new_danger[0]
        key = d['class']
        if (now_time - st.session_state.last_alert_time[key]) >= cooldown:
            msg = generate_alert(d['class'], d['position_x'], d['area_ratio'])
            warn_ph.markdown(f'<div class="alert-danger">🚨 {msg}</div>', unsafe_allow_html=True)
            audio = get_audio_bytes(msg)
            if audio: 
                audio_ph.empty()
                play_audio_safe(audio_ph, audio)
                add_log(f"BAHAYA: {msg}")
            st.session_state.last_alert_time[key] = now_time
            st.session_state.detected_classes.add(d['class'])
        status_ph.markdown('<div class="pills"><span class="pill pill-run">● AKTIF</span><span class="pill pill-danger">● BAHAYA</span></div>', unsafe_allow_html=True)
    
    elif new_waspada and enable_audio:
        d = new_waspada[0]
        key = d['class']
        if (now_time - st.session_state.last_alert_time[key]) >= cooldown:
            msg = generate_alert(d['class'], d['position_x'], d['area_ratio'])
            warn_ph.markdown(f'<div class="alert-warning">⚡ {msg}</div>', unsafe_allow_html=True)
            audio = get_audio_bytes(msg)
            if audio:
                audio_ph.empty()
                play_audio_safe(audio_ph, audio)
                add_log(f"WASPADA: {msg}")
            st.session_state.last_alert_time[key] = now_time
            st.session_state.detected_classes.add(d['class'])
        status_ph.markdown('<div class="pills"><span class="pill pill-run">● AKTIF</span><span class="pill pill-warning" style="background:#fff5e0;color:#cc8800;border-color:#ffcc66;">● WASPADA</span></div>', unsafe_allow_html=True)
    
    elif new_rambu and enable_audio:
        d = new_rambu[0]
        key = d['class']
        if (now_time - st.session_state.last_alert_time[key]) >= cooldown:
            msg = generate_rambu_alert(d['class'])
            warn_ph.markdown(f'<div class="alert-info">ℹ️ {msg}</div>', unsafe_allow_html=True)
            audio = get_audio_bytes(msg)
            if audio:
                audio_ph.empty()
                play_audio_safe(audio_ph, audio)
                add_log(f"RAMBU: {msg}")
            st.session_state.last_alert_time[key] = now_time
            st.session_state.detected_classes.add(d['class'])
        status_ph.markdown('<div class="pills"><span class="pill pill-run">● AKTIF</span><span class="pill pill-ocr">● RAMBU</span></div>', unsafe_allow_html=True)
    
    else:
        warn_ph.markdown('<div class="alert-success">✅ Jalur aman</div>', unsafe_allow_html=True)
        status_ph.markdown('<div class="pills"><span class="pill pill-run">● AKTIF</span><span class="pill pill-ok">● AMAN</span></div>', unsafe_allow_html=True)
    
    return danger, rambu

def handle_ocr_dedup(frame, ocr_eng, min_conf, en_tts, ocr_ph, aocr_ph):
    text = perform_ocr_on_frame(frame, ocr_eng, min_conf)
    if text and text != "Tidak ada teks terdeteksi" and len(text) > 3:
        # FIX: threshold 0.5 agar lebih sensitif
        if not texts_are_similar(text, st.session_state.last_ocr_text, threshold=0.5):
            st.session_state.last_ocr_text = text
            ocr_ph.markdown(f'<div class="ocr-result">📝 {text}</div>', unsafe_allow_html=True)
            add_log(f"OCR: {text[:40]}")
            if en_tts:
                a = get_audio_bytes(f"Ada tulisan: {text}")
                if a: 
                    aocr_ph.empty()
                    play_audio_safe(aocr_ph, a)
            return True
        else:
            ocr_ph.markdown(f'<div class="ocr-result" style="opacity:0.6;">📝 {text}<br><small>(Teks sama)</small></div>', unsafe_allow_html=True)
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
with c2: st.markdown('<div class="header-text"><h1>Asisten Navigasi Tunanetra</h1><p>Deteksi Objek & Teks Terpadu v5.10</p></div>', unsafe_allow_html=True)
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
    with st.expander("🏔️ Model 2: Tangga/Lubang (best.pt)"):
        up2 = st.file_uploader("Upload best.pt", type=['pt'], key='m2up')
        if up2 and st.button("Load M2"):
            with st.spinner("Memuat..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as tmp: tmp.write(up2.read())
                from ultralytics import YOLO
                st.session_state.model2 = YOLO(tmp.name)
                st.success("✅ M2 Dimuat!")
    with st.expander("🚦 Model 3: Rambu (best_rambu.pt)"):
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
    
    enable_audio = st.checkbox("🔊 Audio", value=True)
    alert_cooldown = st.slider("Cooldown (s)", 1, 5, 2, help="1 = suara lebih sering")
    ocr_min_conf = st.slider("OCR Confidence", 0.1, 0.9, 0.30, 0.05)
    ocr_scan_interval = st.slider("OCR Scan Interval", 1, 30, 10)
    frame_skip = st.slider("Frame Skip", 1, 10, 2)
    enable_tts = st.checkbox("🔊 TTS", value=True)
    show_logs = st.checkbox("📋 Logs", value=True)
    
    if st.button("🔁 Reset Suara"):
        st.session_state.last_alert_time = defaultdict(lambda: -99.0)
        st.session_state.last_ocr_text = ''
        st.session_state.detected_classes = set()
        st.success("Reset!")

# ─────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🎯 Detection", "📖 Text Reading", "📊 Statistics"])

with tab1:
    mode = st.radio("Mode:", ["📹 Webcam (Foto)", "📤 Upload Video"], horizontal=True)
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

    if show_logs: 
        log_ph = st.expander("📋 Logs")

    # ============================================================
    # WEBCAM
    # ============================================================
    if mode == "📹 Webcam (Foto)":
        st.markdown('<div class="alert-info">📸 Ambil foto → otomatis deteksi. Klik "Baca Teks" untuk OCR.</div>', unsafe_allow_html=True)
        
        _, col_btn = st.columns(2)
        with col_btn:
            btn_ocr_cam = st.button("📖 Baca Teks", key="ocr_cam", use_container_width=True)
        
        cam = st.camera_input("📸 Ambil foto", key="cam_tab1")
        
        if cam:
            if not st.session_state.model1 and not st.session_state.model2:
                st.error("⚠️ Load YOLO dulu!")
            else:
                from PIL import Image
                img = np.array(Image.open(cam))
                bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) if len(img.shape) == 3 else img
                bgr = cv2.resize(bgr, (640, 480))
                orig = bgr.copy()
                ann, dets = process_frame_detection_multi(bgr, st.session_state.model1, st.session_state.model2, st.session_state.model3)
                frame_ph.image(cv2.cvtColor(ann, cv2.COLOR_BGR2RGB), caption=f"{len(dets)} objek", use_container_width=True)
                for d in dets: st.session_state.detection_history.append(d)
                st.session_state.detection_history = st.session_state.detection_history[-500:]
                dl, rl = handle_alerts(dets, time.time(), enable_audio, warn_ph, status_ph, audio_alert_ph, cooldown=alert_cooldown)
                m_det.metric("Objek", len(dets))
                m_danger.metric("⚠️", len(dl))
                if btn_ocr_cam and st.session_state.ocr_engine:
                    text = perform_ocr_on_frame(orig, st.session_state.ocr_engine, ocr_min_conf)
                    if text and text != "Tidak ada teks terdeteksi" and len(text) > 3:
                        ocr_ph.markdown(f'<div class="ocr-result">📝 {text}</div>', unsafe_allow_html=True)
                        if enable_tts:
                            a = get_audio_bytes(f"Ada tulisan: {text}")
                            if a: 
                                audio_ocr_ph.empty()
                                play_audio_safe(audio_ocr_ph, a)
                    else:
                        ocr_ph.markdown('<div class="ocr-result" style="opacity:0.5">📝 Tidak ada teks</div>', unsafe_allow_html=True)
                if show_logs:
                    log_ph.markdown(render_log(), unsafe_allow_html=True)

    # ============================================================
    # UPLOAD VIDEO - FIX SUARA ALL
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
                st.session_state.last_alert_time = defaultdict(lambda: -99.0)
                st.session_state.detected_classes = set()
                ocr_ph.empty()
                st.info("🔄 Video baru diupload.")

        col1, col2 = st.columns(2)
        with col1: btn_start = st.button("▶️ Start Detection", use_container_width=True)
        with col2: btn_baca = st.button("📖 Baca Teks", use_container_width=True)

        if btn_baca:
            st.session_state.ocr_triggered_vid = True
            st.session_state.ocr_frame_count = 0
            st.info("🔍 OCR aktif.")

        if uploaded and btn_start:
            if st.session_state.model1 is None and st.session_state.model2 is None:
                st.error("⚠️ Load YOLO dulu!")
                st.stop()

            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
                tmp.write(uploaded.read())
                vid_path = tmp.name

            try:
                cap = cv2.VideoCapture(vid_path)
                if not cap.isOpened():
                    st.error("❌ Video gagal dibuka!")
                    st.stop()
                
                total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

                m1, m2, m3 = st.session_state.model1, st.session_state.model2, st.session_state.model3
                ocr = st.session_state.ocr_engine

                cnt, start = 0, time.time()
                prog = st.progress(0)
                
                # RESET UNTUK SUARA
                st.session_state.last_alert_time = defaultdict(lambda: -99.0)
                st.session_state.detected_classes = set()
                st.session_state.last_ocr_text = ''

                while True:
                    ok, frame = cap.read()
                    if not ok: break
                    cnt += 1
                    
                    if cnt % frame_skip != 0: continue
                    frame = cv2.resize(frame, (640, 480))
                    orig = frame.copy()

                    prog.progress(min(cnt / max(total, 1), 1.0), text=f"Frame {cnt}/{total} ({cnt/fps:.1f}s)")

                    ann, dets = process_frame_detection_multi(frame, m1, m2, m3)
                    frame_ph.image(cv2.cvtColor(ann, cv2.COLOR_BGR2RGB), use_container_width=True)
                    st.session_state.last_frame = orig

                    for d in dets: st.session_state.detection_history.append(d)
                    st.session_state.detection_history = st.session_state.detection_history[-500:]

                    now_time = time.time()
                    danger_list, rambu_list = handle_alerts(dets, now_time, enable_audio, warn_ph, status_ph, audio_alert_ph, cooldown=alert_cooldown)

                    # FIX: OCR threshold 0.5
                    if st.session_state.ocr_triggered_vid and ocr is not None:
                        st.session_state.ocr_frame_count += 1
                        if st.session_state.ocr_frame_count % ocr_scan_interval == 0:
                            handle_ocr_dedup(orig, ocr, ocr_min_conf, enable_tts, ocr_ph, audio_ocr_ph)

                    m_det.metric("Deteksi", len(dets))
                    m_danger.metric("⚠️ Bahaya", len(danger_list))
                    elapsed = time.time() - start
                    m_fps.metric("FPS", f"{cnt/elapsed:.1f}" if elapsed > 0 else "0")

                    if show_logs:
                        log_ph.markdown(render_log(), unsafe_allow_html=True)
                    
                    # FIX: delay minimal
                    time.sleep(0.01)

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
    mode2 = st.radio("Input:", ["📷 Capture", "📤 Upload"], horizontal=True)
    img_ph2, res_ph2, aud_ph2 = st.empty(), st.empty(), st.empty()

    def run_ocr_image(pil_img):
        arr = np.array(pil_img)
        if len(arr.shape) == 3 and arr.shape[2] >= 3:
            arr = cv2.cvtColor(arr[:, :, :3], cv2.COLOR_RGB2BGR)
        img_ph2.image(pil_img, use_container_width=True)
        text = perform_ocr_on_frame(arr, st.session_state.ocr_engine, ocr_min_conf)
        res_ph2.markdown(f'<div class="ocr-result">📝 {text}</div>', unsafe_allow_html=True)
        if text and text != "Tidak ada teks terdeteksi" and len(text) > 3 and enable_tts:
            a = get_audio_bytes(f"Ada tulisan: {text}")
            if a:
                aud_ph2.empty()
                play_audio_safe(aud_ph2, a)
                st.success("🔊 Audio diputar")

    if mode2 == "📷 Capture":
        cam_ocr = st.camera_input("📸 Ambil foto", key="cam_tab2_ocr")
        if cam_ocr is not None:
            if st.session_state.ocr_engine is None: st.error("⚠️ Load OCR dulu!")
            else:
                from PIL import Image
                run_ocr_image(Image.open(cam_ocr))
    else:
        up_img = st.file_uploader("Upload gambar", type=['jpg','jpeg','png','bmp'])
        if up_img:
            if st.session_state.ocr_engine is None: st.error("⚠️ Load OCR dulu!")
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
        rambu = len([d for d in hist if d['risk_level'] == 'RAMBU'])

        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(f'<div class="stat-box"><div class="stat-value">{total}</div><div class="stat-label">Total</div></div>', unsafe_allow_html=True)
        c2.markdown(f'<div class="stat-box"><div class="stat-value" style="color:#ff4444;">{danger}</div><div class="stat-label">Bahaya</div></div>', unsafe_allow_html=True)
        c3.markdown(f'<div class="stat-box"><div class="stat-value" style="color:#ffaa00;">{warn}</div><div class="stat-label">Waspada</div></div>', unsafe_allow_html=True)
        c4.markdown(f'<div class="stat-box"><div class="stat-value" style="color:#00c073;">{aman + rambu}</div><div class="stat-label">Aman/Rambu</div></div>', unsafe_allow_html=True)

        df = pd.DataFrame([{
            'Waktu': d['timestamp'].strftime("%H:%M:%S"),
            'Objek': get_indo_name(d['class']),
            'Confidence': f"{d['confidence']:.1%}",
            'Risiko': d['risk_level'],
            'Posisi': 'Kiri' if d['position_x'] < 0.35 else ('Kanan' if d['position_x'] > 0.65 else 'Depan'),
        } for d in hist[-100:]])
        st.dataframe(df, use_container_width=True)

        if st.button("🗑️ Hapus Data"):
            st.session_state.detection_history = []
            st.session_state.detected_classes = set()
            st.rerun()
    else:
        st.info("📊 Belum ada data.")

st.divider()
st.markdown("""
<div style="text-align:center; color:#999; font-size:0.8rem; padding:1rem 0;">
    <strong>Asisten Navigasi Tunanetra v5.10</strong> • YOLOv11 • EasyOCR • gTTS
</div>
""", unsafe_allow_html=True)
