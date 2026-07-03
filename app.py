# =====================================================================
# ASISTEN NAVIGASI TUNANETRA - STREAMLIT DASHBOARD v4.3
# =====================================================================
# CHANGELOG v4.3:
# 1. WebRTC live video streaming (bukan take picture)
# 2. Tombol ON/OFF kamera
# 3. OCR improved: multi-pass, better preprocessing, less typo
# 4. Alert bahaya: SEKALI per objek + kasih tau harus ngapain
# 5. Obstacle (lubang/tangga/tiang): conf diturunkan khusus
# 6. detection_history aktif
# 7. conf_threshold diteruskan ke semua mode
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
import threading
import queue
import av

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

.alert-danger {
    background: #fff0f0; border: 1px solid #ffaaaa; border-left: 4px solid #ff4444;
    border-radius: 8px; padding: 1rem; color: #cc2222; font-weight: 600; margin: 0.8rem 0;
}
.alert-info {
    background: #f0f6ff; border: 1px solid #90c8ff; border-left: 4px solid #3f8bff;
    border-radius: 8px; padding: 1rem; color: #0066cc; margin: 0.8rem 0;
}
.alert-warning {
    background: #fffbe6; border: 1px solid #ffe58f; border-left: 4px solid #faad14;
    border-radius: 8px; padding: 1rem; color: #ad6800; font-weight: 600; margin: 0.8rem 0;
}
.alert-success {
    background: #f0fff8; border: 1px solid #80ecc0; border-left: 4px solid #00c073;
    border-radius: 8px; padding: 1rem; color: #00955a; margin: 0.8rem 0;
}

.ocr-result {
    background: #f8fbff; border: 2px solid #3f8bff; border-radius: 10px;
    padding: 1.4rem; color: #0066cc; font-size: 1.05rem; min-height: 60px;
}

.stat-box {
    background: linear-gradient(135deg, #f5f7ff 0%, #faf8ff 100%);
    border-radius: 10px; padding: 1.2rem; text-align: center;
    border: 1px solid #e0e4f0;
}
.stat-value { font-size: 2rem; font-weight: 700; color: #6c3fff; }
.stat-label { font-size: 0.8rem; color: #999; text-transform: uppercase; letter-spacing: 1px; font-weight: 600; }

.stButton > button {
    border-radius: 10px; border: none; font-weight: 600;
    padding: 0.8rem 1.4rem; transition: all 0.2s ease;
    text-transform: uppercase; letter-spacing: 0.5px;
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
    'alerted_objects': set(),  # Track objek yang SUDAH di-alert (sekali saja)
    'ocr_triggered': False, 'ocr_triggered_cam': False,
    'ocr_frame_count': 0,
    'last_uploaded_name': None,
    'last_ocr_text': '',
    'webrtc_result_queue': None,
    'cam_frame_count': 0,
}
for key, default_value in session_defaults.items():
    if key not in st.session_state:
        st.session_state[key] = default_value

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
# TERJEMAHAN & ALERT
# ============================================================
def get_indo_name(name):
    t = {
        'person':'orang', 'car':'mobil', 'bus':'bus', 'truck':'truk',
        'motorcycle':'motor', 'bicycle':'sepeda', 'dog':'anjing', 'cat':'kucing',
        'pothole':'lubang jalan', 'stairs':'tangga', 'obstacle':'rintangan',
        'road-barrier':'pembatas jalan', 'pole':'tiang', 'train':'kereta',
        'stop sign':'rambu stop', 'traffic light':'lampu lalu lintas',
        'sidewalk':'trotoar', 'crosswalk':'jalur penyeberangan', 'tree':'pohon',
        'animal':'hewan', 'vehicle':'kendaraan',
        'dilarang masuk':'dilarang masuk', 'dilarang parkir':'dilarang parkir',
        'dilarang berhenti':'dilarang berhenti', 'hati-hati':'hati-hati',
        'rumah sakit':'rumah sakit', 'masjid':'masjid', 'gereja':'gereja',
        'pom bensin':'pom bensin', 'tempat parkir':'tempat parkir',
        'jalur sepeda':'jalur sepeda', 'batas kecepatan':'batas kecepatan',
        'persimpangan':'persimpangan', 'ikuti arah bundaran':'ikuti arah bundaran',
        # Tambahan obstacle aliases
        'hole':'lubang jalan', 'crack':'retakan jalan', 'bump':'polisi tidur',
        'step':'anak tangga', 'curb':'trotoar tinggi', 'fence':'pagar',
        'wall':'dinding', 'pillar':'pilar', 'bollard':'bollard',
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

def is_obstacle(cls_lower):
    """Cek apakah objek termasuk obstacle/rintangan"""
    obstacle_kw = ['pothole', 'lubang', 'hole', 'stairs', 'tangga', 'step',
                   'obstacle', 'rintangan', 'road-barrier', 'pembatas',
                   'pole', 'tiang', 'pillar', 'bollard', 'crack', 'retakan',
                   'bump', 'polisi tidur', 'curb', 'fence', 'pagar',
                   'wall', 'dinding', 'tree', 'pohon']
    return any(kw in cls_lower for kw in obstacle_kw)

def is_vehicle(cls_lower):
    vehicle_kw = ['car', 'mobil', 'bus', 'truck', 'truk', 'vehicle',
                  'kendaraan', 'motorcycle', 'motor', 'bicycle', 'sepeda',
                  'train', 'kereta']
    return any(kw in cls_lower for kw in vehicle_kw)

def is_person(cls_lower):
    return 'person' in cls_lower or 'orang' in cls_lower

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
        'pom bensin':'Ada pom bensin',
        'tempat parkir':'Ada tempat parkir',
        'jalur sepeda':'Ada jalur sepeda',
        'batas kecepatan':'Ada batas kecepatan',
        'persimpangan':'Ada persimpangan di depan',
    }
    for k, v in suara.items():
        if k in n or n in k: return v
    return f"Ada {get_indo_name(n)}"

def generate_alert_with_action(name, pos_x, area):
    """Generate alert LENGKAP dengan instruksi apa yang harus dilakukan"""
    nama = get_indo_name(name)
    
    # Posisi
    if pos_x < 0.3:
        pos = "di sebelah kiri Anda"
        arah = "Berjalanlah ke kanan untuk menghindarinya"
    elif pos_x > 0.7:
        pos = "di sebelah kanan Anda"
        arah = "Berjalanlah ke kiri untuk menghindarinya"
    else:
        pos = "tepat di depan Anda"
        arah = "Berhenti dan belok ke samping untuk menghindarinya"
    
    # Jarak berdasarkan area
    if area > 0.15:
        jarak = "sangat dekat"
        urgency = "Segera"
    elif area > 0.08:
        jarak = "cukup dekat"
        urgency = "Harap"
    else:
        jarak = "di depan"
        urgency = "Hati-hati"
    
    return f"{urgency}! Ada {nama} {jarak} {pos}. {arah}."

# ============================================================
# DETEKSI FRAME — IMPROVED OBSTACLE DETECTION
# ============================================================
def process_frame_detection(frame, model, conf=0.4, is_obstacle_model=False):
    """
    Deteksi objek pada frame.
    is_obstacle_model=True: turunkan confidence untuk obstacle agar lebih sensitif
    """
    if model is None:
        return frame, []
    try:
        # Untuk model obstacle, turunkan confidence threshold
        effective_conf = max(conf - 0.15, 0.15) if is_obstacle_model else conf
        
        results = model.predict(frame, conf=effective_conf, iou=0.4, verbose=False)
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
                    cls_lower = cls_name.lower()
                    
                    # Filter area minimum — lebih kecil untuk obstacle
                    if is_obstacle(cls_lower):
                        if area_ratio < 0.005: continue  # Obstacle: threshold kecil
                    else:
                        if area_ratio < 0.015: continue
                    
                    pos_x = ((x1 + x2) / 2) / frame_w
                    
                    # Risk level classification
                    if is_obstacle(cls_lower):
                        # OBSTACLE: lebih sensitif
                        if conf_score > 0.20:
                            if area_ratio > 0.06: risk_level = 'BAHAYA'
                            elif area_ratio > 0.02: risk_level = 'WASPADA'
                            else: risk_level = 'AMAN'
                        else:
                            risk_level = 'AMAN'
                    elif is_vehicle(cls_lower):
                        if conf_score > 0.35 and area_ratio > 0.12: risk_level = 'BAHAYA'
                        elif conf_score > 0.30 and area_ratio > 0.06: risk_level = 'WASPADA'
                        else: risk_level = 'AMAN'
                    elif is_person(cls_lower):
                        if conf_score > 0.40 and area_ratio > 0.08: risk_level = 'WASPADA'
                        else: risk_level = 'AMAN'
                    else:
                        risk_level = 'AMAN'
                    
                    detections.append({
                        'class': cls_name, 'confidence': float(conf_score),
                        'area_ratio': area_ratio, 'position_x': pos_x,
                        'risk_level': risk_level, 'bbox': (x1, y1, x2, y2),
                        'timestamp': datetime.now()
                    })
                    
                    # Draw bbox
                    if risk_level == 'BAHAYA': color = (0, 0, 255)
                    elif risk_level == 'WASPADA': color = (0, 165, 255)
                    else: color = (0, 255, 0)
                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    label = f"{get_indo_name(cls_name)} {conf_score:.0%}"
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    fs, th = 0.6, 2
                    (tw, tht), _ = cv2.getTextSize(label, font, fs, th)
                    cv2.rectangle(frame, (x1, y1 - tht - 10), (x1 + tw + 4, y1), color, -1)
                    cv2.putText(frame, label, (x1 + 2, y1 - 5), font, fs, (255, 255, 255), th)
        
        return frame, detections
    except Exception as e:
        logger.error(f"Detection error: {e}")
        return frame, []

def process_frame_detection_multi(frame, model1, model2, model3, conf=0.4):
    """
    Proses frame berurutan pada M1, M2, M3.
    M2 (obstacle) mendapat perlakuan khusus: conf lebih rendah.
    """
    frame_annotated = frame.copy()
    all_detections = []
    
    # M1: General detection (YOLO default)
    if model1 is not None:
        frame_annotated, dets = process_frame_detection(
            frame_annotated, model1, conf, is_obstacle_model=False)
        all_detections.extend(dets)
    
    # M2: Obstacle model — conf diturunkan otomatis
    if model2 is not None:
        frame_annotated, dets = process_frame_detection(
            frame_annotated, model2, conf, is_obstacle_model=True)
        all_detections.extend(dets)
    
    # M3: Rambu model
    if model3 is not None:
        frame_annotated, dets = process_frame_detection(
            frame_annotated, model3, conf, is_obstacle_model=False)
        all_detections.extend(dets)
    
    return frame_annotated, all_detections

# ============================================================
# OCR — IMPROVED MULTI-PASS
# ============================================================
def perform_ocr_on_frame(frame, ocr_engine, min_conf=0.20):
    """OCR dengan multi-pass preprocessing untuk mengurangi typo"""
    if ocr_engine is None:
        return "OCR engine tidak tersedia"
    try:
        # Prepare grayscale
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
        
        h, w = gray.shape
        
        # Upscale jika terlalu kecil
        if w < 640:
            scale = 800 / w
            gray = cv2.resize(gray, (800, int(h * scale)), interpolation=cv2.INTER_CUBIC)
        
        # ── PREPROCESSING PIPELINE ──
        # Pass 1: CLAHE + Sharpen
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        kernel_sharp = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        sharpened = cv2.filter2D(enhanced, -1, kernel_sharp)
        
        # Pass 2: Adaptive threshold (good for signs & text on backgrounds)
        adaptive = cv2.adaptiveThreshold(
            enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 15, 8)
        
        # Pass 3: Otsu threshold (good for high contrast text)
        _, otsu = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Try multiple preprocessed images, collect all results
        all_texts = []
        images_to_try = [sharpened, enhanced, adaptive, otsu, frame]
        
        for img in images_to_try:
            try:
                results = ocr_engine.readtext(img, detail=1, paragraph=False)
                if results:
                    for (bbox, text, conf) in results:
                        text = text.strip()
                        if len(text) > 1 and conf > min_conf:
                            all_texts.append((text, conf))
            except:
                continue
            # Stop jika sudah dapat hasil yang baik
            if any(conf > 0.5 for _, conf in all_texts):
                break
        
        if all_texts:
            # Sort by confidence, ambil yang terbaik
            all_texts.sort(key=lambda x: x[1], reverse=True)
            
            # Deduplicate: ambil teks unik dengan conf tertinggi
            seen = set()
            unique_texts = []
            for text, conf in all_texts:
                # Clean text — GENTLE, keep Unicode
                cleaned = re.sub(r'[^\w\s\.,!?\-:;/()@#%&]', '', text, flags=re.UNICODE)
                cleaned = cleaned.strip()
                norm = cleaned.lower()
                if norm not in seen and len(cleaned) > 1:
                    seen.add(norm)
                    unique_texts.append(cleaned)
            
            if unique_texts:
                result_text = ' '.join(unique_texts)
                result_text = ' '.join(result_text.split())  # Remove extra spaces
                if len(result_text) > 2:
                    return result_text
        
        return "Tidak ada teks terdeteksi"
    except Exception as e:
        return f"Error: {str(e)[:30]}"

def normalize_ocr_text(text):
    if not text: return ""
    return ' '.join(text.split()).lower().strip()

# ============================================================
# ALERT HANDLER — SEKALI PER OBJEK + INSTRUKSI
# ============================================================
def handle_alerts_once(dets, now_time, enable_audio, alert_cooldown,
                       warn_ph, status_ph, audio_danger_ph, audio_rambu_ph):
    """
    Alert system:
    - BAHAYA: alert SEKALI per jenis objek + instruksi lengkap
    - WASPADA: alert dengan cooldown
    - RAMBU: alert dengan cooldown
    - Setelah objek hilang dari frame, reset agar bisa alert lagi
    """
    danger = [d for d in dets if d['risk_level'] == 'BAHAYA']
    waspada = [d for d in dets if d['risk_level'] == 'WASPADA']
    rambu_dets = [d for d in dets if is_rambu(d['class'])]
    
    # Track objek yang sekarang terlihat
    current_objects = set()
    for d in dets:
        current_objects.add(d['class'])
    
    # Reset alert untuk objek yang sudah HILANG dari frame
    gone_objects = st.session_state.alerted_objects - current_objects
    for obj in gone_objects:
        st.session_state.alerted_objects.discard(obj)
    
    # ── PRIORITAS 1: BAHAYA ──
    if danger and enable_audio:
        d = danger[0]
        key = d['class']
        
        # Alert SEKALI per objek (selama masih terlihat, tidak diulang)
        if key not in st.session_state.alerted_objects:
            msg = generate_alert_with_action(d['class'], d['position_x'], d['area_ratio'])
            warn_ph.markdown(f'<div class="alert-danger">🚨 {msg}</div>',
                             unsafe_allow_html=True)
            audio = get_audio_bytes(msg)
            if audio:
                audio_danger_ph.empty()
                play_audio_safe(audio_danger_ph, audio)
            st.session_state.alerted_objects.add(key)
            add_log(f"BAHAYA: {get_indo_name(key)}")
        else:
            # Masih terlihat, tampilkan warning tapi tanpa audio
            nama = get_indo_name(d['class'])
            warn_ph.markdown(
                f'<div class="alert-danger">🚨 Masih ada {nama} di depan. Tetap waspada!</div>',
                unsafe_allow_html=True)
        
        status_ph.markdown(
            '<div class="pills"><span class="pill pill-run">● AKTIF</span>'
            '<span class="pill pill-danger">● BAHAYA</span></div>',
            unsafe_allow_html=True)
    
    # ── PRIORITAS 2: WASPADA ──
    elif waspada and enable_audio:
        d = waspada[0]
        key = f"waspada_{d['class']}"
        if (now_time - st.session_state.last_alert_time[key]) >= alert_cooldown:
            msg = generate_alert_with_action(d['class'], d['position_x'], d['area_ratio'])
            warn_ph.markdown(f'<div class="alert-warning">⚡ {msg}</div>',
                             unsafe_allow_html=True)
            audio = get_audio_bytes(msg)
            if audio:
                audio_danger_ph.empty()
                play_audio_safe(audio_danger_ph, audio)
            st.session_state.last_alert_time[key] = now_time
            add_log(f"WASPADA: {get_indo_name(d['class'])}")
        status_ph.markdown(
            '<div class="pills"><span class="pill pill-run">● AKTIF</span>'
            '<span class="pill pill-danger" style="background:#fffbe6;color:#ad6800;border-color:#ffe58f;">● WASPADA</span></div>',
            unsafe_allow_html=True)
    
    # ── PRIORITAS 3: RAMBU ──
    elif rambu_dets and enable_audio:
        d = rambu_dets[0]
        key = f"rambu_{d['class']}"
        if (now_time - st.session_state.last_alert_time[key]) >= alert_cooldown:
            msg = generate_rambu_alert(d['class'])
            warn_ph.markdown(f'<div class="alert-info">ℹ️ {msg}</div>',
                             unsafe_allow_html=True)
            audio = get_audio_bytes(msg)
            if audio:
                audio_rambu_ph.empty()
                play_audio_safe(audio_rambu_ph, audio)
            st.session_state.last_alert_time[key] = now_time
            add_log(f"RAMBU: {msg}")
        status_ph.markdown(
            '<div class="pills"><span class="pill pill-run">● AKTIF</span>'
            '<span class="pill pill-ocr">● RAMBU</span></div>',
            unsafe_allow_html=True)
    
    # ── AMAN ──
    else:
        warn_ph.markdown('<div class="alert-success">✅ Jalur aman. Silakan jalan terus.</div>',
                         unsafe_allow_html=True)
        status_ph.markdown(
            '<div class="pills"><span class="pill pill-run">● AKTIF</span>'
            '<span class="pill pill-ok">● AMAN</span></div>',
            unsafe_allow_html=True)
    
    return danger, rambu_dets

def handle_ocr(frame, ocr_engine, ocr_min_conf, enable_tts, ocr_ph, audio_ocr_ph):
    """Handle OCR — sekali per teks unik"""
    text = perform_ocr_on_frame(frame, ocr_engine, ocr_min_conf)
    if text and text != "Tidak ada teks terdeteksi" and len(text) > 5:
        norm_text = normalize_ocr_text(text)
        norm_last = normalize_ocr_text(st.session_state.last_ocr_text)
        
        if norm_text != norm_last:
            st.session_state.last_ocr_text = text
            ocr_ph.markdown(f'<div class="ocr-result">📝 {text}</div>',
                            unsafe_allow_html=True)
            add_log(f"OCR: {text[:40]}...")
            if enable_tts:
                audio = get_audio_bytes(f"Ada tulisan: {text}")
                if audio:
                    audio_ocr_ph.empty()
                    play_audio_safe(audio_ocr_ph, audio)
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
        models['m2'] = YOLO(path2) if path2 else None
        models['m3'] = YOLO(path3) if path3 else None
        return models
    except:
        return {}

@st.cache_resource(show_spinner=False)
def load_ocr():
    try:
        import easyocr
        return easyocr.Reader(['id', 'en'], gpu=False)
    except:
        return None

# ─────────────────────────────────────────────────────────────────────
# WEBRTC VIDEO PROCESSOR
# ─────────────────────────────────────────────────────────────────────
from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoProcessorBase

class YOLOVideoProcessor(VideoProcessorBase):
    """
    Process video frames dalam WebRTC callback.
    Model dan settings disimpan sebagai class attributes.
    """
    def __init__(self):
        self.model1 = None
        self.model2 = None
        self.model3 = None
        self.conf = 0.4
        self.result_queue = queue.Queue(maxsize=5)
        self.frame_count = 0
    
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.resize(img, (640, 480))
        self.frame_count += 1
        
        # Deteksi setiap frame
        if self.model1 is not None or self.model2 is not None or self.model3 is not None:
            img_annotated, dets = process_frame_detection_multi(
                img, self.model1, self.model2, self.model3, self.conf)
            
            # Kirim hasil deteksi ke main thread (non-blocking)
            try:
                self.result_queue.put_nowait({
                    'detections': dets,
                    'frame_count': self.frame_count,
                    'timestamp': time.time()
                })
            except queue.Full:
                pass  # Skip jika queue penuh
            
            return av.VideoFrame.from_ndarray(img_annotated, format="bgr24")
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")

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
        <p>Deteksi Objek + Rambu + Baca Teks — Real-time</p>
    </div>
    """, unsafe_allow_html=True)
st.divider()

# ─────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Pengaturan")
    
    if st.button("📥 Load YOLO", use_container_width=True):
        with st.spinner("Loading YOLO..."):
            models = load_yolo_models()
            st.session_state.model1 = models.get('m1')
            add_log("YOLO loaded")
            st.success("✅ YOLO Dimuat!")
    
    if st.button("📥 Load OCR", use_container_width=True):
        with st.spinner("Loading OCR..."):
            st.session_state.ocr_engine = load_ocr()
            add_log("OCR loaded")
            st.success("✅ OCR Dimuat!")
    
    st.markdown("---")
    st.info("💡 Upload file model (.pt) bisa memakan waktu 1-5 menit.")
    
    with st.expander("🏔️ Model 2: Tangga/Lubang/Rintangan"):
        up2 = st.file_uploader("Upload best.pt (M2)", type=['pt'], key='m2up')
        if up2 and st.button("Load M2"):
            with st.spinner("Memuat Model 2 (Obstacle)..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as tmp:
                    tmp.write(up2.read())
                from ultralytics import YOLO
                st.session_state.model2 = YOLO(tmp.name)
                add_log("M2 (obstacle) loaded")
                st.success("✅ M2 (Tangga/Lubang) Dimuat!")
    
    with st.expander("🚦 Model 3: Rambu"):
        up3 = st.file_uploader("Upload best_rambu.pt (M3)", type=['pt'], key='m3up')
        if up3 and st.button("Load M3"):
            with st.spinner("Memuat Model 3 (Rambu)..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as tmp:
                    tmp.write(up3.read())
                from ultralytics import YOLO
                st.session_state.model3 = YOLO(tmp.name)
                add_log("M3 (rambu) loaded")
                st.success("✅ M3 (Rambu) Dimuat!")
    
    st.markdown("---")
    s1 = "✅" if st.session_state.model1 else "⚠️"
    s2 = "✅" if st.session_state.model2 else "⚠️"
    s3 = "✅" if st.session_state.model3 else "⚠️"
    so = "✅" if st.session_state.ocr_engine else "⚠️"
    st.markdown(
        f'<div class="pills">'
        f'<span class="pill pill-model">{s1} M1 Umum</span>'
        f'<span class="pill pill-model">{s2} M2 Rintangan</span>'
        f'<span class="pill pill-model">{s3} M3 Rambu</span>'
        f'<span class="pill pill-model">{so} OCR</span>'
        f'</div>', unsafe_allow_html=True)
    
    conf_threshold = st.slider("Confidence Deteksi", 0.1, 0.9, 0.35, 0.05,
                               help="Untuk rintangan, confidence otomatis diturunkan 15%")
    enable_audio = st.checkbox("🔊 Audio Alert", value=True)
    alert_cooldown = st.slider("Cooldown Alert (s)", 2, 10, 5)
    ocr_min_conf = st.slider("OCR Confidence", 0.1, 0.9, 0.15, 0.05)
    ocr_scan_interval = st.slider("OCR Interval (frame)", 1, 10, 3)
    enable_tts = st.checkbox("🔊 TTS Baca Teks", value=True)
    show_logs = st.checkbox("📋 Show Logs", value=True)

# ─────────────────────────────────────────────────────────────────────
# MAIN TABS
# ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🎯 Detection", "📖 Text Reading", "📊 Statistics"])

# ═════════════════════════════════════════════════════════════════════
# TAB 1: DETECTION
# ═════════════════════════════════════════════════════════════════════
with tab1:
    mode = st.radio("Mode:", ["📹 Kamera Live", "📤 Upload Video"], horizontal=True)
    st.divider()

    # ============================================================
    # MODE KAMERA LIVE — WebRTC Streaming
    # ============================================================
    if mode == "📹 Kamera Live":
        st.markdown("""
        <div class="alert-info">
        📹 <strong>Kamera Live:</strong> Klik START untuk menyalakan kamera.
        Klik STOP untuk mematikan. Video dan deteksi berjalan real-time.
        </div>
        """, unsafe_allow_html=True)
        
        ocr_cam_on = st.checkbox("📖 Aktifkan Baca Teks (OCR) saat streaming", value=False)
        if ocr_cam_on:
            st.session_state.ocr_triggered_cam = True
            st.session_state.last_ocr_text = ''
        else:
            st.session_state.ocr_triggered_cam = False
        
        # Placeholders for alerts & metrics
        status_cam = st.empty()
        warn_cam = st.empty()
        ocr_cam_ph = st.empty()
        audio_danger_cam = st.empty()
        audio_rambu_cam = st.empty()
        audio_ocr_cam = st.empty()
        
        c1, c2, c3 = st.columns(3)
        with c1: m_det_cam = st.empty()
        with c2: m_danger_cam = st.empty()
        with c3: m_status_cam = st.empty()
        
        if show_logs:
            log_cam = st.expander("📋 Logs").empty()
        
        # WebRTC Streamer
        ctx = webrtc_streamer(
            key="blind-assist-cam",
            mode=WebRtcMode.SENDRECV,
            media_stream_constraints={"video": True, "audio": False},
            video_processor_factory=YOLOVideoProcessor,
            async_processing=True,
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        )
        
        # Set models ke processor saat aktif
        if ctx.video_processor:
            ctx.video_processor.model1 = st.session_state.model1
            ctx.video_processor.model2 = st.session_state.model2
            ctx.video_processor.model3 = st.session_state.model3
            ctx.video_processor.conf = conf_threshold
            
            # Polling results dari processor
            if ctx.state.playing:
                m_status_cam.metric("Status", "🟢 Live")
                
                # Poll detection results
                result_placeholder = st.empty()
                while ctx.state.playing:
                    try:
                        result = ctx.video_processor.result_queue.get(timeout=1.0)
                        dets = result['detections']
                        
                        # Simpan ke history
                        for d in dets:
                            st.session_state.detection_history.append(d)
                        st.session_state.detection_history = \
                            st.session_state.detection_history[-500:]
                        
                        # Handle alerts
                        now_time = time.time()
                        danger_list, rambu_list = handle_alerts_once(
                            dets, now_time, enable_audio, alert_cooldown,
                            warn_cam, status_cam, audio_danger_cam, audio_rambu_cam)
                        
                        # Metrics
                        m_det_cam.metric("Deteksi", len(dets))
                        m_danger_cam.metric("⚠️ Bahaya", len(danger_list))
                        
                        # OCR (setiap N frame)
                        if st.session_state.ocr_triggered_cam:
                            st.session_state.cam_frame_count += 1
                            if st.session_state.cam_frame_count % ocr_scan_interval == 0:
                                if st.session_state.last_frame is not None:
                                    handle_ocr(
                                        st.session_state.last_frame,
                                        st.session_state.ocr_engine,
                                        ocr_min_conf, enable_tts,
                                        ocr_cam_ph, audio_ocr_cam)
                        
                        if show_logs:
                            log_cam.markdown(
                                '<br>'.join([f'[{ts}] {msg}'
                                             for ts, msg in st.session_state.log[:10]]),
                                unsafe_allow_html=True)
                    except queue.Empty:
                        continue
            else:
                m_status_cam.metric("Status", "⚪ Stopped")
                st.session_state.alerted_objects.clear()
        else:
            if st.session_state.model1 is None:
                st.warning("⚠️ Load YOLO dulu di sidebar sebelum start kamera!")

    # ============================================================
    # MODE UPLOAD VIDEO
    # ============================================================
    else:
        uploaded = st.file_uploader("Upload video", type=['mp4', 'avi', 'mov', 'mkv'])
        
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
        
        if show_logs:
            log_ph = st.expander("📋 Logs").empty()
        
        # Reset OCR saat video baru
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
        with col1:
            btn_start = st.button("▶️ Start Detection", use_container_width=True)
        with col2:
            btn_baca = st.button("📖 Baca Teks", use_container_width=True)
        
        if btn_baca:
            st.session_state.ocr_triggered = True
            st.session_state.ocr_frame_count = 0
            st.session_state.last_ocr_text = ''
            st.info("🔍 OCR aktif — teks akan dibaca sepanjang video.")
        
        if uploaded and btn_start:
            if st.session_state.model1 is None and st.session_state.model2 is None:
                st.error("⚠️ Load minimal satu model YOLO di sidebar!")
                st.stop()
            
            # Reset alerted objects
            st.session_state.alerted_objects.clear()
            
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
                tmp.write(uploaded.read())
                vid_path = tmp.name
            
            try:
                cap = cv2.VideoCapture(vid_path)
                total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                
                m1 = st.session_state.model1
                m2 = st.session_state.model2
                m3 = st.session_state.model3
                ocr = st.session_state.ocr_engine
                
                cnt, start = 0, time.time()
                prog = st.progress(0)
                
                while True:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    cnt += 1
                    frame = cv2.resize(frame, (640, 480))
                    orig = frame.copy()
                    now_sec = cnt / fps
                    
                    prog.progress(min(cnt / max(total, 1), 1.0),
                                  text=f"Frame {cnt}/{total} ({now_sec:.1f}s)")
                    
                    # DETEKSI — conf_threshold dari slider
                    frame_ann, dets = process_frame_detection_multi(
                        frame, m1, m2, m3, conf_threshold)
                    
                    frame_ph.image(cv2.cvtColor(frame_ann, cv2.COLOR_BGR2RGB),
                                   use_container_width=True)
                    st.session_state.last_frame = orig
                    
                    # Simpan detection history
                    for d in dets:
                        st.session_state.detection_history.append(d)
                    st.session_state.detection_history = \
                        st.session_state.detection_history[-500:]
                    
                    # ALERT — balanced, sekali per objek
                    now_time = time.time()
                    danger_list, rambu_list = handle_alerts_once(
                        dets, now_time, enable_audio, alert_cooldown,
                        warn_ph, status_ph, audio_danger_ph, audio_rambu_ph)
                    
                    # OCR — setiap N frame
                    if st.session_state.ocr_triggered and ocr is not None:
                        st.session_state.ocr_frame_count += 1
                        if st.session_state.ocr_frame_count % ocr_scan_interval == 0:
                            handle_ocr(orig, ocr, ocr_min_conf, enable_tts,
                                       ocr_ph, audio_ocr_ph)
                    
                    # Metrics
                    m_det.metric("Deteksi", len(dets))
                    m_danger.metric("⚠️ Bahaya", len(danger_list))
                    elapsed = time.time() - start
                    m_fps.metric("FPS", f"{cnt/elapsed:.1f}" if elapsed > 0 else "0")
                    
                    if show_logs:
                        log_ph.markdown(
                            '<br>'.join([f'[{ts}] {msg}'
                                         for ts, msg in st.session_state.log[:10]]),
                            unsafe_allow_html=True)
                    
                    time.sleep(0.001)
                
                cap.release()
                prog.empty()
                st.success(f"✅ Selesai! Total: {cnt} frame")
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
                st.error("⚠️ Load OCR dulu di sidebar!")
            else:
                from PIL import Image
                img = Image.open(cam_ocr)
                arr = np.array(img)
                img_ph2.image(img, use_container_width=True)
                text = perform_ocr_on_frame(arr, st.session_state.ocr_engine, ocr_min_conf)
                res_ph2.markdown(f'<div class="ocr-result">📝 {text}</div>',
                                 unsafe_allow_html=True)
                if text and text != "Tidak ada teks terdeteksi" and len(text) > 5 and enable_tts:
                    audio = get_audio_bytes(f"Ada tulisan: {text}")
                    if audio:
                        aud_ph2.empty()
                        play_audio_safe(aud_ph2, audio)
                        st.success("🔊 Audio diputar")
    else:
        up_img = st.file_uploader("Upload gambar", type=['jpg', 'jpeg', 'png', 'bmp'])
        if up_img:
            from PIL import Image
            img = Image.open(up_img)
            arr = np.array(img)
            img_ph2.image(img, use_container_width=True)
            if st.session_state.ocr_engine is None:
                st.error("⚠️ Load OCR dulu di sidebar!")
            else:
                text = perform_ocr_on_frame(arr, st.session_state.ocr_engine, ocr_min_conf)
                res_ph2.markdown(f'<div class="ocr-result">📝 {text}</div>',
                                 unsafe_allow_html=True)
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
        c1.markdown(
            f'<div class="stat-box"><div class="stat-value">{total}</div>'
            f'<div class="stat-label">Total</div></div>', unsafe_allow_html=True)
        c2.markdown(
            f'<div class="stat-box"><div class="stat-value" style="color:#ff4444;">'
            f'{danger}</div><div class="stat-label">Bahaya</div></div>', unsafe_allow_html=True)
        c3.markdown(
            f'<div class="stat-box"><div class="stat-value" style="color:#faad14;">'
            f'{warn}</div><div class="stat-label">Waspada</div></div>', unsafe_allow_html=True)
        c4.markdown(
            f'<div class="stat-box"><div class="stat-value" style="color:#00c073;">'
            f'{aman}</div><div class="stat-label">Aman</div></div>', unsafe_allow_html=True)
        
        # Detail tabel
        df = pd.DataFrame([{
            'Waktu': d['timestamp'].strftime("%H:%M:%S"),
            'Objek': get_indo_name(d['class']),
            'Confidence': f"{d['confidence']:.0%}",
            'Risiko': d['risk_level'],
            'Posisi': 'Kiri' if d['position_x'] < 0.3
                      else ('Kanan' if d['position_x'] > 0.7 else 'Depan'),
        } for d in hist[-100:]])
        st.dataframe(df, use_container_width=True)
        
        # Ringkasan per objek
        st.markdown("#### 📈 Ringkasan per Objek")
        class_counts = defaultdict(lambda: {'total': 0, 'bahaya': 0, 'waspada': 0})
        for d in hist:
            name = get_indo_name(d['class'])
            class_counts[name]['total'] += 1
            if d['risk_level'] == 'BAHAYA': class_counts[name]['bahaya'] += 1
            elif d['risk_level'] == 'WASPADA': class_counts[name]['waspada'] += 1
        
        summary_df = pd.DataFrame([
            {'Objek': name, 'Total': v['total'],
             'Bahaya': v['bahaya'], 'Waspada': v['waspada']}
            for name, v in sorted(class_counts.items(),
                                  key=lambda x: x[1]['total'], reverse=True)
        ])
        st.dataframe(summary_df, use_container_width=True)
        
        if st.button("🗑️ Hapus Semua Data"):
            st.session_state.detection_history = []
            st.session_state.alerted_objects.clear()
            st.rerun()
    else:
        st.info("📊 Belum ada data. Mulai deteksi di tab Detection.")

# ─────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────
st.divider()
st.markdown("""
<div style="text-align:center; color:#999; font-size:0.8rem; padding:1rem 0;">
    <strong>Asisten Navigasi Tunanetra v4.3</strong> • YOLOv11 • EasyOCR • gTTS • WebRTC
</div>
""", unsafe_allow_html=True)
