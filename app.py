# =====================================================================
# ASISTEN NAVIGASI TUNANETRA v7.1 (CLOUD / STREAMLIT SHARE EDITION)
# FIX: Kamera WebRTC Anti-Crash & Anti-Macet
# =====================================================================

import streamlit as st
import cv2, os, re, time, base64, tempfile, queue
import numpy as np
import pandas as pd
from gtts import gTTS
from datetime import datetime
from io import BytesIO
from collections import defaultdict
import logging
import random

try:
    from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
    import av
    WEBRTC_OK = True
except ImportError:
    WEBRTC_OK = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(page_title="Asisten Tunanetra", page_icon="👁️", layout="wide", initial_sidebar_state="collapsed")

# ─────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
html, body, [data-testid="stAppViewContainer"] { background: linear-gradient(135deg, #f5f7ff 0%, #faf8ff 100%) !important; font-family: 'Inter', sans-serif; }
.header-logo { width:50px;height:50px;background:linear-gradient(135deg,#6c3fff,#3f8bff);border-radius:14px;display:flex;align-items:center;justify-content:center;font-size:1.4rem;box-shadow:0 4px 15px rgba(108,63,255,.3); color:white;}
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
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────────────
session_defaults = {
    'model1': None, 'model2': None, 'model3': None, 'ocr_engine': None,
    'log': [], 'detection_history': [],
    'last_alert_time': defaultdict(lambda: -99.0),
    'ocr_triggered_vid': False,
    'ocr_frame_count': 0, 'last_uploaded_name': None,
    'last_ocr_text': '', 'last_ocr_time': 0.0,
}
for key, value in session_defaults.items():
    if key not in st.session_state: st.session_state[key] = value

def add_log(msg):
    st.session_state.log.insert(0, (time.strftime("%H:%M:%S"), msg))
    st.session_state.log = st.session_state.log[:30]

# ─────────────────────────────────────────────────────────────────────
# AUDIO & TTS
# ─────────────────────────────────────────────────────────────────────
def play_audio_safe(placeholder, audio_bytes):
    if not audio_bytes: return False
    b64 = base64.b64encode(audio_bytes).decode()
    uid = f"a_{int(time.time()*1000)}_{random.randint(1000,9999)}"
    placeholder.markdown(f"""
        <audio id="{uid}" autoplay>
            <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
        </audio>
        <script>
        (function(){{
            var el=document.getElementById("{uid}");
            if(el){{
                var p=el.play();
                if(p!==undefined){{
                    p.catch(function(e){{
                        document.addEventListener('click',function handler(){{
                            el.play(); document.removeEventListener('click',handler);
                        }});
                    }});
                }}
            }}
        }})();
        </script>
    """, unsafe_allow_html=True)
    return True

def get_audio_bytes(text, lang='id'):
    try:
        buf=BytesIO()
        gTTS(text=text,lang=lang,slow=False).write_to_fp(buf)
        buf.seek(0); return buf.read()
    except Exception: return None

# ─────────────────────────────────────────────────────────────────────
# TRANSLATION & ALERT MESSAGES
# ─────────────────────────────────────────────────────────────────────
INDO_NAMES = {
    'person':'orang', 'car':'mobil', 'bus':'bus', 'truck':'truk',
    'motorcycle':'motor', 'bicycle':'sepeda', 'dog':'anjing', 'cat':'kucing',
    'pothole':'lubang jalan', 'stairs':'tangga', 'obstacle':'rintangan',
    'road-barrier':'pembatas jalan', 'pole':'tiang', 'train':'kereta',
    'stop sign':'rambu stop', 'traffic light':'lampu lalu lintas',
    'sidewalk':'trotoar', 'crosswalk':'jalur penyeberangan', 'tree':'pohon',
    'animal':'hewan', 'vehicle':'kendaraan', 'hole':'lubang jalan', 'step':'tangga',
}

def get_indo_name(name):
    n = name.lower()
    if n in INDO_NAMES: return INDO_NAMES[n]
    for k, v in INDO_NAMES.items():
        if k in n or n in k: return v
    return n

RAMBU_KW = {'rambu', 'lampu lalu lintas', 'traffic light', 'stop sign', 'dilarang', 'hati-hati', 'rumah sakit', 'masjid', 'gereja', 'pom bensin', 'parkir', 'persimpangan', 'bundaran', 'jalur', 'kecepatan', 'berhenti', 'masuk'}

def is_rambu(name): return any(k in name.lower() for k in RAMBU_KW)

def generate_rambu_alert(name):
    n = name.lower()
    m = {'stop sign':'Ada rambu stop. Berhenti!', 'rambu stop':'Ada rambu stop. Berhenti!', 'dilarang masuk':'Ada rambu dilarang masuk. Jangan masuk!', 'dilarang parkir':'Ada rambu dilarang parkir', 'dilarang berhenti':'Ada rambu dilarang berhenti', 'hati-hati':'Ada rambu hati-hati. Waspada!'}
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

# ─────────────────────────────────────────────────────────────────────
# OCR - ANTI TYPO & ANTI SPAM
# ─────────────────────────────────────────────────────────────────────
COMMON_WORDS = {'jln':'jalan','dlarang':'dilarang','parkr':'parkir','msuk':'masuk','brhenti':'berhenti','hti-hti':'hati-hati','rmah':'rumah','msjid':'masjid','grja':'gereja','skolah':'sekolah','bhaya':'bahaya','kluar':'keluar','blok':'belok','knan':'kanan','kri':'kiri','dpan':'depan','belkang':'belakang','stp':'stop','lmpu':'lampu','mrah':'merah','hjau':'hijau','kunng':'kuning','pnyebrangan':'penyeberangan','zebr':'zebra','crss':'cross','mtr':'meter','kec':'kecepatan','maks':'maksimal','min':'minimal','spb':'SPBU'}

def autocorrect_word(word):
    if len(word) <= 1: return word
    low = word.lower()
    if low in COMMON_WORDS: return COMMON_WORDS[low]
    best_match, best_sim = None, 0
    for k, v in COMMON_WORDS.items():
        if abs(len(k) - len(low)) > 2: continue
        sim = sum(1 for c in low if c in k) / max(len(k), len(low))
        if sim > 0.75 and sim > best_sim: best_sim, best_match = sim, v
    return best_match if best_match else word

def fix_spelled_text(text):
    if not text: return text
    words = text.split()
    result_words, i = [], 0
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
        else: result_words.append(words[i]); i += 1
    return ' '.join(result_words)

def clean_ocr_text(raw_text):
    if not raw_text or raw_text == "Tidak ada teks terdeteksi": return raw_text
    text = fix_spelled_text(raw_text)
    text = re.sub(r'[^a-zA-Z0-9\s\.,!?\-:/()%]', '', text)
    corrected = [autocorrect_word(w) for w in text.split() if len(w) > 1 or w.isdigit()]
    text = ' '.join(' '.join(corrected).split())
    return text.title() if len(text) >= 3 else "Tidak ada teks terdeteksi"

OCR_ALLOWLIST = ('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 .,-:;/()!?%&@')

def perform_ocr_on_frame(frame, ocr_engine, min_conf=0.35):
    if ocr_engine is None: return "OCR engine tidak tersedia"
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape)==3 else frame
        h, w = gray.shape
        if w < 1000:
            scale = 1000 / w
            gray = cv2.resize(gray, (1000, int(h * scale)), interpolation=cv2.INTER_CUBIC)
        denoised = cv2.fastNlMeansDenoising(gray, None, h=10, templateWindowSize=7, searchWindowSize=21)
        enhanced = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(denoised)

        results = None
        try: results = ocr_engine.readtext(enhanced, detail=1, paragraph=False, allowlist=OCR_ALLOWLIST, text_threshold=0.6, low_text=0.35, contrast_ths=0.1, adjust_contrast=0.7)
        except: pass
        if not results:
            try: results = ocr_engine.readtext(gray, detail=1, paragraph=False, allowlist=OCR_ALLOWLIST)
            except: pass

        if results:
            texts = [text.strip() for (bbox, text, conf) in results if len(text.strip()) >= 2 and conf >= min_conf]
            if texts: return clean_ocr_text(' '.join(' '.join(texts).split()))
        return "Tidak ada teks terdeteksi"
    except: return "Error OCR"

def texts_are_similar(t1, t2, threshold=0.75):
    if not t1 or not t2: return False
    a, b = ' '.join(t1.lower().split()), ' '.join(t2.lower().split())
    if a == b: return True
    if len(a) < 3 or len(b) < 3: return a == b
    s1, s2 = set(a[i:i+2] for i in range(len(a)-1)), set(b[i:i+2] for i in range(len(b)-1))
    if not s1 or not s2: return a == b
    return len(s1 & s2) / len(s1 | s2) >= threshold

def handle_ocr_dedup(frame, ocr_eng, min_conf, en_tts, ocr_ph, aocr_ph, now_time):
    text = perform_ocr_on_frame(frame, ocr_eng, min_conf)
    if (now_time - st.session_state.last_ocr_time) > 6.0:
        st.session_state.last_ocr_text = '' 
        
    if text and text != "Tidak ada teks terdeteksi" and len(text) > 3:
        st.session_state.last_ocr_time = now_time
        if not texts_are_similar(text, st.session_state.last_ocr_text):
            st.session_state.last_ocr_text = text
            ocr_ph.markdown(f'<div class="ocr-result">📝 {text}</div>', unsafe_allow_html=True)
            if en_tts:
                a = get_audio_bytes(f"Ada tulisan: {text}")
                if a: play_audio_safe(aocr_ph, a)
            return True
        else:
            ocr_ph.markdown(f'<div class="ocr-result" style="opacity:0.6;">📝 {text}<br><small>(Sudah dibaca)</small></div>', unsafe_allow_html=True)
    return False

# ─────────────────────────────────────────────────────────────────────
# DETEKSI YOLO - LUBANG & ORANG FIX
# ─────────────────────────────────────────────────────────────────────
OBSTACLE_KW = ['pothole', 'lubang', 'hole', 'stairs', 'stair', 'step', 'tangga', 'obstacle', 'rintangan', 'road-barrier', 'barrier', 'pembatas', 'pole', 'tiang']
VEHICLE_KW = ['car', 'mobil', 'bus', 'truck', 'truk', 'vehicle', 'kendaraan', 'motorcycle', 'motor', 'bicycle', 'sepeda', 'train', 'kereta']
PERSON_KW = ['person', 'orang']

def process_frame_detection(frame, model, conf=0.4, source='m1'):
    if model is None: return frame, []
    try:
        effective_conf = max(conf - 0.20, 0.10) if source == 'm2' else conf
        results = model.predict(frame, conf=effective_conf, iou=0.45, verbose=False)
        detections = []
        
        if results and len(results) > 0 and results[0].boxes is not None:
            r = results[0]
            boxes, confidences, classes = r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(), r.boxes.cls.cpu().numpy().astype(int)
            fh, fw = frame.shape[:2]

            for box, conf_score, cls_idx in zip(boxes, confidences, classes):
                x1, y1, x2, y2 = map(int, box)
                cls_name = r.names[cls_idx]
                cls_lower = cls_name.lower()

                is_obstacle = any(k in cls_lower for k in OBSTACLE_KW)
                is_vehicle = any(k in cls_lower for k in VEHICLE_KW)
                is_person = any(k in cls_lower for k in PERSON_KW)

                area = (x2 - x1) * (y2 - y1)
                area_ratio = area / (fw * fh)

                min_area = 0.002 if source in ['m2', 'm3'] else 0.015
                if area_ratio < min_area: continue

                pos_x = ((x1 + x2) / 2) / fw

                if is_obstacle:
                    if area_ratio > 0.04: risk_level = 'BAHAYA'
                    elif area_ratio > 0.01: risk_level = 'WASPADA'
                    else: risk_level = 'AMAN'
                elif is_vehicle:
                    if conf_score > 0.45 and area_ratio > 0.15: risk_level = 'BAHAYA'
                    elif conf_score > 0.35 and area_ratio > 0.05: risk_level = 'WASPADA'
                    else: risk_level = 'AMAN'
                elif is_person:
                    if conf_score > 0.40 and area_ratio > 0.10: risk_level = 'WASPADA'
                    else: risk_level = 'AMAN'
                elif source == 'm3': risk_level = 'RAMBU'
                else: risk_level = 'AMAN'

                detections.append({'class': cls_name, 'confidence': float(conf_score), 'area_ratio': area_ratio, 'position_x': pos_x, 'risk_level': risk_level, 'bbox': (x1, y1, x2, y2), 'timestamp': datetime.now()})

                color = (0, 0, 255) if risk_level == 'BAHAYA' else (0, 165, 255) if risk_level == 'WASPADA' else (255, 165, 0) if risk_level == 'RAMBU' else (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                lbl = f"{get_indo_name(cls_name)} {conf_score:.2f}"
                (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
                cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
                cv2.putText(frame, lbl, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

        return frame, detections
    except: return frame, []

def process_frame_detection_multi(frame, m1, m2, m3, conf=0.4):
    out, all_d = frame.copy(), []
    if m1: out, d = process_frame_detection(out, m1, conf, source='m1'); all_d.extend(d)
    if m2: out, d = process_frame_detection(out, m2, conf, source='m2'); all_d.extend(d)
    if m3: out, d = process_frame_detection(out, m3, conf, source='m3'); all_d.extend(d)
    return out, all_d

def handle_alerts_once(dets, now_time, enable_audio, warn_ph, status_ph, audio_ph, cooldown=3):
    danger = sorted([d for d in dets if d['risk_level'] == 'BAHAYA'], key=lambda x:x['area_ratio'], reverse=True)
    waspada = sorted([d for d in dets if d['risk_level'] == 'WASPADA'], key=lambda x:x['area_ratio'], reverse=True)
    rambu = sorted([d for d in dets if d['risk_level'] == 'RAMBU'], key=lambda x:x['area_ratio'], reverse=True)

    if danger:
        d = danger[0]
        cls_key = f"danger_{d['class']}"
        if (now_time - st.session_state.last_alert_time[cls_key]) > cooldown:
            msg = generate_alert(d['class'], d['position_x'], d['area_ratio'])
            warn_ph.markdown(f'<div class="alert-danger">⚠️ {msg}</div>', unsafe_allow_html=True)
            if enable_audio:
                a = get_audio_bytes(msg)
                if a: play_audio_safe(audio_ph, a)
            st.session_state.last_alert_time[cls_key] = now_time
        status_ph.markdown('<div class="pills"><span class="pill pill-run">● YOLO</span><span class="pill pill-danger">● BAHAYA</span></div>', unsafe_allow_html=True)

    elif waspada:
        d = waspada[0]
        cls_key = f"waspada_{d['class']}"
        if (now_time - st.session_state.last_alert_time[cls_key]) > cooldown:
            msg = generate_alert(d['class'], d['position_x'], d['area_ratio'])
            warn_ph.markdown(f'<div class="alert-warning">⚡ {msg}</div>', unsafe_allow_html=True)
            if enable_audio:
                a = get_audio_bytes(msg)
                if a: play_audio_safe(audio_ph, a)
            st.session_state.last_alert_time[cls_key] = now_time
        status_ph.markdown('<div class="pills"><span class="pill pill-run">● YOLO</span><span class="pill pill-warning" style="background:#fff5e0;color:#cc8800;border-color:#ffcc66;">● WASPADA</span></div>', unsafe_allow_html=True)

    elif rambu:
        d = rambu[0]
        cls_key = f"rambu_{d['class']}"
        if (now_time - st.session_state.last_alert_time[cls_key]) > cooldown:
            msg = generate_rambu_alert(d['class'])
            warn_ph.markdown(f'<div class="alert-info">ℹ️ {msg}</div>', unsafe_allow_html=True)
            if enable_audio:
                a = get_audio_bytes(msg)
                if a: play_audio_safe(audio_ph, a)
            st.session_state.last_alert_time[cls_key] = now_time
        status_ph.markdown('<div class="pills"><span class="pill pill-run">● YOLO</span><span class="pill pill-ocr">● RAMBU</span></div>', unsafe_allow_html=True)

    else:
        warn_ph.markdown('<div class="alert-success">✅ Area Terpantau Aman</div>', unsafe_allow_html=True)
        status_ph.markdown('<div class="pills"><span class="pill pill-run">● YOLO</span><span class="pill pill-ok">● Aman</span></div>', unsafe_allow_html=True)

    return danger, rambu

# ─────────────────────────────────────────────────────────────────────
# WEBRTC PROCESSOR UNTUK CLOUD (ANTI-CRASH)
# ─────────────────────────────────────────────────────────────────────
if WEBRTC_OK:
    class YOLOLiveProcessor(VideoProcessorBase):
        def __init__(self):
            self.model1, self.model2, self.model3 = None, None, None
            self.conf = 0.4
            self.frame_skip = 2
            self._cnt = 0
            self.result_queue = queue.Queue(maxsize=5)
            self._last_ann = None

        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")
            try:
                self._cnt += 1
                h, w = img.shape[:2]
                
                # Optimasi agar kamera tidak berat/mati di Cloud
                scale = 480 / w
                new_w, new_h = 480, int(h * scale)
                img_small = cv2.resize(img, (new_w, new_h))
                
                if self._cnt % self.frame_skip == 0:
                    ann, dets = process_frame_detection_multi(img_small, self.model1, self.model2, self.model3, self.conf)
                    # Kembalikan resolusi sesuai permintaan frame browser asli
                    self._last_ann = cv2.resize(ann, (w, h))
                    try: self.result_queue.put_nowait(dets)
                    except queue.Full: pass
                    out_frame = self._last_ann
                else:
                    out_frame = self._last_ann if self._last_ann is not None else img
                    
                return av.VideoFrame.from_ndarray(out_frame, format="bgr24")
            except Exception as e:
                logger.error(f"WebRTC Error: {e}")
                # Jika crash, kembalikan frame asli agar kamera TIDAK MATI
                return av.VideoFrame.from_ndarray(img, format="bgr24")

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
with c2: st.markdown('<div class="header-text"><h1>Asisten Navigasi Tunanetra</h1><p>Deteksi Objek & Teks Terpadu (Cloud Edition v7.1)</p></div>', unsafe_allow_html=True)
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
    
    conf_threshold = st.slider("Confidence Deteksi Base", 0.1, 0.9, 0.35, 0.05)
    enable_audio = st.checkbox("🔊 Audio Alert", value=True)
    alert_cooldown = st.slider("Waktu Ulang Suara (s)", 2, 10, 4)
    ocr_min_conf = st.slider("OCR Confidence", 0.1, 0.9, 0.30, 0.05)
    ocr_scan_interval = st.slider("OCR Scan Tiap N Frame", 1, 30, 10)
    frame_skip = st.slider("Frame Skip (Biar Ringan)", 1, 10, 2)
    enable_tts = st.checkbox("🔊 TTS Baca Teks", value=True)
    
    if st.button("🔁 Reset Daftar Suara"):
        st.session_state.last_alert_time.clear()
        st.session_state.last_ocr_text = ''
        st.success("Daftar memori suara di-reset!")

# ─────────────────────────────────────────────────────────────────────
# TABS UTAMA
# ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🎯 Detection", "📖 Text Reading", "📊 Statistics"])

with tab1:
    mode = st.radio("Mode:", ["📹 Webcam LIVE (Cloud)", "📤 Upload Video"], horizontal=True)
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

    # ============================================================
    # MODE WEBCAM (WebRTC untuk Cloud - ANTI CRASH)
    # ============================================================
    if mode == "📹 Webcam LIVE (Cloud)":
        if not WEBRTC_OK:
            st.error("⚠️ Library belum terpasang. Tambahkan 'streamlit-webrtc' dan 'av' ke requirements.txt")
        else:
            st.info("💡 Klik START untuk menyalakan kamera. Jika hanya muncul video mentah, berarti model YOLO sedang melakukan loading.")
            ctx = webrtc_streamer(
                key="live-yolo",
                mode=WebRtcMode.SENDRECV,
                video_processor_factory=YOLOLiveProcessor,
                rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
                async_processing=True,
            )

            if ctx.video_processor:
                ctx.video_processor.model1 = st.session_state.model1
                ctx.video_processor.model2 = st.session_state.model2
                ctx.video_processor.model3 = st.session_state.model3
                ctx.video_processor.conf = conf_threshold
                ctx.video_processor.frame_skip = frame_skip

            if ctx.state.playing:
                fps_t0, fps_n = time.time(), 0
                while ctx.state.playing:
                    try: dets = ctx.video_processor.result_queue.get(timeout=1.0)
                    except queue.Empty: continue

                    fps_n += 1
                    now_time = time.time()
                    
                    for d in dets: st.session_state.detection_history.append(d)
                    st.session_state.detection_history = st.session_state.detection_history[-500:]

                    danger_list, rambu_list = handle_alerts_once(dets, now_time, enable_audio, warn_ph, status_ph, audio_alert_ph, alert_cooldown)

                    m_det.metric("Deteksi", len(dets))
                    m_danger.metric("⚠️ Bahaya", len(danger_list))
                    elapsed = now_time - fps_t0
                    m_fps.metric("Deteksi/s", f"{fps_n/elapsed:.1f}" if elapsed > 0 else "0")

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
                st.session_state.last_alert_time.clear()
                ocr_ph.empty()
                st.info("🔄 Video baru diupload. Silakan klik 'Start' atau 'Baca Teks'.")

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

                    video_sec = cnt / fps
                    prog.progress(min(cnt / max(total, 1), 1.0), text=f"Frame {cnt}/{total} ({video_sec:.1f}s)")

                    frame_ann, dets = process_frame_detection_multi(frame, m1, m2, m3, conf_threshold)
                    frame_ph.image(cv2.cvtColor(frame_ann, cv2.COLOR_BGR2RGB), use_container_width=True)
                    st.session_state.last_frame = orig

                    for d in dets: st.session_state.detection_history.append(d)
                    st.session_state.detection_history = st.session_state.detection_history[-500:]

                    # WAKTU VIDEO ASLI agar suara tidak macet
                    danger_list, rambu_list = handle_alerts_once(dets, video_sec, enable_audio, warn_ph, status_ph, audio_alert_ph, alert_cooldown)

                    if st.session_state.ocr_triggered_vid and ocr is not None:
                        st.session_state.ocr_frame_count += 1
                        if st.session_state.ocr_frame_count % ocr_scan_interval == 0:
                            handle_ocr_dedup(orig, ocr, ocr_min_conf, enable_tts, ocr_ph, audio_ocr_ph, video_sec)

                    m_det.metric("Deteksi", len(dets))
                    m_danger.metric("⚠️ Bahaya", len(danger_list))
                    elapsed = time.time() - start
                    m_fps.metric("FPS", f"{cnt/elapsed:.1f}" if elapsed > 0 else "0")
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
        rmb = len([d for d in hist if d['risk_level'] == 'RAMBU'])

        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(f'<div class="stat-box"><div class="stat-value">{total}</div><div class="stat-label">Total</div></div>', unsafe_allow_html=True)
        c2.markdown(f'<div class="stat-box"><div class="stat-value" style="color:#ff4444;">{danger}</div><div class="stat-label">Bahaya</div></div>', unsafe_allow_html=True)
        c3.markdown(f'<div class="stat-box"><div class="stat-value" style="color:#ffaa00;">{warn}</div><div class="stat-label">Waspada</div></div>', unsafe_allow_html=True)
        c4.markdown(f'<div class="stat-box"><div class="stat-value" style="color:#00c073;">{aman+rmb}</div><div class="stat-label">Aman/Rambu</div></div>', unsafe_allow_html=True)

        df = pd.DataFrame([{
            'Waktu': d['timestamp'].strftime("%H:%M:%S"),
            'Objek': get_indo_name(d['class']),
            'Confidence': f"{d['confidence']:.2f}",
            'Risiko': d['risk_level'],
            'Posisi': 'Kiri' if d['position_x'] < 0.35 else ('Kanan' if d['position_x'] > 0.65 else 'Depan'),
        } for d in hist[-100:]])
        st.dataframe(df, use_container_width=True)

        if st.button("🗑️ Hapus Semua Data"):
            st.session_state.detection_history = []
            st.session_state.last_alert_time.clear()
            st.rerun()
    else:
        st.info("📊 Belum ada data deteksi. Mulai deteksi di tab Detection.")

st.divider()
st.markdown("""
<div style="text-align:center; color:#999; font-size:0.8rem; padding:1rem 0;">
    <strong>Asisten Navigasi Tunanetra v7.1 (Cloud Edition)</strong> • YOLOv11 • EasyOCR • gTTS
</div>
""", unsafe_allow_html=True)
