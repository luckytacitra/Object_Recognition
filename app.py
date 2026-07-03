# =====================================================================
# ASISTEN NAVIGASI TUNANETRA - STREAMLIT DASHBOARD v5.1
# =====================================================================
# PERBAIKAN v5.1:
# 1. Video: optimasi ekstrim (resize 320x240, skip frame lebih agresif)
# 2. Deteksi lubang: M2 conf 0.15, area filter 0.005
# 3. Hanya obstacle (lubang/tangga/rintangan) yang dianggap BAHAYA
# 4. Orang/kendaraan = WASPADA (bukan BAHAYA)
# 5. OCR: anti-typo + anti-mengeja + autocorrect total
# 6. Alert: sekali per objek, tidak berulang
# =====================================================================

import streamlit as st
import cv2, os, re, time, base64, tempfile
import numpy as np
import pandas as pd
from gtts import gTTS
from datetime import datetime
from io import BytesIO
from collections import defaultdict
import logging, random

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(page_title="Asisten Tunanetra", page_icon="👁️",
                   layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
html, body, [data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #f5f7ff 0%, #faf8ff 100%) !important;
    font-family: 'Inter', sans-serif;
}
.header-logo { width:50px;height:50px;background:linear-gradient(135deg,#6c3fff,#3f8bff);border-radius:14px;display:flex;align-items:center;justify-content:center;font-size:1.4rem;box-shadow:0 4px 15px rgba(108,63,255,.3); }
.header-text h1 { font-size:1.8rem;color:#1a1a2e;margin:0; }
.header-text p { font-size:.85rem;color:#888;margin:0; }
.pills { display:flex;gap:.6rem;flex-wrap:wrap;margin-top:.8rem; }
.pill { padding:.35rem 1rem;border-radius:20px;font-size:.75rem;font-weight:600;border:1px solid transparent; }
.pill-run { background:#ede8ff;color:#6c3fff;border-color:#c4b0ff; }
.pill-ok { background:#e6fff5;color:#00955a;border-color:#80ecc0; }
.pill-danger { background:#fff0f0;color:#cc2222;border-color:#ffaaaa; }
.pill-warn { background:#fffbe6;color:#ad6800;border-color:#ffe58f; }
.pill-ocr { background:#e8f4ff;color:#0066cc;border-color:#90c8ff; }
.pill-model { background:#f0e8ff;color:#6c3fff;border-color:#c4b0ff; }
.alert-danger { background:#fff0f0;border:1px solid #ffaaaa;border-left:4px solid #ff4444;border-radius:8px;padding:1rem;color:#cc2222;font-weight:600;margin:.8rem 0; }
.alert-warning { background:#fffbe6;border:1px solid #ffe58f;border-left:4px solid #faad14;border-radius:8px;padding:1rem;color:#ad6800;font-weight:600;margin:.8rem 0; }
.alert-info { background:#f0f6ff;border:1px solid #90c8ff;border-left:4px solid #3f8bff;border-radius:8px;padding:1rem;color:#0066cc;margin:.8rem 0; }
.alert-success { background:#f0fff8;border:1px solid #80ecc0;border-left:4px solid #00c073;border-radius:8px;padding:1rem;color:#00955a;margin:.8rem 0; }
.ocr-result { background:#f8fbff;border:2px solid #3f8bff;border-radius:10px;padding:1.4rem;color:#0066cc;font-size:1.05rem;min-height:60px;white-space:pre-wrap; }
.stat-box { background:linear-gradient(135deg,#f5f7ff,#faf8ff);border-radius:10px;padding:1.2rem;text-align:center;border:1px solid #e0e4f0; }
.stat-value { font-size:2rem;font-weight:700;color:#6c3fff; }
.stat-label { font-size:.8rem;color:#999;text-transform:uppercase;letter-spacing:1px;font-weight:600; }
.stButton > button { border-radius:10px;border:none;font-weight:600;padding:.8rem 1.4rem;transition:all .2s;text-transform:uppercase;letter-spacing:.5px; }
.stButton > button:hover { transform:translateY(-2px);box-shadow:0 4px 12px rgba(108,63,255,.3); }
#MainMenu, footer { visibility:hidden;display:none; }
</style>
""", unsafe_allow_html=True)

# ─── SESSION STATE ───
session_defaults = {
    'model1':None,'model2':None,'model3':None,'ocr_engine':None,
    'last_frame':None,'log':[],'detection_history':[],
    'last_alert_time': defaultdict(lambda:-99.0),
    'alerted_objects': set(),
    'ocr_triggered':False,'ocr_frame_count':0,
    'last_uploaded_name':None,'last_ocr_text':'',
}
for k,v in session_defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

def add_log(msg):
    st.session_state.log.insert(0,(time.strftime("%H:%M:%S"),msg))
    st.session_state.log = st.session_state.log[:30]

# ─── AUDIO ───
def play_audio_safe(ph, audio_bytes):
    if audio_bytes:
        b64 = base64.b64encode(audio_bytes).decode()
        uid = f"a_{int(time.time()*1000)}_{random.randint(1000,9999)}"
        ph.markdown(f'<audio autoplay style="display:none" id="{uid}"><source src="data:audio/mp3;base64,{b64}" type="audio/mp3"></audio><script>setTimeout(function(){{document.getElementById("{uid}").play()}},100)</script>', unsafe_allow_html=True)

def get_audio_bytes(text, lang='id'):
    try:
        buf = BytesIO()
        gTTS(text=text, lang=lang, slow=False).write_to_fp(buf)
        buf.seek(0)
        return buf.read()
    except: return None

# ─── TERJEMAHAN ───
INDO_NAMES = {
    'person':'orang','car':'mobil','bus':'bus','truck':'truk',
    'motorcycle':'motor','bicycle':'sepeda','dog':'anjing','cat':'kucing',
    'pothole':'lubang','stairs':'tangga','obstacle':'rintangan',
    'road-barrier':'pembatas jalan','pole':'tiang','train':'kereta',
    'stop sign':'rambu stop','traffic light':'lampu lalu lintas',
    'sidewalk':'trotoar','crosswalk':'zebra cross','tree':'pohon',
    'dilarang masuk':'dilarang masuk','dilarang parkir':'dilarang parkir',
    'dilarang berhenti':'dilarang berhenti','hati-hati':'hati-hati',
    'rumah sakit':'rumah sakit','masjid':'masjid','gereja':'gereja',
    'pom bensin':'pom bensin','tempat parkir':'tempat parkir',
    'jalur sepeda':'jalur sepeda','batas kecepatan':'batas kecepatan',
    'persimpangan':'persimpangan','ikuti arah bundaran':'ikuti arah bundaran',
}
def get_indo_name(name):
    n = name.lower()
    if n in INDO_NAMES: return INDO_NAMES[n]
    for k,v in INDO_NAMES.items():
        if k in n or n in k: return v
    return n

RAMBU_KW = {'rambu','lampu lalu lintas','traffic light','stop sign','dilarang',
            'hati-hati','rumah sakit','masjid','gereja','pom bensin','parkir',
            'persimpangan','bundaran','jalur','kecepatan','berhenti','masuk'}
def is_rambu(name):
    n = name.lower()
    return any(k in n for k in RAMBU_KW)

def generate_rambu_alert(name):
    n = name.lower()
    m = {'stop sign':'Ada rambu stop. Berhenti!','rambu stop':'Ada rambu stop. Berhenti!',
         'dilarang masuk':'Dilarang masuk!','hati-hati':'Hati-hati!',
         'lampu lalu lintas':'Ada lampu lalu lintas','rumah sakit':'Ada rumah sakit',
         'masjid':'Ada masjid','gereja':'Ada gereja','pom bensin':'Ada pom bensin',
         'persimpangan':'Ada persimpangan'}
    for k,v in m.items():
        if k in n: return v
    return f"Ada {get_indo_name(n)}"

def generate_alert(name, pos_x, area):
    nama = get_indo_name(name)
    if pos_x < 0.3: pos,arah = "di kiri","Geser ke kanan"
    elif pos_x > 0.7: pos,arah = "di kanan","Geser ke kiri"
    else: pos,arah = "di depan","Berhenti"
    if area > 0.15: return f"Awas! {nama} sangat dekat {pos}! {arah} sekarang!"
    elif area > 0.06: return f"Hati-hati, ada {nama} {pos}. {arah}."
    else: return f"Ada {nama} {pos}. {arah} pelan-pelan."

# ============================================================
# OCR — ANTI TYPO + ANTI MENGEJA (FULL)
# ============================================================
COMMON_WORDS = {
    'jln':'jalan','jl':'jalan','jalan':'jalan',
    'dlarang':'dilarang','dlrang':'dilarang','dilarag':'dilarang',
    'dilarng':'dilarang','dilrang':'dilarang','dilarangg':'dilarang',
    'parkr':'parkir','prkir':'parkir','pakir':'parkir','parkirr':'parkir',
    'masuk':'masuk','msuk':'masuk','mausk':'masuk','mauk':'masuk',
    'brhenti':'berhenti','brheti':'berhenti','berheti':'berhenti','berhentii':'berhenti',
    'hti-hti':'hati-hati','hati-hti':'hati-hati','hti-hati':'hati-hati',
    'rmah':'rumah','rumh':'rumah','rumahh':'rumah','sakit':'sakit','sakt':'sakit',
    'msjid':'masjid','masjd':'masjid','masjidh':'masjid',
    'grja':'gereja','greja':'gereja','gerja':'gereja',
    'awas':'awas','bahaya':'bahaya','bhaya':'bahaya','bhy':'bahaya',
    'kluar':'keluar','kluaar':'keluar','blok':'belok','belk':'belok',
    'knan':'kanan','kri':'kiri','dpan':'depan','belkang':'belakang',
    'stp':'stop','stopp':'stop','stop':'stop',
    'lmpu':'lampu','lampo':'lampu','lampuu':'lampu',
    'mrah':'merah','hjau':'hijau','kunng':'kuning',
    'pnyebrangan':'penyeberangan','penyebrangn':'penyeberangan','penyeberangn':'penyeberangan',
    'zebr':'zebra','crss':'cross',
    'kec':'kecepatan','kecepata':'kecepatan','maks':'maksimal','min':'minimal',
    'spbu':'SPBU','spb':'SPBU',
    'rintangn':'rintangan','rintanga':'rintangan','obstacle':'rintangan',
    'lubang':'lubang','luba':'lubang','lubangg':'lubang',
    'tangga':'tangga','tanggaa':'tangga','tanga':'tangga',
}
# Kata yang tidak perlu (noise)
NOISE_WORDS = {'a','i','u','e','o','yang','dan','di','ke','dari','untuk','dengan',
               'pada','atau','itu','ini','yang','akan','telah','sudah','bisa','dapat'}

def autocorrect_word(word):
    if len(word) <= 1: return word
    low = word.lower()
    # Hapus huruf berulang (contoh: "lubangg" -> "lubang")
    clean = re.sub(r'(.)\1{2,}', r'\1', low)
    if clean in COMMON_WORDS: return COMMON_WORDS[clean]
    # Cek similarity
    best = None; best_sim = 0
    for k,v in COMMON_WORDS.items():
        if abs(len(k)-len(clean)) > 3: continue
        common = sum(1 for c in clean if c in k)
        sim = common / max(len(k), len(clean))
        if sim > 0.7 and sim > best_sim:
            best_sim = sim; best = v
    return best if best else word

def fix_spelled_text(text):
    if not text: return text
    # Gabung huruf terpisah: "D I L A R A N G" -> "DILARANG"
    words = text.split()
    result = []; i=0
    while i < len(words):
        if len(words[i]) == 1 and words[i].isalpha():
            chars = [words[i]]; j=i+1
            while j < len(words) and len(words[j]) == 1 and words[j].isalpha():
                chars.append(words[j]); j+=1
            if len(chars) >= 3:
                result.append(''.join(chars))
            else:
                result.extend(chars)
            i = j
        else:
            result.append(words[i]); i += 1
    return ' '.join(result)

def clean_ocr_text(raw_text):
    if not raw_text: return "Tidak ada teks terdeteksi"
    # Step 1: Fix mengeja
    text = fix_spelled_text(raw_text)
    # Step 2: Clean
    text = re.sub(r'[^a-zA-Z0-9\s\.,!?\-:/()%]', '', text)
    # Step 3: Autocorrect
    words = text.split()
    corrected = []
    for w in words:
        if len(w) <= 1: continue
        if w.lower() in NOISE_WORDS: continue
        corrected.append(autocorrect_word(w))
    if not corrected: return "Tidak ada teks terdeteksi"
    text = ' '.join(corrected)
    text = ' '.join(text.split())
    if len(text) < 3: return "Tidak ada teks terdeteksi"
    return text

def perform_ocr_on_frame(frame, ocr_engine, min_conf=0.15):
    if ocr_engine is None: return "OCR tidak tersedia"
    try:
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else: gray = frame
        h,w = gray.shape
        if w < 400:
            scale = 640/w
            gray = cv2.resize(gray, (640, int(h*scale)), interpolation=cv2.INTER_CUBIC)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
        sharp = cv2.filter2D(enhanced, -1, kernel)
        # Try multiple
        all_texts = []
        for img in [sharp, enhanced, frame]:
            try:
                results = ocr_engine.readtext(img, detail=1, paragraph=False)
                if results:
                    for (bbox, text, conf) in results:
                        text = text.strip()
                        if len(text) > 0 and conf > min_conf:
                            all_texts.append((text, conf))
            except: continue
            if any(c > 0.4 for _,c in all_texts): break
        if all_texts:
            all_texts.sort(key=lambda x: x[1], reverse=True)
            seen=set(); unique=[]
            for text, conf in all_texts:
                norm = text.lower().strip()
                if norm not in seen and len(text.strip())>0:
                    seen.add(norm); unique.append(text)
            raw = ' '.join(unique)
            cleaned = clean_ocr_text(raw)
            return cleaned
        return "Tidak ada teks terdeteksi"
    except Exception as e:
        return f"Error: {str(e)[:30]}"

def texts_are_similar(t1, t2, threshold=0.6):
    if not t1 or not t2: return False
    a,b = ' '.join(t1.lower().split()), ' '.join(t2.lower().split())
    if a == b: return True
    if len(a) < 3 or len(b) < 3: return a == b
    def bg(s): return set(s[i:i+2] for i in range(len(s)-1))
    s1,s2 = bg(a),bg(b)
    if not s1 or not s2: return a == b
    return len(s1 & s2) / len(s1 | s2) >= threshold

# ============================================================
# DETEKSI — HANYA OBSTACLE YANG BAHAYA
# ============================================================
OBSTACLE_KW = ['pothole','lubang','hole','stairs','tangga','step','obstacle',
               'rintangan','road-barrier','pembatas','pole','tiang','pillar',
               'bollard','crack','bump','curb','fence','pagar']
VEHICLE_KW = ['car','mobil','bus','truck','truk','vehicle','kendaraan',
              'motorcycle','motor','bicycle','sepeda','train','kereta']
PERSON_KW = ['person','orang']

def classify_object(cls_lower):
    if any(k in cls_lower for k in OBSTACLE_KW): return 'obstacle'
    if any(k in cls_lower for k in VEHICLE_KW): return 'vehicle'
    if any(k in cls_lower for k in PERSON_KW): return 'person'
    return 'other'

def process_frame_detection(frame, model, conf=0.4, is_m2=False):
    if model is None: return frame, []
    try:
        # M2: conf lebih rendah
        effective_conf = max(conf - 0.15, 0.12) if is_m2 else conf
        results = model.predict(frame, conf=effective_conf, iou=0.45, verbose=False)
        detections = []
        if results and len(results) > 0:
            result = results[0]
            if result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes.xyxy.cpu().numpy()
                confs = result.boxes.conf.cpu().numpy()
                clses = result.boxes.cls.cpu().numpy().astype(int)
                fh, fw = frame.shape[:2]
                for box, cs, ci in zip(boxes, confs, clses):
                    x1,y1,x2,y2 = map(int, box)
                    cn = result.names[ci]
                    area = (x2-x1)*(y2-y1)
                    ar = area/(fw*fh)
                    px = ((x1+x2)/2)/fw
                    cat = classify_object(cn.lower())
                    # Filter area
                    if cat == 'obstacle':
                        if ar < 0.005: continue
                        min_conf_show = 0.20 if is_m2 else 0.30
                        if cs < min_conf_show: continue
                    elif cat == 'vehicle':
                        if ar < 0.02 or cs < 0.40: continue
                    elif cat == 'person':
                        if ar < 0.015 or cs < 0.35: continue
                    else:
                        if ar < 0.015: continue
                    # RISK: HANYA OBSTACLE YANG BAHAYA
                    if cat == 'obstacle':
                        if cs > 0.40 and ar > 0.04:
                            risk_level = 'BAHAYA'
                        elif cs > 0.25 and ar > 0.015:
                            risk_level = 'WASPADA'
                        else:
                            risk_level = 'AMAN'
                    elif cat == 'vehicle':
                        if cs > 0.50 and ar > 0.10:
                            risk_level = 'WASPADA'
                        else:
                            risk_level = 'AMAN'
                    elif cat == 'person':
                        if cs > 0.45 and ar > 0.06:
                            risk_level = 'WASPADA'
                        else:
                            risk_level = 'AMAN'
                    else:
                        risk_level = 'AMAN'
                    detections.append({
                        'class':cn,'confidence':float(cs),'area_ratio':ar,
                        'position_x':px,'risk_level':risk_level,'bbox':(x1,y1,x2,y2),
                        'timestamp':datetime.now()
                    })
                    color = (0,0,255) if risk_level=='BAHAYA' else (0,165,255) if risk_level=='WASPADA' else (0,255,0)
                    cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)
                    lbl = f"{get_indo_name(cn)} {cs:.0%}"
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    (tw,th),_ = cv2.getTextSize(lbl,font,0.55,2)
                    cv2.rectangle(frame,(x1,y1-th-8),(x1+tw+4,y1),color,-1)
                    cv2.putText(frame,lbl,(x1+2,y1-4),font,0.55,(255,255,255),2)
        return frame, detections
    except Exception as e:
        logger.error(f"Det err: {e}")
        return frame, []

def process_frame_detection_multi(frame, m1, m2, m3, conf=0.4):
    out = frame.copy()
    all_d = []
    if m1:
        out, d = process_frame_detection(out, m1, conf, is_m2=False)
        all_d.extend(d)
    if m2:
        out, d = process_frame_detection(out, m2, conf, is_m2=True)
        all_d.extend(d)
    if m3:
        out, d = process_frame_detection(out, m3, conf, is_m2=False)
        all_d.extend(d)
    return out, all_d

# ─── ALERT ───
def handle_alerts(dets, now_t, en_audio, cooldown, warn_ph, status_ph, ad_ph, ar_ph):
    danger = [d for d in dets if d['risk_level'] == 'BAHAYA']
    waspada = [d for d in dets if d['risk_level'] == 'WASPADA']
    rambu = [d for d in dets if is_rambu(d['class'])]
    
    current = {d['class'] for d in dets}
    st.session_state.alerted_objects &= current

    if danger and en_audio:
        d = danger[0]
        k = d['class']
        if k not in st.session_state.alerted_objects:
            msg = generate_alert(d['class'],d['position_x'],d['area_ratio'])
            warn_ph.markdown(f'<div class="alert-danger">🚨 {msg}</div>',unsafe_allow_html=True)
            a = get_audio_bytes(msg)
            if a: ad_ph.empty(); play_audio_safe(ad_ph, a)
            st.session_state.alerted_objects.add(k)
            add_log(f"BAHAYA: {get_indo_name(k)}")
        else:
            warn_ph.markdown(f'<div class="alert-danger">🚨 Masih ada {get_indo_name(d["class"])}!</div>',unsafe_allow_html=True)
        status_ph.markdown('<div class="pills"><span class="pill pill-run">● AKTIF</span><span class="pill pill-danger">● BAHAYA</span></div>',unsafe_allow_html=True)
    elif waspada and en_audio:
        d = waspada[0]
        k = f"w_{d['class']}"
        if (now_t - st.session_state.last_alert_time[k]) >= cooldown:
            msg = generate_alert(d['class'],d['position_x'],d['area_ratio'])
            warn_ph.markdown(f'<div class="alert-warning">⚡ {msg}</div>',unsafe_allow_html=True)
            a = get_audio_bytes(msg)
            if a: ad_ph.empty(); play_audio_safe(ad_ph, a)
            st.session_state.last_alert_time[k] = now_t
        status_ph.markdown('<div class="pills"><span class="pill pill-run">● AKTIF</span><span class="pill pill-warn">● WASPADA</span></div>',unsafe_allow_html=True)
    elif rambu and en_audio:
        d = rambu[0]
        k = f"r_{d['class']}"
        if (now_t - st.session_state.last_alert_time[k]) >= cooldown:
            msg = generate_rambu_alert(d['class'])
            warn_ph.markdown(f'<div class="alert-info">ℹ️ {msg}</div>',unsafe_allow_html=True)
            a = get_audio_bytes(msg)
            if a: ar_ph.empty(); play_audio_safe(ar_ph, a)
            st.session_state.last_alert_time[k] = now_t
        status_ph.markdown('<div class="pills"><span class="pill pill-run">● AKTIF</span><span class="pill pill-ocr">● RAMBU</span></div>',unsafe_allow_html=True)
    else:
        warn_ph.markdown('<div class="alert-success">✅ Jalur aman</div>',unsafe_allow_html=True)
        status_ph.markdown('<div class="pills"><span class="pill pill-run">● AKTIF</span><span class="pill pill-ok">● AMAN</span></div>',unsafe_allow_html=True)
    return danger, rambu

def handle_ocr_dedup(frame, ocr_eng, min_conf, en_tts, ocr_ph, aocr_ph):
    text = perform_ocr_on_frame(frame, ocr_eng, min_conf)
    if text and text != "Tidak ada teks terdeteksi" and len(text) > 3:
        if not texts_are_similar(text, st.session_state.last_ocr_text):
            st.session_state.last_ocr_text = text
            ocr_ph.markdown(f'<div class="ocr-result">📝 {text}</div>',unsafe_allow_html=True)
            add_log(f"OCR: {text[:40]}")
            if en_tts:
                a = get_audio_bytes(f"Ada tulisan: {text}")
                if a: aocr_ph.empty(); play_audio_safe(aocr_ph, a)
            return True
    return False

# ─── LOAD MODELS ───
@st.cache_resource(show_spinner=False)
def load_yolo_models(p1='yolo11s.pt',p2=None,p3=None):
    try:
        from ultralytics import YOLO
        m = {}
        try: m['m1'] = YOLO(p1)
        except: m['m1'] = None
        m['m2'] = YOLO(p2) if p2 else None
        m['m3'] = YOLO(p3) if p3 else None
        return m
    except: return {}

@st.cache_resource(show_spinner=False)
def load_ocr():
    try:
        import easyocr
        return easyocr.Reader(['id','en'], gpu=False)
    except: return None

# ─── HEADER ───
c1,c2 = st.columns([0.1,0.9])
with c1: st.markdown('<div class="header-logo">👁️</div>',unsafe_allow_html=True)
with c2: st.markdown('<div class="header-text"><h1>Asisten Navigasi Tunanetra</h1><p>Deteksi Objek + Baca Teks — v5.1</p></div>',unsafe_allow_html=True)
st.divider()

# ─── SIDEBAR ───
with st.sidebar:
    st.markdown("### ⚙️ Pengaturan")
    if st.button("📥 Load YOLO",use_container_width=True):
        with st.spinner("Loading..."): models=load_yolo_models(); st.session_state.model1=models.get('m1'); add_log("YOLO loaded"); st.success("✅ YOLO!")
    if st.button("📥 Load OCR",use_container_width=True):
        with st.spinner("Loading..."): st.session_state.ocr_engine=load_ocr(); add_log("OCR loaded"); st.success("✅ OCR!")
    st.markdown("---")
    with st.expander("🏔️ Model 2: Tangga/Lubang"):
        up2 = st.file_uploader("Upload best.pt",type=['pt'],key='m2up')
        if up2 and st.button("Load M2"):
            with st.spinner("Memuat..."):
                with tempfile.NamedTemporaryFile(delete=False,suffix='.pt') as tmp: tmp.write(up2.read())
                from ultralytics import YOLO; st.session_state.model2=YOLO(tmp.name); add_log("M2 loaded"); st.success("✅ M2!")
    with st.expander("🚦 Model 3: Rambu"):
        up3 = st.file_uploader("Upload best_rambu.pt",type=['pt'],key='m3up')
        if up3 and st.button("Load M3"):
            with st.spinner("Memuat..."):
                with tempfile.NamedTemporaryFile(delete=False,suffix='.pt') as tmp: tmp.write(up3.read())
                from ultralytics import YOLO; st.session_state.model3=YOLO(tmp.name); add_log("M3 loaded"); st.success("✅ M3!")
    st.markdown("---")
    s1="✅" if st.session_state.model1 else "⚠️"
    s2="✅" if st.session_state.model2 else "⚠️"
    s3="✅" if st.session_state.model3 else "⚠️"
    so="✅" if st.session_state.ocr_engine else "⚠️"
    st.markdown(f'<div class="pills"><span class="pill pill-model">{s1} M1</span><span class="pill pill-model">{s2} M2</span><span class="pill pill-model">{s3} M3</span><span class="pill pill-model">{so} OCR</span></div>',unsafe_allow_html=True)
    conf_threshold = st.slider("Confidence",0.1,0.9,0.35,0.05)
    enable_audio = st.checkbox("🔊 Audio",value=True)
    alert_cooldown = st.slider("Cooldown (s)",2,10,5)
    ocr_min_conf = st.slider("OCR Conf",0.1,0.9,0.15,0.05)
    ocr_interval = st.slider("OCR Interval",1,10,5)
    enable_tts = st.checkbox("🔊 TTS",value=True)
    show_logs = st.checkbox("📋 Logs",value=True)
    frame_skip = st.slider("Frame Skip",1,10,3,help="Skip N frame. Lebih tinggi = lebih lancar")

# ─── TABS ───
tab1, tab2, tab3 = st.tabs(["🎯 Detection","📖 Text Reading","📊 Statistics"])

with tab1:
    mode = st.radio("Mode:",["📹 Webcam","📤 Upload Video"],horizontal=True)
    st.divider()

    if mode == "📹 Webcam":
        st.markdown('<div class="alert-info">📸 Ambil foto → otomatis dianalisis.</div>',unsafe_allow_html=True)
        ocr_on = st.checkbox("📖 Baca Teks (OCR)",value=False)
        frame_ph=st.empty(); status_ph=st.empty(); warn_ph=st.empty(); ocr_ph=st.empty()
        ad_ph=st.empty(); ar_ph=st.empty(); aocr_ph=st.empty()
        c1,c2,c3=st.columns(3)
        with c1: md=st.empty()
        with c2: mb=st.empty()
        with c3: mm=st.empty()
        if show_logs: lp=st.expander("📋 Logs").empty()
        cam = st.camera_input("📸 Ambil foto")
        if cam:
            if not st.session_state.model1 and not st.session_state.model2:
                st.error("⚠️ Load YOLO dulu!")
            else:
                from PIL import Image
                img=np.array(Image.open(cam))
                bgr=cv2.cvtColor(img,cv2.COLOR_RGB2BGR) if len(img.shape)==3 and img.shape[2]==3 else img
                bgr=cv2.resize(bgr,(640,480)); orig=bgr.copy()
                ann,dets=process_frame_detection_multi(bgr,st.session_state.model1,st.session_state.model2,st.session_state.model3,conf_threshold)
                frame_ph.image(cv2.cvtColor(ann,cv2.COLOR_BGR2RGB),caption=f"{len(dets)} objek",use_container_width=True)
                for d in dets: st.session_state.detection_history.append(d)
                st.session_state.detection_history=st.session_state.detection_history[-500:]
                dl,rl=handle_alerts(dets,time.time(),enable_audio,alert_cooldown,warn_ph,status_ph,ad_ph,ar_ph)
                md.metric("Deteksi",len(dets)); mb.metric("⚠️",len(dl)); mm.metric("Mode","📸")
                if ocr_on and st.session_state.ocr_engine:
                    handle_ocr_dedup(orig,st.session_state.ocr_engine,ocr_min_conf,enable_tts,ocr_ph,aocr_ph)
                if show_logs: lp.markdown('<br>'.join([f'[{t}] {m}' for t,m in st.session_state.log[:10]]),unsafe_allow_html=True)

    else:
        uploaded = st.file_uploader("Upload video",type=['mp4','avi','mov','mkv','webm','m4v'])
        frame_ph=st.empty(); status_ph=st.empty(); warn_ph=st.empty(); ocr_ph=st.empty()
        ad_ph=st.empty(); ar_ph=st.empty(); aocr_ph=st.empty()
        c1,c2,c3=st.columns(3)
        with c1: md=st.empty()
        with c2: mb=st.empty()
        with c3: mf=st.empty()
        if show_logs: lp=st.expander("📋 Logs").empty()

        if uploaded:
            cn=uploaded.name
            if st.session_state.last_uploaded_name != cn:
                st.session_state.ocr_triggered=False; st.session_state.ocr_frame_count=0
                st.session_state.last_uploaded_name=cn; st.session_state.last_ocr_text=''
                st.session_state.alerted_objects.clear(); ocr_ph.empty()

        col1,col2=st.columns(2)
        with col1: btn_start=st.button("▶️ Start",use_container_width=True)
        with col2: btn_ocr=st.button("📖 Baca Teks",use_container_width=True)
        if btn_ocr:
            st.session_state.ocr_triggered=True; st.session_state.ocr_frame_count=0
            st.session_state.last_ocr_text=''; st.info("🔍 OCR aktif.")

        if uploaded and btn_start:
            if not st.session_state.model1 and not st.session_state.model2:
                st.error("⚠️ Load YOLO!"); st.stop()
            st.session_state.alerted_objects.clear()
            
            suffix = os.path.splitext(uploaded.name)[1] or '.mp4'
            with tempfile.NamedTemporaryFile(delete=False,suffix=suffix) as tmp:
                tmp.write(uploaded.read()); vid_path=tmp.name
            
            try:
                cap = cv2.VideoCapture(vid_path)
                if not cap.isOpened():
                    st.error("❌ Video tidak bisa dibuka! Coba format MP4 H.264.")
                    st.stop()
                
                total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps_v=cap.get(cv2.CAP_PROP_FPS) or 30.0
                m1,m2,m3=st.session_state.model1,st.session_state.model2,st.session_state.model3
                ocr=st.session_state.ocr_engine
                cnt,start=0,time.time()
                prog=st.progress(0)
                
                while True:
                    ok,frame=cap.read()
                    if not ok: break
                    cnt+=1
                    if cnt % frame_skip != 0: continue
                    
                    # RESIZE KECIL AGAR CEPAT
                    frame=cv2.resize(frame,(320,240))
                    orig=frame.copy()
                    now_sec=cnt/fps_v
                    prog.progress(min(cnt/max(total,1),1.0),text=f"{cnt}/{total} ({now_sec:.1f}s)")
                    
                    ann,dets=process_frame_detection_multi(frame,m1,m2,m3,conf_threshold)
                    frame_ph.image(cv2.cvtColor(ann,cv2.COLOR_BGR2RGB),use_container_width=True)
                    st.session_state.last_frame=orig
                    for d in dets: st.session_state.detection_history.append(d)
                    st.session_state.detection_history=st.session_state.detection_history[-500:]
                    
                    dl,rl=handle_alerts(dets,time.time(),enable_audio,alert_cooldown,warn_ph,status_ph,ad_ph,ar_ph)
                    
                    if st.session_state.ocr_triggered and ocr:
                        st.session_state.ocr_frame_count+=1
                        if st.session_state.ocr_frame_count % ocr_interval == 0:
                            handle_ocr_dedup(orig,ocr,ocr_min_conf,enable_tts,ocr_ph,aocr_ph)
                    
                    md.metric("Deteksi",len(dets)); mb.metric("⚠️",len(dl))
                    el=time.time()-start
                    mf.metric("FPS",f"{cnt/el:.1f}" if el>0 else "0")
                    if show_logs: lp.markdown('<br>'.join([f'[{t}] {m}' for t,m in st.session_state.log[:10]]),unsafe_allow_html=True)
                    time.sleep(0.001)
                
                cap.release(); prog.empty()
                st.success(f"✅ Selesai! {cnt} frame.")
                st.session_state.ocr_triggered=False
            except Exception as e:
                st.error(f"Error: {e}")
            finally:
                try: os.unlink(vid_path)
                except: pass

with tab2:
    st.markdown("### 📖 Text Reading")
    mode2=st.radio("Input:",["📷 Kamera","📤 Upload"],horizontal=True)
    ip,rp,ap=st.empty(),st.empty(),st.empty()
    if mode2=="📷 Kamera":
        cam2=st.camera_input("📸 Foto untuk baca teks")
        if cam2:
            if not st.session_state.ocr_engine: st.error("⚠️ Load OCR!")
            else:
                from PIL import Image
                img=Image.open(cam2); arr=np.array(img)
                ip.image(img,use_container_width=True)
                text=perform_ocr_on_frame(arr,st.session_state.ocr_engine,ocr_min_conf)
                rp.markdown(f'<div class="ocr-result">📝 {text}</div>',unsafe_allow_html=True)
                if text and text!="Tidak ada teks terdeteksi" and len(text)>3 and enable_tts:
                    a=get_audio_bytes(f"Ada tulisan: {text}")
                    if a: ap.empty(); play_audio_safe(ap,a)
    else:
        ui=st.file_uploader("Upload gambar",type=['jpg','jpeg','png','bmp'])
        if ui:
            from PIL import Image
            img=Image.open(ui); arr=np.array(img)
            ip.image(img,use_container_width=True)
            if not st.session_state.ocr_engine: st.error("⚠️ Load OCR!")
            else:
                text=perform_ocr_on_frame(arr,st.session_state.ocr_engine,ocr_min_conf)
                rp.markdown(f'<div class="ocr-result">📝 {text}</div>',unsafe_allow_html=True)
                if text and text!="Tidak ada teks terdeteksi" and len(text)>3 and enable_tts:
                    a=get_audio_bytes(f"Ada tulisan: {text}")
                    if a: ap.empty(); play_audio_safe(ap,a)

with tab3:
    st.markdown("### 📊 Statistics")
    if st.session_state.detection_history:
        h=st.session_state.detection_history
        t=len(h); dg=len([d for d in h if d['risk_level']=='BAHAYA'])
        w=len([d for d in h if d['risk_level']=='WASPADA']); am=len([d for d in h if d['risk_level']=='AMAN'])
        c1,c2,c3,c4=st.columns(4)
        c1.markdown(f'<div class="stat-box"><div class="stat-value">{t}</div><div class="stat-label">Total</div></div>',unsafe_allow_html=True)
        c2.markdown(f'<div class="stat-box"><div class="stat-value" style="color:#ff4444">{dg}</div><div class="stat-label">Bahaya</div></div>',unsafe_allow_html=True)
        c3.markdown(f'<div class="stat-box"><div class="stat-value" style="color:#faad14">{w}</div><div class="stat-label">Waspada</div></div>',unsafe_allow_html=True)
        c4.markdown(f'<div class="stat-box"><div class="stat-value" style="color:#00c073">{am}</div><div class="stat-label">Aman</div></div>',unsafe_allow_html=True)
        df=pd.DataFrame([{'Waktu':d['timestamp'].strftime("%H:%M:%S"),'Objek':get_indo_name(d['class']),'Conf':f"{d['confidence']:.0%}",'Risk':d['risk_level'],'Pos':'Kiri' if d['position_x']<0.3 else('Kanan' if d['position_x']>0.7 else 'Depan')} for d in h[-100:]])
        st.dataframe(df,use_container_width=True)
        if st.button("🗑️ Hapus"):
            st.session_state.detection_history=[]; st.session_state.alerted_objects.clear(); st.rerun()
    else:
        st.info("📊 Belum ada data.")

st.divider()
st.markdown('<div style="text-align:center;color:#999;font-size:.8rem;padding:1rem 0"><strong>Asisten Navigasi Tunanetra v5.1</strong> • YOLOv11 • EasyOCR • gTTS</div>',unsafe_allow_html=True)
