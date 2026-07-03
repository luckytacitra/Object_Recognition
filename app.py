# =====================================================================
# ASISTEN NAVIGASI TUNANETRA - STREAMLIT v6.1 (FIX SUARA + DETEKSI + VIDEO)
# Basis: v5.12 + logika Colab Fiks_Indo_Video_2
# =====================================================================
# FIX v6.1:
# 1. Suara: multiple audio elements + JS force play + longer delay
# 2. YOLO M1: conf=0.35 (dari 0.7 — terlalu tinggi bikin miss)
# 3. M2: conf=0.10 tetap, case-insensitive class matching
# 4. Video: delay sinkron FPS asli, resize 640x480
# 5. OCR: multi-variant dari Colab SEL 12
# =====================================================================

import streamlit as st
import cv2, os, re, time, base64, tempfile
import numpy as np
import pandas as pd
from gtts import gTTS
from datetime import datetime
from io import BytesIO
from collections import defaultdict
from difflib import get_close_matches
import logging, random

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(page_title="Asisten Tunanetra", page_icon="👁️", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
*{box-sizing:border-box;margin:0;padding:0}
html,body,[data-testid="stAppViewContainer"]{background:linear-gradient(135deg,#f5f7ff,#faf8ff)!important;font-family:'Inter',sans-serif}
.header-logo{width:50px;height:50px;background:linear-gradient(135deg,#6c3fff,#3f8bff);border-radius:14px;display:flex;align-items:center;justify-content:center;font-size:1.4rem;box-shadow:0 4px 15px rgba(108,63,255,.3);color:white}
.header-text h1{font-size:1.8rem;color:#1a1a2e;margin:0}
.header-text p{font-size:.85rem;color:#888;margin:0}
.pills{display:flex;gap:.6rem;flex-wrap:wrap;margin-top:.8rem}
.pill{padding:.35rem 1rem;border-radius:20px;font-size:.75rem;font-weight:600;border:1px solid transparent}
.pill-run{background:#ede8ff;color:#6c3fff;border-color:#c4b0ff}
.pill-ok{background:#e6fff5;color:#00955a;border-color:#80ecc0}
.pill-danger{background:#fff0f0;color:#cc2222;border-color:#ffaaaa}
.pill-ocr{background:#e8f4ff;color:#0066cc;border-color:#90c8ff}
.pill-model{background:#f0e8ff;color:#6c3fff;border-color:#c4b0ff}
.alert-danger{background:#fff0f0;border:1px solid #ffaaaa;border-left:4px solid #ff4444;border-radius:8px;padding:1rem;color:#cc2222;font-weight:600;margin:.8rem 0}
.alert-warning{background:#fffbe6;border:1px solid #ffe58f;border-left:4px solid #faad14;border-radius:8px;padding:1rem;color:#ad6800;font-weight:600;margin:.8rem 0}
.alert-info{background:#f0f6ff;border:1px solid #90c8ff;border-left:4px solid #3f8bff;border-radius:8px;padding:1rem;color:#0066cc;margin:.8rem 0}
.alert-success{background:#f0fff8;border:1px solid #80ecc0;border-left:4px solid #00c073;border-radius:8px;padding:1rem;color:#00955a;margin:.8rem 0}
.ocr-result{background:#f8fbff;border:2px solid #3f8bff;border-radius:10px;padding:1.4rem;color:#0066cc;font-size:1.05rem;min-height:60px}
.stat-box{background:linear-gradient(135deg,#f5f7ff,#faf8ff);border-radius:10px;padding:1.2rem;text-align:center;border:1px solid #e0e4f0}
.stat-value{font-size:2rem;font-weight:700;color:#6c3fff}
.stat-label{font-size:.8rem;color:#999;text-transform:uppercase;letter-spacing:1px;font-weight:600}
.stButton>button{border-radius:10px;border:none;font-weight:600;padding:.8rem 1.4rem;transition:all .2s;text-transform:uppercase;letter-spacing:.5px}
.stButton>button:hover{transform:translateY(-2px);box-shadow:0 4px 12px rgba(108,63,255,.3)}
#MainMenu,footer{visibility:hidden;display:none}
</style>
""", unsafe_allow_html=True)

# ─── ID_MAP dari Colab ───
ID_MAP_VALID = {
    "person":"orang","car":"mobil","bus":"bus","truck":"truk",
    "motorcycle":"motor","bicycle":"sepeda","train":"kereta",
    "dog":"anjing","cat":"kucing",
    "stop sign":"rambu stop","traffic light":"lampu lalu lintas",
    "Stairs":"tangga","Pothole":"lubang","Over-bridge":"jembatan penyeberangan",
    "Railway":"rel kereta","Road-barrier":"pembatas jalan","Sidewalk":"trotoar",
    "Crosswalk":"jalur penyeberangan","Obstacle":"rintangan","Pole":"tiang",
    "Tree":"pohon","Vehicle":"kendaraan","Animal":"hewan",
    "Traffic-sign":"rambu","Traffic-light":"lampu lalu lintas",
    "Person":"orang","Train":"kereta",
    "Dilarang Masuk":"dilarang masuk","Dilarang Parkir":"dilarang parkir",
    "Dilarang Berhenti":"dilarang berhenti","Dilarang Belok Kanan":"dilarang belok kanan",
    "Dilarang Putar Balik":"dilarang putar balik","Dilarang Mendahului":"dilarang mendahului",
    "Dilarang Berjalan Terus":"dilarang berjalan terus",
    "Hati-Hati":"hati-hati","Lampu Lalu Lintas":"lampu lalu lintas",
    "Rumah Sakit":"rumah sakit","Masjid":"masjid","Gereja":"gereja",
    "SPBU":"pom bensin","Tempat Parkir":"tempat parkir",
    "Pemberhentian Bus":"pemberhentian bus",
    "Perintah Ikuti Bundaran":"ikuti arah bundaran",
    "Perintah Jalur Sepeda":"jalur sepeda",
    "Perintah Lajur Kiri":"gunakan lajur kiri",
    "Perintah Pilih Satu Jalur":"pilih satu jalur",
    "Persimpangan 3 Prioritas":"persimpangan tiga",
    "Persimpangan Empat":"persimpangan empat",
    "Putar Balik":"area putar balik",
    "Banyak Anak-Anak":"area banyak anak-anak",
    "Larangan Kecepatan - 30km-jam":"batas kecepatan 30 km per jam",
    "Larangan Kecepatan - 40km-jam":"batas kecepatan 40 km per jam",
}
KELAS_VALID_LOWER = {k.lower(): k for k in ID_MAP_VALID}

def id_nama(n):
    if not isinstance(n, str): return str(n)
    nc = n.strip()
    if nc in ID_MAP_VALID: return ID_MAP_VALID[nc]
    for k, v in ID_MAP_VALID.items():
        if k.lower() == nc.lower(): return v
    return nc

# Dari Colab SEL 7
KELAS_YOLO_RELEVAN = {"person","bicycle","car","motorcycle","bus","truck",
    "traffic light","stop sign","fire hydrant","dog","cat"}
KELAS_BEST_DIPAKAI = {"Stairs","Pothole","Over-bridge","Railway",
    "Road-barrier","Sidewalk","Crosswalk","Obstacle","Pole","Vehicle","Animal"}
KELAS_BEST_LOWER = {k.lower(): k for k in KELAS_BEST_DIPAKAI}

# Dari Colab SEL 11
FATAL_OBJECTS = {"pothole","obstacle","stairs","road-barrier","pole"}
CONF_MIN_FATAL = {"pothole":0.40,"obstacle":0.45,"stairs":0.35,"road-barrier":0.35,"pole":0.35}

RAMBU_KW = {'rambu','lampu lalu lintas','stop sign','dilarang','hati-hati',
    'rumah sakit','masjid','gereja','spbu','pom bensin','parkir',
    'persimpangan','bundaran','jalur','pemberhentian','kecepatan'}

def is_rambu(n): return any(k in n.lower() for k in RAMBU_KW)
def is_obstacle(n):
    return any(k in n.lower() for k in ['pothole','lubang','stairs','tangga',
        'obstacle','rintangan','road-barrier','pembatas','pole','tiang'])

# ─── SESSION STATE ───
_d = {
    'model1':None,'model2':None,'model3':None,'ocr_engine':None,
    'last_frame':None,'log':[],'detection_history':[],
    'last_alert_time':defaultdict(lambda:-99.0),
    'ocr_triggered_vid':False,'ocr_frame_count':0,
    'last_uploaded_name':None,'last_ocr_text':'',
    'detected_classes':set(),'active_objects':set(),
    'audio_counter':0,
}
for k,v in _d.items():
    if k not in st.session_state: st.session_state[k]=v

def add_log(msg):
    st.session_state.log.insert(0,(time.strftime("%H:%M:%S"),str(msg)))
    st.session_state.log=st.session_state.log[:30]

def render_log():
    if not st.session_state.log: return '<div style="color:#ccc;text-align:center;padding:1rem">Belum ada aktivitas</div>'
    rows=[]
    for item in st.session_state.log[:10]:
        if isinstance(item,(tuple,list)) and len(item)>=2:
            rows.append(f'[{item[0]}] {item[1]}')
    return '<br>'.join(rows) if rows else '<div style="color:#ccc">-</div>'

# ─── AUDIO — FIX: force play dengan JS retry ───
def play_audio_safe(placeholder, audio_bytes):
    """FIX: Multiple play attempts + user interaction workaround"""
    if not audio_bytes: return False
    b64 = base64.b64encode(audio_bytes).decode()
    st.session_state.audio_counter += 1
    uid = f"audio_{st.session_state.audio_counter}_{int(time.time()*1000)}"
    # JS yang lebih agresif: retry play setiap 200ms sampai berhasil
    html_code = f"""
        <div id="container_{uid}">
            <audio id="{uid}" preload="auto">
                <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
            </audio>
        </div>
        <script>
        (function() {{
            var audio = document.getElementById("{uid}");
            if (!audio) return;
            var attempts = 0;
            function tryPlay() {{
                if (attempts > 10) return;
                attempts++;
                var p = audio.play();
                if (p !== undefined) {{
                    p.catch(function(e) {{
                        setTimeout(tryPlay, 300);
                    }});
                }}
            }}
            audio.volume = 1.0;
            tryPlay();
            // Juga coba saat user klik apapun
            document.addEventListener('click', function handler() {{
                audio.play();
                document.removeEventListener('click', handler);
            }}, {{once: true}});
        }})();
        </script>
    """
    placeholder.markdown(html_code, unsafe_allow_html=True)
    return True

def get_audio_bytes(text, lang='id'):
    try:
        buf=BytesIO(); gTTS(text=text,lang=lang,slow=False).write_to_fp(buf)
        buf.seek(0); return buf.read()
    except: return None

# ─── ALERT MESSAGES ───
def generate_alert(name, pos_x, area):
    nama = id_nama(name)
    if pos_x < 0.35: pos,arah = "di kiri","Geser ke kanan"
    elif pos_x > 0.65: pos,arah = "di kanan","Geser ke kiri"
    else: pos,arah = "di depan","Berhenti"
    if area > 0.12: return f"Awas! {nama} sangat dekat {pos}. {arah} sekarang!"
    elif area > 0.04: return f"Awas! Ada {nama} {pos}. {arah}!"
    else: return f"Hati-hati, ada {nama} {pos}. {arah} pelan-pelan."

def generate_rambu_alert(name):
    n = name.lower()
    m = {'stop sign':'Rambu stop. Berhenti!','rambu stop':'Rambu stop. Berhenti!',
         'dilarang masuk':'Dilarang masuk!','hati-hati':'Hati-hati!',
         'lampu lalu lintas':'Ada lampu lalu lintas','rumah sakit':'Ada rumah sakit',
         'masjid':'Ada masjid','gereja':'Ada gereja','pom bensin':'Ada pom bensin',
         'persimpangan':'Ada persimpangan'}
    for k,v in m.items():
        if k in n: return v
    return f"Ada {id_nama(n)}"

# ─── OCR — dari Colab SEL 12 multi-variant ───
KATA_DASAR = {
    'kiri','kanan','depan','belakang','jalan','trotoar','lubang','tangga',
    'rambu','lampu','truk','mobil','motor','sepeda','bus','kereta',
    'rumah','toko','masjid','gereja','sekolah','kantor','parkir','berhenti',
    'hati','awas','bahaya','zona','inspirasi','photobox','toilet','mushola',
    'keluar','masuk','lantai','lift','dilarang','merokok','darurat',
    'informasi','loket','tiket','kasir','push','pull','tarik','dorong',
}

def perform_ocr_on_frame(frame, ocr_engine, min_conf=0.30):
    if ocr_engine is None: return "OCR engine tidak tersedia"
    try:
        if len(frame.shape)==3: gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        else: gray=frame
        h,w=gray.shape
        if w<800:
            s=800/w; gray=cv2.resize(gray,(800,int(h*s)),interpolation=cv2.INTER_CUBIC)
        # Multi-variant preprocessing dari Colab
        clahe=cv2.createCLAHE(clipLimit=2.5,tileGridSize=(8,8))
        enhanced=clahe.apply(gray)
        kern=np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
        sharpened=cv2.filter2D(enhanced,-1,kern)
        
        semua=[]
        for img in [sharpened, enhanced, gray, frame]:
            try:
                results=ocr_engine.readtext(img,detail=1,paragraph=False)
                if results:
                    for (bbox,text,conf) in results:
                        text=text.strip()
                        if len(text)>=2 and conf>=min_conf:
                            semua.append((text,conf))
            except: continue
            if any(c>0.7 for _,c in semua): break
        
        if not semua: return "Tidak ada teks terdeteksi"
        # Deduplikasi
        seen=set(); unik=[]
        for t,c in sorted(semua,key=lambda x:x[1],reverse=True):
            key=t.lower().strip()
            if key not in seen: seen.add(key); unik.append((t,c))
        # Gabung top results
        raw=' '.join([t for t,c in unik[:5]])
        # Clean + autocorrect
        raw=re.sub(r'[^a-zA-Z0-9\s\.,!?\-:/()%]','',raw)
        words=raw.split(); corrected=[]
        for w in words:
            if len(w)<2: continue
            matches=get_close_matches(w.lower(),KATA_DASAR,n=1,cutoff=0.7)
            if matches and matches[0]!=w.lower():
                corrected.append(matches[0].capitalize() if w[0].isupper() else matches[0])
            else: corrected.append(w)
        result=' '.join(corrected)
        return result.title() if len(result)>=3 else "Tidak ada teks terdeteksi"
    except Exception as e: return f"Error: {str(e)[:30]}"

def handle_ocr_dedup(frame,ocr_eng,min_conf,en_tts,ocr_ph,aocr_ph):
    text=perform_ocr_on_frame(frame,ocr_eng,min_conf)
    if text and text!="Tidak ada teks terdeteksi" and len(text)>3:
        if text!=st.session_state.last_ocr_text:
            st.session_state.last_ocr_text=text
            ocr_ph.markdown(f'<div class="ocr-result">📝 {text}</div>',unsafe_allow_html=True)
            add_log(f"📖 TEKS: {text[:40]}")
            if en_tts:
                a=get_audio_bytes(f"Ada tulisan: {text}")
                if a:
                    aocr_ph.empty()
                    time.sleep(0.3)  # Delay agar browser siap
                    play_audio_safe(aocr_ph,a)
                    add_log(f"🔊 OCR SUARA: {text[:30]}")
                    return True
    return False

# ─── DETEKSI — conf dari Colab ───
def process_frame_detection(frame, model, conf=0.35, model_type='m1'):
    """
    model_type: 'm1'=YOLO umum, 'm2'=obstacle, 'm3'=rambu
    """
    if model is None: return frame, []
    try:
        # Conf per model — dari Colab
        if model_type=='m1': effective_conf=0.35
        elif model_type=='m2': effective_conf=0.10  # rendah agar lubang kedetect
        else: effective_conf=0.70  # rambu harus yakin
        
        results=model.predict(frame,conf=effective_conf,iou=0.45,verbose=False)
        detections=[]
        if results and results[0].boxes is not None and len(results[0].boxes)>0:
            r=results[0]
            boxes=r.boxes.xyxy.cpu().numpy()
            confs=r.boxes.conf.cpu().numpy()
            clses=r.boxes.cls.cpu().numpy().astype(int)
            fh,fw=frame.shape[:2]
            
            for box,cs,ci in zip(boxes,confs,clses):
                cn=r.names[ci]
                cn_lower=cn.lower()
                
                # Filter class berdasarkan model
                if model_type=='m1':
                    if cn_lower not in KELAS_YOLO_RELEVAN: continue
                elif model_type=='m2':
                    # Case-insensitive matching
                    if cn_lower not in KELAS_BEST_LOWER: continue
                    cn=KELAS_BEST_LOWER[cn_lower]
                # m3: tampilkan semua
                
                x1,y1,x2,y2=map(int,box)
                ar=(x2-x1)*(y2-y1)/(fw*fh)
                px=((x1+x2)/2)/fw
                cy=(y1+y2)/2
                
                if ar<0.002: continue
                # Colab BATAS_BAWAH_REL=0.9
                if model_type=='m2' and (cy/fh)>0.9: continue
                
                # Risk level
                if model_type=='m2':
                    cn_l=cn.lower()
                    is_fatal=cn_l in FATAL_OBJECTS
                    min_c=CONF_MIN_FATAL.get(cn_l,0.30)
                    if is_fatal and cs>=min_c and ar>=0.03: rl='BAHAYA'
                    elif is_fatal and cs>=(min_c*0.7) and ar>=0.01: rl='WASPADA'
                    else: rl='AMAN'
                elif model_type=='m3' or is_rambu(cn):
                    rl='RAMBU'
                else:
                    rl='AMAN'
                
                if rl=='AMAN' and ar<0.01: continue
                
                detections.append({
                    'class':cn,'confidence':float(cs),'area_ratio':ar,
                    'position_x':px,'risk_level':rl,'bbox':(x1,y1,x2,y2),
                    'timestamp':datetime.now(),'source':model_type
                })
                
                color=(0,0,255) if rl=='BAHAYA' else (0,165,255) if rl=='WASPADA' else (255,165,0) if rl=='RAMBU' else (0,255,0)
                cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)
                label=f"{id_nama(cn)} {cs:.0%}"
                ft=cv2.FONT_HERSHEY_SIMPLEX
                (tw,th),_=cv2.getTextSize(label,ft,0.5,2)
                cv2.rectangle(frame,(x1,y1-th-6),(x1+tw+4,y1),color,-1)
                cv2.putText(frame,label,(x1+2,y1-3),ft,0.5,(255,255,255),2)
        return frame, detections
    except Exception as e:
        logger.error(f"Det err: {e}")
        return frame, []

def process_frame_detection_multi(frame, m1, m2, m3):
    out=frame.copy(); all_d=[]
    if m1: out,d=process_frame_detection(out,m1,model_type='m1'); all_d.extend(d)
    if m2: out,d=process_frame_detection(out,m2,model_type='m2'); all_d.extend(d)
    if m3: out,d=process_frame_detection(out,m3,model_type='m3'); all_d.extend(d)
    return out, all_d

# ─── ALERT — dari v5.12 logic tapi dengan delay untuk suara ───
def handle_alerts(dets, now_time, enable_audio, warn_ph, status_ph, audio_ph, cooldown=2):
    danger=[d for d in dets if d['risk_level']=='BAHAYA']
    waspada=[d for d in dets if d['risk_level']=='WASPADA']
    rambu=[d for d in dets if d['risk_level']=='RAMBU']
    
    current={d['class'] for d in dets}
    st.session_state.active_objects=current
    new_danger=[d for d in danger if d['class'] not in st.session_state.detected_classes]
    new_waspada=[d for d in waspada if d['class'] not in st.session_state.detected_classes]
    new_rambu=[d for d in rambu if d['class'] not in st.session_state.detected_classes]
    
    played=False
    if new_danger and enable_audio:
        d=new_danger[0]
        msg=generate_alert(d['class'],d['position_x'],d['area_ratio'])
        warn_ph.markdown(f'<div class="alert-danger">🚨 {msg}</div>',unsafe_allow_html=True)
        a=get_audio_bytes(msg)
        if a:
            audio_ph.empty()
            time.sleep(0.2)
            play_audio_safe(audio_ph,a); played=True
            add_log(f"🔊 {msg}")
        st.session_state.detected_classes.add(d['class'])
        status_ph.markdown('<div class="pills"><span class="pill pill-run">● AKTIF</span><span class="pill pill-danger">● BAHAYA</span></div>',unsafe_allow_html=True)
    elif new_waspada and enable_audio:
        d=new_waspada[0]
        msg=generate_alert(d['class'],d['position_x'],d['area_ratio'])
        warn_ph.markdown(f'<div class="alert-warning">⚡ {msg}</div>',unsafe_allow_html=True)
        a=get_audio_bytes(msg)
        if a:
            audio_ph.empty(); time.sleep(0.2)
            play_audio_safe(audio_ph,a); played=True
            add_log(f"🔊 {msg}")
        st.session_state.detected_classes.add(d['class'])
        status_ph.markdown('<div class="pills"><span class="pill pill-run">● AKTIF</span><span class="pill pill-danger" style="background:#fffbe6;color:#ad6800;border-color:#ffe58f">● WASPADA</span></div>',unsafe_allow_html=True)
    elif new_rambu and enable_audio:
        d=new_rambu[0]
        msg=generate_rambu_alert(d['class'])
        warn_ph.markdown(f'<div class="alert-info">ℹ️ {msg}</div>',unsafe_allow_html=True)
        a=get_audio_bytes(msg)
        if a:
            audio_ph.empty(); time.sleep(0.2)
            play_audio_safe(audio_ph,a); played=True
            add_log(f"🔊 {msg}")
        st.session_state.detected_classes.add(d['class'])
        status_ph.markdown('<div class="pills"><span class="pill pill-run">● AKTIF</span><span class="pill pill-ocr">● RAMBU</span></div>',unsafe_allow_html=True)
    else:
        warn_ph.markdown('<div class="alert-success">✅ Jalur aman</div>',unsafe_allow_html=True)
        status_ph.markdown('<div class="pills"><span class="pill pill-run">● AKTIF</span><span class="pill pill-ok">● AMAN</span></div>',unsafe_allow_html=True)
    
    # Delay ekstra kalau ada suara agar browser sempat play
    if played: time.sleep(1.5)
    return danger, rambu

# ─── MODELS ───
@st.cache_resource(show_spinner=False)
def load_yolo(p='yolo11s.pt'):
    try:
        from ultralytics import YOLO; return YOLO(p)
    except: return None

@st.cache_resource(show_spinner=False)
def load_ocr():
    try:
        import easyocr; return easyocr.Reader(['id','en'],gpu=False)
    except: return None

# ─── HEADER ───
c1,c2=st.columns([0.1,0.9])
with c1: st.markdown('<div class="header-logo">👁️</div>',unsafe_allow_html=True)
with c2: st.markdown('<div class="header-text"><h1>Asisten Navigasi Tunanetra</h1><p>v6.1 — Colab-synced</p></div>',unsafe_allow_html=True)
st.divider()

# ─── SIDEBAR ───
with st.sidebar:
    st.markdown("### ⚙️ Pengaturan")
    if st.button("📥 Load YOLO (M1)",use_container_width=True):
        with st.spinner("Loading..."): st.session_state.model1=load_yolo(); st.success("✅ YOLO!")
    if st.button("📥 Load OCR",use_container_width=True):
        with st.spinner("Loading..."): st.session_state.ocr_engine=load_ocr(); st.success("✅ OCR!")
    st.markdown("---")
    with st.expander("🏔️ M2: best.pt (Lubang/Tangga)"):
        up2=st.file_uploader("Upload best.pt",type=['pt'],key='m2up')
        if up2 and st.button("Load M2"):
            with st.spinner("Memuat..."):
                with tempfile.NamedTemporaryFile(delete=False,suffix='.pt') as tmp: tmp.write(up2.read())
                from ultralytics import YOLO; st.session_state.model2=YOLO(tmp.name)
                names=st.session_state.model2.names
                st.success(f"✅ M2! Classes: {list(names.values())[:6]}")
    with st.expander("🚦 M3: best_rambu.pt"):
        up3=st.file_uploader("Upload best_rambu.pt",type=['pt'],key='m3up')
        if up3 and st.button("Load M3"):
            with st.spinner("Memuat..."):
                with tempfile.NamedTemporaryFile(delete=False,suffix='.pt') as tmp: tmp.write(up3.read())
                from ultralytics import YOLO; st.session_state.model3=YOLO(tmp.name)
                st.success("✅ M3!")
    st.markdown("---")
    s1="✅" if st.session_state.model1 else "⚠️"
    s2="✅" if st.session_state.model2 else "⚠️"
    s3="✅" if st.session_state.model3 else "⚠️"
    so="✅" if st.session_state.ocr_engine else "⚠️"
    st.markdown(f'<div class="pills"><span class="pill pill-model">{s1} M1</span><span class="pill pill-model">{s2} M2</span><span class="pill pill-model">{s3} M3</span><span class="pill pill-model">{so} OCR</span></div>',unsafe_allow_html=True)
    enable_audio=st.checkbox("🔊 Audio",value=True)
    alert_cooldown=st.slider("Cooldown (s)",1,5,2)
    ocr_min_conf=st.slider("OCR Conf",0.1,0.9,0.30,0.05)
    ocr_scan_interval=st.slider("OCR Interval (frame)",5,30,10)
    frame_skip=st.slider("Frame Skip",1,5,2)
    enable_tts=st.checkbox("🔊 TTS",value=True)
    show_logs=st.checkbox("📋 Logs",value=True)
    if st.button("🔁 Reset"):
        st.session_state.last_alert_time=defaultdict(lambda:-99.0)
        st.session_state.last_ocr_text=''; st.session_state.detected_classes=set()
        st.session_state.active_objects=set(); st.success("Reset!")

# ─── TABS ───
tab1,tab2,tab3=st.tabs(["🎯 Detection","📖 Text Reading","📊 Statistics"])

with tab1:
    mode=st.radio("Mode:",["📹 Webcam (Foto)","📤 Upload Video"],horizontal=True)
    st.divider()
    frame_ph=st.empty(); status_ph=st.empty(); warn_ph=st.empty(); ocr_ph=st.empty()
    audio_alert_ph=st.empty(); audio_ocr_ph=st.empty()
    c1,c2,c3=st.columns(3)
    with c1: m_det=st.empty()
    with c2: m_danger=st.empty()
    with c3: m_fps=st.empty()
    if show_logs: log_exp=st.expander("📋 Logs")

    if mode=="📹 Webcam (Foto)":
        st.markdown('<div class="alert-info">📸 Ambil foto → otomatis deteksi.</div>',unsafe_allow_html=True)
        _,col_btn=st.columns(2)
        with col_btn: btn_ocr_cam=st.button("📖 Baca Teks",key="ocr_cam",use_container_width=True)
        cam=st.camera_input("📸 Ambil foto",key="cam_tab1")
        if cam:
            if not st.session_state.model1 and not st.session_state.model2:
                st.error("⚠️ Load YOLO dulu!")
            else:
                from PIL import Image
                img=np.array(Image.open(cam))
                bgr=cv2.cvtColor(img,cv2.COLOR_RGB2BGR) if len(img.shape)==3 else img
                bgr=cv2.resize(bgr,(640,480)); orig=bgr.copy()
                ann,dets=process_frame_detection_multi(bgr,st.session_state.model1,st.session_state.model2,st.session_state.model3)
                frame_ph.image(cv2.cvtColor(ann,cv2.COLOR_BGR2RGB),caption=f"{len(dets)} objek",use_container_width=True)
                for d in dets: st.session_state.detection_history.append(d)
                st.session_state.detection_history=st.session_state.detection_history[-500:]
                dl,rl=handle_alerts(dets,time.time(),enable_audio,warn_ph,status_ph,audio_alert_ph,cooldown=alert_cooldown)
                m_det.metric("Objek",len(dets)); m_danger.metric("⚠️",len(dl))
                if btn_ocr_cam and st.session_state.ocr_engine:
                    text=perform_ocr_on_frame(orig,st.session_state.ocr_engine,ocr_min_conf)
                    if text and text!="Tidak ada teks terdeteksi" and len(text)>3:
                        ocr_ph.markdown(f'<div class="ocr-result">📝 {text}</div>',unsafe_allow_html=True)
                        if enable_tts:
                            a=get_audio_bytes(f"Ada tulisan: {text}")
                            if a: audio_ocr_ph.empty();time.sleep(0.3);play_audio_safe(audio_ocr_ph,a)
                    else:
                        ocr_ph.markdown('<div class="ocr-result" style="opacity:0.5">📝 Tidak ada teks</div>',unsafe_allow_html=True)
                if show_logs: log_exp.markdown(render_log(),unsafe_allow_html=True)
    else:
        uploaded=st.file_uploader("Upload video",type=['mp4','avi','mov','mkv','webm'])
        if uploaded:
            cn=uploaded.name
            if st.session_state.last_uploaded_name!=cn:
                st.session_state.ocr_triggered_vid=False;st.session_state.ocr_frame_count=0
                st.session_state.last_uploaded_name=cn;st.session_state.last_ocr_text=''
                st.session_state.detected_classes=set();st.session_state.active_objects=set()
                st.session_state.last_alert_time=defaultdict(lambda:-99.0)
                st.info("🔄 Video baru.")
        col1,col2=st.columns(2)
        with col1: btn_start=st.button("▶️ Start",use_container_width=True)
        with col2: btn_baca=st.button("📖 Baca Teks",use_container_width=True)
        if btn_baca:
            st.session_state.ocr_triggered_vid=True;st.session_state.ocr_frame_count=0
            st.session_state.last_ocr_text='';st.info("🔍 OCR aktif.")
        if uploaded and btn_start:
            if not st.session_state.model1 and not st.session_state.model2:
                st.error("⚠️ Load YOLO!");st.stop()
            st.session_state.detected_classes=set();st.session_state.active_objects=set()
            st.session_state.last_alert_time=defaultdict(lambda:-99.0)
            st.session_state.last_ocr_text=''
            suf=os.path.splitext(uploaded.name)[1] or '.mp4'
            with tempfile.NamedTemporaryFile(delete=False,suffix=suf) as tmp:
                tmp.write(uploaded.read());vp=tmp.name
            try:
                cap=cv2.VideoCapture(vp)
                if not cap.isOpened(): st.error("❌ Video gagal!");st.stop()
                tot=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps_v=cap.get(cv2.CAP_PROP_FPS) or 30.0
                m1,m2,m3=st.session_state.model1,st.session_state.model2,st.session_state.model3
                ocr=st.session_state.ocr_engine
                cnt=0;t0=time.time()
                prog=st.progress(0)
                # Hitung delay per frame agar sesuai kecepatan asli
                delay_per_frame = frame_skip / fps_v
                while True:
                    ok,fr=cap.read()
                    if not ok: break
                    cnt+=1
                    if cnt%frame_skip!=0: continue
                    frame_start=time.time()
                    fr=cv2.resize(fr,(640,480))
                    orig=fr.copy()
                    sec=cnt/fps_v
                    prog.progress(min(cnt/max(tot,1),1.0),text=f"Frame {cnt}/{tot} ({sec:.0f}s)")
                    ann,dets=process_frame_detection_multi(fr,m1,m2,m3)
                    frame_ph.image(cv2.cvtColor(ann,cv2.COLOR_BGR2RGB),use_container_width=True)
                    st.session_state.last_frame=orig
                    for d in dets: st.session_state.detection_history.append(d)
                    st.session_state.detection_history=st.session_state.detection_history[-500:]
                    dl,rl=handle_alerts(dets,time.time(),enable_audio,warn_ph,status_ph,audio_alert_ph,cooldown=alert_cooldown)
                    if st.session_state.ocr_triggered_vid and ocr:
                        st.session_state.ocr_frame_count+=1
                        if st.session_state.ocr_frame_count%ocr_scan_interval==0:
                            handle_ocr_dedup(orig,ocr,ocr_min_conf,enable_tts,ocr_ph,audio_ocr_ph)
                    m_det.metric("Deteksi",len(dets));m_danger.metric("⚠️",len(dl))
                    el=time.time()-t0
                    m_fps.metric("FPS",f"{cnt/el:.1f}" if el>0 else "0")
                    if show_logs: log_exp.markdown(render_log(),unsafe_allow_html=True)
                    # Delay agar video jalan sesuai kecepatan asli
                    proc_time=time.time()-frame_start
                    wait=max(delay_per_frame-proc_time,0.001)
                    time.sleep(wait)
                cap.release();prog.empty()
                st.success(f"✅ Selesai! {cnt} frame.")
                st.session_state.ocr_triggered_vid=False
            except Exception as e: st.error(f"Error: {e}")
            finally:
                try: os.unlink(vp)
                except: pass

with tab2:
    st.markdown("### 📖 Text Reading")
    mode2=st.radio("Input:",["📷 Capture","📤 Upload"],horizontal=True)
    ip,rp,ap=st.empty(),st.empty(),st.empty()
    def run_ocr_img(pil_img):
        arr=np.array(pil_img)
        if len(arr.shape)==3 and arr.shape[2]>=3: arr=cv2.cvtColor(arr[:,:,:3],cv2.COLOR_RGB2BGR)
        ip.image(pil_img,use_container_width=True)
        text=perform_ocr_on_frame(arr,st.session_state.ocr_engine,ocr_min_conf)
        rp.markdown(f'<div class="ocr-result">📝 {text}</div>',unsafe_allow_html=True)
        if text and text!="Tidak ada teks terdeteksi" and len(text)>3 and enable_tts:
            a=get_audio_bytes(f"Ada tulisan: {text}")
            if a: ap.empty();time.sleep(0.3);play_audio_safe(ap,a);st.success("🔊 Diputar")
    if mode2=="📷 Capture":
        cm=st.camera_input("📸",key="cam_tab2")
        if cm:
            if not st.session_state.ocr_engine: st.error("⚠️ Load OCR!")
            else:
                from PIL import Image; run_ocr_img(Image.open(cm))
    else:
        ui=st.file_uploader("Upload gambar",type=['jpg','jpeg','png','bmp'])
        if ui:
            if not st.session_state.ocr_engine: st.error("⚠️ Load OCR!")
            else:
                from PIL import Image; run_ocr_img(Image.open(ui))

with tab3:
    st.markdown("### 📊 Statistics")
    if st.session_state.detection_history:
        h=st.session_state.detection_history;t=len(h)
        dg=len([d for d in h if d['risk_level']=='BAHAYA'])
        w=len([d for d in h if d['risk_level']=='WASPADA'])
        am=len([d for d in h if d['risk_level']=='AMAN'])
        rm=len([d for d in h if d['risk_level']=='RAMBU'])
        c1,c2,c3,c4=st.columns(4)
        c1.markdown(f'<div class="stat-box"><div class="stat-value">{t}</div><div class="stat-label">Total</div></div>',unsafe_allow_html=True)
        c2.markdown(f'<div class="stat-box"><div class="stat-value" style="color:#ff4444">{dg}</div><div class="stat-label">Bahaya</div></div>',unsafe_allow_html=True)
        c3.markdown(f'<div class="stat-box"><div class="stat-value" style="color:#faad14">{w}</div><div class="stat-label">Waspada</div></div>',unsafe_allow_html=True)
        c4.markdown(f'<div class="stat-box"><div class="stat-value" style="color:#00c073">{am+rm}</div><div class="stat-label">Aman/Rambu</div></div>',unsafe_allow_html=True)
        df=pd.DataFrame([{
            'Waktu':d['timestamp'].strftime("%H:%M:%S"),'Objek':id_nama(d['class']),
            'Conf':f"{d['confidence']:.0%}",'Risk':d['risk_level'],
            'Model':d.get('source','?'),
            'Pos':'Kiri' if d['position_x']<0.35 else('Kanan' if d['position_x']>0.65 else'Depan'),
        } for d in h[-100:]])
        st.dataframe(df,use_container_width=True)
        if st.button("🗑️ Hapus"):
            st.session_state.detection_history=[];st.session_state.detected_classes=set();st.rerun()
    else: st.info("📊 Belum ada data.")

st.divider()
st.markdown('<div style="text-align:center;color:#999;font-size:.8rem;padding:1rem 0"><strong>v6.1</strong> • YOLOv11 • EasyOCR • gTTS</div>',unsafe_allow_html=True)
