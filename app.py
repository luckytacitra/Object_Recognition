# =====================================================================
# ASISTEN NAVIGASI TUNANETRA v6.1 — GABUNGAN v5.12 + COLAB
# FIX: Suara keluar, YOLO detect, video speed, OCR akurat
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

# ─── ID_MAP dari Colab SEL 2 ───
ID_MAP = {
    "person":"orang","car":"mobil","bus":"bus","truck":"truk",
    "motorcycle":"motor","bicycle":"sepeda","train":"kereta",
    "dog":"anjing","cat":"kucing","fire hydrant":"hidran",
    "stop sign":"rambu stop","traffic light":"lampu lalu lintas",
    "Animal":"hewan","Person":"orang","Vehicle":"kendaraan","Train":"kereta",
    "Traffic-light":"lampu lalu lintas",
    "Stairs":"tangga","Pothole":"lubang di jalan",
    "Over-bridge":"jembatan penyeberangan","Railway":"rel kereta",
    "Road-barrier":"pembatas jalan","Sidewalk":"trotoar",
    "Crosswalk":"jalur penyeberangan","Obstacle":"rintangan",
    "Pole":"tiang","Tree":"pohon","Traffic-sign":"rambu",
    "Dilarang Masuk":"dilarang masuk","Dilarang Parkir":"dilarang parkir",
    "Dilarang Berhenti":"dilarang berhenti","Dilarang Belok Kanan":"dilarang belok kanan",
    "Dilarang Putar Balik":"dilarang putar balik","Dilarang Mendahului":"dilarang mendahului",
    "Dilarang Berjalan Terus":"dilarang berjalan terus",
    "Hati-Hati":"rambu hati-hati","Gereja":"gereja",
    "Lampu Lalu Lintas":"lampu lalu lintas",
    "Larangan Kecepatan - 30km-jam":"batas kecepatan 30",
    "Larangan Kecepatan - 40km-jam":"batas kecepatan 40",
    "Masjid":"masjid","Rumah Sakit":"rumah sakit","SPBU":"pom bensin",
    "Tempat Parkir":"tempat parkir","Pemberhentian Bus":"pemberhentian bus",
    "Perintah Ikuti Bundaran":"ikuti arah bundaran",
    "Perintah Jalur Sepeda":"jalur sepeda","Perintah Lajur Kiri":"gunakan lajur kiri",
    "Persimpangan 3 Prioritas":"persimpangan tiga",
    "Persimpangan Empat":"persimpangan empat",
    "Putar Balik":"area putar balik","Banyak Anak-Anak":"area banyak anak-anak",
    "Balai Pertolongan Pertama":"balai pertolongan pertama",
}

# Lookup case-insensitive
ID_MAP_LOWER = {k.lower(): v for k, v in ID_MAP.items()}

def id_nama(n):
    if not isinstance(n, str): return str(n)
    nc = n.strip()
    if nc in ID_MAP: return ID_MAP[nc]
    nl = nc.lower()
    if nl in ID_MAP_LOWER: return ID_MAP_LOWER[nl]
    return nc

# Dari Colab SEL 5+7
KELAS_YOLO_RELEVAN = {
    "person","bicycle","car","motorcycle","bus","truck",
    "traffic light","stop sign","fire hydrant","dog","cat",
}
KELAS_BEST_DIPAKAI = {
    "Stairs","Pothole","Over-bridge","Railway",
    "Road-barrier","Sidewalk","Crosswalk","Obstacle","Pole",
    "Vehicle","Animal",
}
KELAS_BEST_LOWER = {k.lower() for k in KELAS_BEST_DIPAKAI}

# Dari Colab SEL 11
FATAL_LOWER = {"pothole","obstacle","stairs","road-barrier","pole"}
CONF_MIN_FATAL = {"pothole":0.40,"obstacle":0.45,"stairs":0.35,"road-barrier":0.35,"pole":0.35}

RAMBU_KW = {'rambu','lampu lalu lintas','stop sign','dilarang','hati-hati',
            'rumah sakit','masjid','gereja','spbu','pom bensin','parkir',
            'persimpangan','bundaran','jalur','kecepatan','berhenti'}

def is_rambu(n): return any(k in n.lower() for k in RAMBU_KW)

# ─── SESSION ───
_def = {
    'model1':None,'model2':None,'model3':None,'ocr_engine':None,
    'last_frame':None,'log':[],'detection_history':[],
    'last_alert_time':defaultdict(lambda:-99.0),
    'detected_classes':set(),'active_objects':set(),
    'ocr_triggered_vid':False,'ocr_frame_count':0,
    'last_uploaded_name':None,'last_ocr_text':'',
    'audio_queue':[],
}
for k,v in _def.items():
    if k not in st.session_state: st.session_state[k]=v

def add_log(m):
    st.session_state.log.insert(0,(time.strftime("%H:%M:%S"),str(m)))
    st.session_state.log=st.session_state.log[:30]

def render_log():
    if not st.session_state.log:
        return '<div style="color:#ccc;text-align:center;padding:1rem">Belum ada aktivitas</div>'
    rows=[]
    for item in st.session_state.log[:10]:
        if isinstance(item,(tuple,list)) and len(item)>=2:
            rows.append(f'[{item[0]}] {item[1]}')
    return '<br>'.join(rows) if rows else 'Belum ada'

# ─── AUDIO — FIX: multiple audio elements + user interaction trigger ───
def play_audio_safe(placeholder, audio_bytes):
    """FIX: Pakai multiple approach agar suara keluar"""
    if not audio_bytes: return False
    b64 = base64.b64encode(audio_bytes).decode()
    uid = f"a_{int(time.time()*1000)}_{random.randint(1000,9999)}"
    # Approach: autoplay + JS play() + onclick fallback
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
                            el.play();
                            document.removeEventListener('click',handler);
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
    except: return None

# ─── ALERT MESSAGES ───
def generate_alert(name, px, ar):
    nm=id_nama(name)
    if px<0.35: pos,arah="di kiri","Geser ke kanan"
    elif px>0.65: pos,arah="di kanan","Geser ke kiri"
    else: pos,arah="di depan","Berhenti"
    if ar>0.12: return f"Awas! {nm} sangat dekat {pos}. {arah} sekarang!"
    elif ar>0.04: return f"Awas! Ada {nm} {pos}. {arah}!"
    else: return f"Hati-hati, ada {nm} {pos}. {arah} pelan-pelan."

def generate_rambu_alert(name):
    n=name.lower()
    m={'stop sign':'Rambu stop. Berhenti!','rambu stop':'Rambu stop. Berhenti!',
       'dilarang masuk':'Dilarang masuk!','hati-hati':'Hati-hati!',
       'lampu lalu lintas':'Ada lampu lalu lintas','rumah sakit':'Ada rumah sakit',
       'masjid':'Ada masjid','gereja':'Ada gereja','pom bensin':'Ada pom bensin',
       'persimpangan':'Ada persimpangan'}
    for k,v in m.items():
        if k in n: return v
    return f"Ada {id_nama(name)}"

# ─── OCR dari v5.12 (sudah bagus) ───
def perform_ocr(frame, engine, min_conf=0.30):
    if engine is None: return ""
    try:
        if len(frame.shape)==3: gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        else: gray=frame
        h,w=gray.shape
        if w<800:
            s=800/w; gray=cv2.resize(gray,(800,int(h*s)),interpolation=cv2.INTER_CUBIC)
        denoised=cv2.fastNlMeansDenoising(gray,None,h=10)
        clahe=cv2.createCLAHE(clipLimit=2.5,tileGridSize=(8,8))
        enhanced=clahe.apply(denoised)
        results=None
        try: results=engine.readtext(enhanced,detail=1,paragraph=False,text_threshold=0.5)
        except: pass
        if not results:
            try: results=engine.readtext(gray,detail=1,paragraph=False)
            except: pass
        if results:
            texts=[]
            for (bbox,txt,conf) in results:
                txt=txt.strip()
                if len(txt)>=2 and conf>=min_conf: texts.append(txt)
            if texts:
                raw=' '.join(texts)
                raw=re.sub(r'[^a-zA-Z0-9\s\.,!?\-:/()%]','',raw)
                # Autocorrect umum
                fixes={'dlarang':'dilarang','dlrang':'dilarang','parkr':'parkir',
                       'msuk':'masuk','brhenti':'berhenti','hti-hti':'hati-hati',
                       'msjid':'masjid','grja':'gereja','rmah':'rumah','lmpu':'lampu'}
                for k,v in fixes.items(): raw=raw.replace(k,v)
                if len(raw)>=3: return raw.title()
        return ""
    except: return ""

# ─── DETEKSI — conf dari Colab, filter dari v5.12 ───
def detect_frame(frame, model, conf, source):
    """
    source='m1': YOLO umum, filter KELAS_YOLO_RELEVAN, TIDAK PERNAH BAHAYA
    source='m2': best.pt, filter KELAS_BEST_DIPAKAI, BISA BAHAYA (fatal only)
    source='m3': rambu, semua class = RAMBU
    """
    if model is None: return frame, []
    try:
        results=model.predict(frame,conf=conf,iou=0.45,verbose=False)
        dets=[]
        if not results or results[0].boxes is None: return frame,dets
        r=results[0]; fh,fw=frame.shape[:2]
        for box,cs,ci in zip(r.boxes.xyxy.cpu().numpy(),
                              r.boxes.conf.cpu().numpy(),
                              r.boxes.cls.cpu().numpy().astype(int)):
            cn=r.names[ci]; cl=cn.lower()
            # Filter berdasarkan source
            if source=='m1':
                if cl not in KELAS_YOLO_RELEVAN: continue
            elif source=='m2':
                if cl not in KELAS_BEST_LOWER: continue
            # source=='m3': semua class OK

            x1,y1,x2,y2=map(int,box)
            ar=(x2-x1)*(y2-y1)/(fw*fh)
            px=((x1+x2)/2)/fw
            cy=(y1+y2)/2
            if ar<0.002: continue
            if source=='m2' and (cy/fh)>0.9: continue  # Colab BATAS_BAWAH_REL

            # Risk level
            if source=='m2':
                is_fatal=cl in FATAL_LOWER
                min_c=CONF_MIN_FATAL.get(cl,0.30)
                if is_fatal and cs>=min_c and ar>=0.03: rl='BAHAYA'
                elif is_fatal and cs>=min_c*0.7 and ar>=0.01: rl='WASPADA'
                else: rl='AMAN'
            elif source=='m3':
                rl='RAMBU'
            else:
                rl='AMAN'  # M1 never BAHAYA

            dets.append({'class':cn,'confidence':float(cs),'area_ratio':ar,
                'position_x':px,'risk_level':rl,'bbox':(x1,y1,x2,y2),
                'timestamp':datetime.now(),'source':source})

            clr=(0,0,255) if rl=='BAHAYA' else (0,165,255) if rl=='WASPADA' else (255,165,0) if rl=='RAMBU' else (0,255,0)
            cv2.rectangle(frame,(x1,y1),(x2,y2),clr,2)
            lbl=f"{id_nama(cn)} {cs:.2f}"
            cv2.putText(frame,lbl,(x1,y1-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,clr,2)
        return frame,dets
    except Exception as e:
        logger.error(f"Det err: {e}"); return frame,[]

def detect_all(frame,m1,m2,m3):
    out=frame.copy(); all_d=[]
    # Conf dari Colab: M1=0.30, M2=0.10 (v5.12 effective), M3=0.70
    out,d=detect_frame(out,m1,0.30,'m1'); all_d.extend(d)
    out,d=detect_frame(out,m2,0.10,'m2'); all_d.extend(d)
    out,d=detect_frame(out,m3,0.70,'m3'); all_d.extend(d)
    return out,all_d

# ─── ALERT — dari v5.12 (sekali per objek baru) ───
def handle_alerts(dets, en_audio, warn_ph, status_ph, audio_ph, cooldown=2):
    danger=[d for d in dets if d['risk_level']=='BAHAYA']
    waspada=[d for d in dets if d['risk_level']=='WASPADA']
    rambu=[d for d in dets if d['risk_level']=='RAMBU']
    now=time.time()

    current={d['class'] for d in dets}
    # Reset classes yang sudah hilang
    gone=st.session_state.detected_classes - current
    st.session_state.detected_classes -= gone

    new_danger=[d for d in danger if d['class'] not in st.session_state.detected_classes]
    new_waspada=[d for d in waspada if d['class'] not in st.session_state.detected_classes]
    new_rambu=[d for d in rambu if d['class'] not in st.session_state.detected_classes]

    if new_danger and en_audio:
        d=new_danger[0]
        msg=generate_alert(d['class'],d['position_x'],d['area_ratio'])
        warn_ph.markdown(f'<div class="alert-danger">🚨 {msg}</div>',unsafe_allow_html=True)
        a=get_audio_bytes(msg)
        if a: audio_ph.empty(); play_audio_safe(audio_ph,a); add_log(f"🔊 {msg}")
        st.session_state.detected_classes.add(d['class'])
        status_ph.markdown('<div class="pills"><span class="pill pill-run">● AKTIF</span><span class="pill pill-danger">● BAHAYA</span></div>',unsafe_allow_html=True)
    elif new_waspada and en_audio:
        d=new_waspada[0]
        msg=generate_alert(d['class'],d['position_x'],d['area_ratio'])
        warn_ph.markdown(f'<div class="alert-warning">⚡ {msg}</div>',unsafe_allow_html=True)
        a=get_audio_bytes(msg)
        if a: audio_ph.empty(); play_audio_safe(audio_ph,a); add_log(f"🔊 {msg}")
        st.session_state.detected_classes.add(d['class'])
        status_ph.markdown('<div class="pills"><span class="pill pill-run">● AKTIF</span><span class="pill pill-danger" style="background:#fffbe6;color:#ad6800;border-color:#ffe58f">● WASPADA</span></div>',unsafe_allow_html=True)
    elif new_rambu and en_audio:
        d=new_rambu[0]
        msg=generate_rambu_alert(d['class'])
        warn_ph.markdown(f'<div class="alert-info">ℹ️ {msg}</div>',unsafe_allow_html=True)
        a=get_audio_bytes(msg)
        if a: audio_ph.empty(); play_audio_safe(audio_ph,a); add_log(f"🔊 {msg}")
        st.session_state.detected_classes.add(d['class'])
        status_ph.markdown('<div class="pills"><span class="pill pill-run">● AKTIF</span><span class="pill pill-ocr">● RAMBU</span></div>',unsafe_allow_html=True)
    else:
        if danger:
            warn_ph.markdown(f'<div class="alert-danger">🚨 Masih ada {id_nama(danger[0]["class"])}!</div>',unsafe_allow_html=True)
            status_ph.markdown('<div class="pills"><span class="pill pill-run">● AKTIF</span><span class="pill pill-danger">● BAHAYA</span></div>',unsafe_allow_html=True)
        else:
            warn_ph.markdown('<div class="alert-success">✅ Jalur aman</div>',unsafe_allow_html=True)
            status_ph.markdown('<div class="pills"><span class="pill pill-run">● AKTIF</span><span class="pill pill-ok">● AMAN</span></div>',unsafe_allow_html=True)
    return danger,rambu

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

# ─── UI ───
c1,c2=st.columns([0.1,0.9])
with c1: st.markdown('<div class="header-logo">👁️</div>',unsafe_allow_html=True)
with c2: st.markdown('<div class="header-text"><h1>Asisten Navigasi Tunanetra</h1><p>Deteksi Objek & Teks — v6.1</p></div>',unsafe_allow_html=True)
st.divider()

with st.sidebar:
    st.markdown("### ⚙️ Pengaturan")
    if st.button("📥 Load YOLO",use_container_width=True):
        with st.spinner("Loading..."): st.session_state.model1=load_yolo(); st.success("✅ YOLO!")
    if st.button("📥 Load OCR",use_container_width=True):
        with st.spinner("Loading..."): st.session_state.ocr_engine=load_ocr(); st.success("✅ OCR!")
    st.markdown("---")
    with st.expander("🏔️ M2: best.pt (Lubang/Tangga)"):
        up2=st.file_uploader("Upload best.pt",type=['pt'],key='m2up')
        if up2 and st.button("Load M2"):
            with st.spinner("Memuat..."):
                with tempfile.NamedTemporaryFile(delete=False,suffix='.pt') as tmp: tmp.write(up2.read())
                from ultralytics import YOLO; st.session_state.model2=YOLO(tmp.name); st.success("✅ M2!")
    with st.expander("🚦 M3: best_rambu.pt"):
        up3=st.file_uploader("Upload best_rambu.pt",type=['pt'],key='m3up')
        if up3 and st.button("Load M3"):
            with st.spinner("Memuat..."):
                with tempfile.NamedTemporaryFile(delete=False,suffix='.pt') as tmp: tmp.write(up3.read())
                from ultralytics import YOLO; st.session_state.model3=YOLO(tmp.name); st.success("✅ M3!")
    st.markdown("---")
    s1="✅" if st.session_state.model1 else "⚠️"
    s2="✅" if st.session_state.model2 else "⚠️"
    s3="✅" if st.session_state.model3 else "⚠️"
    so="✅" if st.session_state.ocr_engine else "⚠️"
    st.markdown(f'<div class="pills"><span class="pill pill-model">{s1} M1</span><span class="pill pill-model">{s2} M2</span><span class="pill pill-model">{s3} M3</span><span class="pill pill-model">{so} OCR</span></div>',unsafe_allow_html=True)
    enable_audio=st.checkbox("🔊 Audio",value=True)
    alert_cooldown=st.slider("Cooldown (s)",1,5,2)
    ocr_min_conf=st.slider("OCR Conf",0.1,0.9,0.30,0.05)
    ocr_scan_interval=st.slider("OCR Interval (frame)",1,30,10)
    frame_skip=st.slider("Frame Skip",1,10,2)
    enable_tts=st.checkbox("🔊 TTS",value=True)
    show_logs=st.checkbox("📋 Logs",value=True)
    if st.button("🔁 Reset Suara"):
        st.session_state.detected_classes=set()
        st.session_state.last_ocr_text=''
        st.success("Reset!")

tab1,tab2,tab3=st.tabs(["🎯 Detection","📖 Text Reading","📊 Statistics"])

with tab1:
    mode=st.radio("Mode:",["📹 Webcam (Foto)","📤 Upload Video"],horizontal=True)
    st.divider()
    frame_ph=st.empty();status_ph=st.empty();warn_ph=st.empty();ocr_ph=st.empty()
    audio_alert_ph=st.empty();audio_ocr_ph=st.empty()
    c1,c2,c3=st.columns(3)
    with c1: m_det=st.empty()
    with c2: m_danger=st.empty()
    with c3: m_fps=st.empty()
    if show_logs: log_exp=st.expander("📋 Logs")

    if mode=="📹 Webcam (Foto)":
        st.markdown('<div class="alert-info">📸 Ambil foto → otomatis deteksi. Klik "Baca Teks" untuk OCR.</div>',unsafe_allow_html=True)
        _,cb=st.columns(2)
        with cb: btn_ocr_cam=st.button("📖 Baca Teks",key="ocr_cam",use_container_width=True)
        cam=st.camera_input("📸 Ambil foto",key="cam_tab1")
        if cam:
            if not st.session_state.model1 and not st.session_state.model2:
                st.error("⚠️ Load YOLO dulu!")
            else:
                from PIL import Image
                img=np.array(Image.open(cam))
                bgr=cv2.cvtColor(img,cv2.COLOR_RGB2BGR) if len(img.shape)==3 else img
                bgr=cv2.resize(bgr,(640,480)); orig=bgr.copy()
                ann,dets=detect_all(bgr,st.session_state.model1,st.session_state.model2,st.session_state.model3)
                frame_ph.image(cv2.cvtColor(ann,cv2.COLOR_BGR2RGB),caption=f"{len(dets)} objek",use_container_width=True)
                for d in dets: st.session_state.detection_history.append(d)
                st.session_state.detection_history=st.session_state.detection_history[-500:]
                dl,rl=handle_alerts(dets,enable_audio,warn_ph,status_ph,audio_alert_ph,alert_cooldown)
                m_det.metric("Objek",len(dets));m_danger.metric("⚠️",len(dl))
                if btn_ocr_cam and st.session_state.ocr_engine:
                    text=perform_ocr(orig,st.session_state.ocr_engine,ocr_min_conf)
                    if text:
                        ocr_ph.markdown(f'<div class="ocr-result">📝 {text}</div>',unsafe_allow_html=True)
                        if enable_tts:
                            a=get_audio_bytes(f"Ada tulisan: {text}")
                            if a: audio_ocr_ph.empty();play_audio_safe(audio_ocr_ph,a)
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
        col1,col2=st.columns(2)
        with col1: btn_start=st.button("▶️ Start Detection",use_container_width=True)
        with col2: btn_baca=st.button("📖 Baca Teks",use_container_width=True)
        if btn_baca:
            st.session_state.ocr_triggered_vid=True;st.session_state.ocr_frame_count=0
            st.session_state.last_ocr_text='';st.info("🔍 OCR aktif.")

        if uploaded and btn_start:
            if not st.session_state.model1 and not st.session_state.model2:
                st.error("⚠️ Load YOLO!");st.stop()
            st.session_state.detected_classes=set()
            with tempfile.NamedTemporaryFile(delete=False,suffix='.mp4') as tmp:
                tmp.write(uploaded.read());vp=tmp.name
            try:
                cap=cv2.VideoCapture(vp)
                if not cap.isOpened(): st.error("❌ Video gagal!");st.stop()
                tot=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fpsv=cap.get(cv2.CAP_PROP_FPS) or 30.0
                m1,m2,m3=st.session_state.model1,st.session_state.model2,st.session_state.model3
                ocr_eng=st.session_state.ocr_engine
                cnt=0;t0=time.time()
                prog=st.progress(0)
                while True:
                    ok,fr=cap.read()
                    if not ok: break
                    cnt+=1
                    if cnt%frame_skip!=0: continue
                    fr=cv2.resize(fr,(640,480)); orig=fr.copy()
                    sec=cnt/fpsv
                    prog.progress(min(cnt/max(tot,1),1.0),text=f"Frame {cnt}/{tot} ({sec:.1f}s)")
                    ann,dets=detect_all(fr,m1,m2,m3)
                    frame_ph.image(cv2.cvtColor(ann,cv2.COLOR_BGR2RGB),use_container_width=True)
                    st.session_state.last_frame=orig
                    for d in dets: st.session_state.detection_history.append(d)
                    st.session_state.detection_history=st.session_state.detection_history[-500:]
                    dl,rl=handle_alerts(dets,enable_audio,warn_ph,status_ph,audio_alert_ph,alert_cooldown)
                    # OCR
                    if st.session_state.ocr_triggered_vid and ocr_eng:
                        st.session_state.ocr_frame_count+=1
                        if st.session_state.ocr_frame_count%ocr_scan_interval==0:
                            text=perform_ocr(orig,ocr_eng,ocr_min_conf)
                            if text and text!=st.session_state.last_ocr_text:
                                st.session_state.last_ocr_text=text
                                ocr_ph.markdown(f'<div class="ocr-result">📝 {text}</div>',unsafe_allow_html=True)
                                add_log(f"📖 {text[:30]}")
                                if enable_tts:
                                    a=get_audio_bytes(f"Ada tulisan: {text}")
                                    if a: audio_ocr_ph.empty();play_audio_safe(audio_ocr_ph,a)
                    m_det.metric("Deteksi",len(dets));m_danger.metric("⚠️",len(dl))
                    el=time.time()-t0
                    m_fps.metric("FPS",f"{cnt/el:.1f}" if el>0 else "0")
                    if show_logs: log_exp.markdown(render_log(),unsafe_allow_html=True)
                    # Delay agar video tidak terlalu cepat
                    time.sleep(max(frame_skip/fpsv - 0.05, 0.01))
                cap.release();prog.empty()
                st.success(f"✅ Selesai! {cnt} frame.")
                st.session_state.ocr_triggered_vid=False
            except Exception as e: st.error(f"Error: {e}")
            finally:
                try: os.unlink(vp)
                except: pass

with tab2:
    st.markdown("### 📖 Text Reading")
    mode2=st.radio("Input:",["📷 Kamera","📤 Upload"],horizontal=True)
    ip,rp,ap=st.empty(),st.empty(),st.empty()
    def run_ocr(pil_img):
        arr=np.array(pil_img)
        if len(arr.shape)==3 and arr.shape[2]>=3:
            arr=cv2.cvtColor(arr[:,:,:3],cv2.COLOR_RGB2BGR)
        ip.image(pil_img,use_container_width=True)
        text=perform_ocr(arr,st.session_state.ocr_engine,ocr_min_conf)
        if text:
            rp.markdown(f'<div class="ocr-result">📝 {text}</div>',unsafe_allow_html=True)
            if enable_tts:
                a=get_audio_bytes(f"Ada tulisan: {text}")
                if a: ap.empty();play_audio_safe(ap,a)
        else:
            rp.markdown('<div class="ocr-result" style="opacity:0.5">📝 Tidak ada teks</div>',unsafe_allow_html=True)
    if mode2=="📷 Kamera":
        cm=st.camera_input("📸",key="cam_tab2")
        if cm:
            if not st.session_state.ocr_engine: st.error("⚠️ Load OCR!")
            else:
                from PIL import Image; run_ocr(Image.open(cm))
    else:
        ui=st.file_uploader("Upload gambar",type=['jpg','jpeg','png','bmp'])
        if ui:
            if not st.session_state.ocr_engine: st.error("⚠️ Load OCR!")
            else:
                from PIL import Image; run_ocr(Image.open(ui))

with tab3:
    st.markdown("### 📊 Statistics")
    if st.session_state.detection_history:
        h=st.session_state.detection_history
        t=len(h);dg=len([d for d in h if d['risk_level']=='BAHAYA'])
        w=len([d for d in h if d['risk_level']=='WASPADA'])
        am=len([d for d in h if d['risk_level']=='AMAN'])
        rm=len([d for d in h if d['risk_level']=='RAMBU'])
        c1,c2,c3,c4=st.columns(4)
        c1.markdown(f'<div class="stat-box"><div class="stat-value">{t}</div><div class="stat-label">Total</div></div>',unsafe_allow_html=True)
        c2.markdown(f'<div class="stat-box"><div class="stat-value" style="color:#ff4444">{dg}</div><div class="stat-label">Bahaya</div></div>',unsafe_allow_html=True)
        c3.markdown(f'<div class="stat-box"><div class="stat-value" style="color:#ffaa00">{w}</div><div class="stat-label">Waspada</div></div>',unsafe_allow_html=True)
        c4.markdown(f'<div class="stat-box"><div class="stat-value" style="color:#00c073">{am+rm}</div><div class="stat-label">Aman/Rambu</div></div>',unsafe_allow_html=True)
        df=pd.DataFrame([{'Waktu':d['timestamp'].strftime("%H:%M:%S"),'Objek':id_nama(d['class']),
            'Conf':f"{d['confidence']:.1%}",'Risk':d['risk_level'],'Model':d.get('source','?'),
            'Pos':'Kiri' if d['position_x']<0.35 else('Kanan' if d['position_x']>0.65 else'Depan'),
        } for d in h[-100:]])
        st.dataframe(df,use_container_width=True)
        if st.button("🗑️ Hapus"):
            st.session_state.detection_history=[];st.session_state.detected_classes=set();st.rerun()
    else: st.info("📊 Belum ada data.")

st.divider()
st.markdown('<div style="text-align:center;color:#999;font-size:.8rem;padding:1rem 0"><strong>Asisten Navigasi Tunanetra v6.1</strong> • YOLOv11 • EasyOCR • gTTS</div>',unsafe_allow_html=True)
