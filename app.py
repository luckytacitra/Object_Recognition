# =====================================================================
# ASISTEN NAVIGASI TUNANETRA v5.1
# =====================================================================
# LOGIKA BARU:
# - Semua model deteksi AKTIF otomatis (M1, M2, M3)
# - BAHAYA = HANYA dari M2 (lubang, tangga, rintangan)
# - Orang, mobil, objek biasa = INFO saja, BUKAN bahaya
# - Rambu (M3) = peringatan info
# - OCR = HANYA jika tombol "Baca Teks" diklik
# - OCR ambil hasil confidence TERTINGGI saja
# - Video: skip banyak frame, resize kecil, cepat
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
.header-logo{width:50px;height:50px;background:linear-gradient(135deg,#6c3fff,#3f8bff);border-radius:14px;display:flex;align-items:center;justify-content:center;font-size:1.4rem;box-shadow:0 4px 15px rgba(108,63,255,.3)}
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

# ─── SESSION ───
_defaults = {
    'model1':None,'model2':None,'model3':None,'ocr_engine':None,
    'last_frame':None,'log':[],'detection_history':[],
    'last_alert_time':defaultdict(lambda:-99.0),
    'alerted_objects':set(),
    'ocr_triggered':False,'ocr_frame_count':0,
    'last_uploaded_name':None,'last_ocr_text':'',
    'm2_classes':set(),  # Track class names dari M2
}
for k,v in _defaults.items():
    if k not in st.session_state: st.session_state[k]=v

def add_log(m):
    st.session_state.log.insert(0,(time.strftime("%H:%M:%S"),m))
    st.session_state.log=st.session_state.log[:30]

# ─── AUDIO ───
def play_audio(ph, ab):
    if ab:
        b=base64.b64encode(ab).decode()
        uid=f"a_{int(time.time()*1000)}_{random.randint(1000,9999)}"
        ph.markdown(f'<audio autoplay style="display:none" id="{uid}"><source src="data:audio/mp3;base64,{b}" type="audio/mp3"></audio><script>setTimeout(function(){{document.getElementById("{uid}").play()}},100)</script>',unsafe_allow_html=True)

def tts(text,lang='id'):
    try:
        buf=BytesIO(); gTTS(text=text,lang=lang,slow=False).write_to_fp(buf); buf.seek(0); return buf.read()
    except: return None

# ─── NAMA INDO ───
INDO={
    'person':'orang','car':'mobil','bus':'bus','truck':'truk',
    'motorcycle':'motor','bicycle':'sepeda','dog':'anjing','cat':'kucing',
    'pothole':'lubang jalan','stairs':'tangga','obstacle':'rintangan',
    'road-barrier':'pembatas jalan','pole':'tiang','train':'kereta api',
    'stop sign':'rambu stop','traffic light':'lampu lalu lintas',
    'sidewalk':'trotoar','crosswalk':'zebra cross','tree':'pohon',
    'dilarang masuk':'dilarang masuk','dilarang parkir':'dilarang parkir',
    'hati-hati':'hati-hati','rumah sakit':'rumah sakit',
    'masjid':'masjid','gereja':'gereja','pom bensin':'pom bensin',
    'persimpangan':'persimpangan','batas kecepatan':'batas kecepatan',
}
def indo(n):
    l=n.lower()
    if l in INDO: return INDO[l]
    for k,v in INDO.items():
        if k in l or l in k: return v
    return l

RAMBU_KW={'rambu','lampu lalu lintas','traffic light','stop sign','dilarang',
           'hati-hati','rumah sakit','masjid','gereja','pom bensin','parkir',
           'persimpangan','bundaran','kecepatan','berhenti'}
def is_rambu(n): return any(k in n.lower() for k in RAMBU_KW)

def alert_msg(name,px,ar):
    nm=indo(name)
    if px<0.3: p,a="di kiri","Geser ke kanan"
    elif px>0.7: p,a="di kanan","Geser ke kiri"
    else: p,a="di depan","Berhenti, belok ke samping"
    if ar>0.12: return f"Awas! {nm} sangat dekat {p}! {a} sekarang!"
    elif ar>0.05: return f"Hati-hati, {nm} {p}. {a}."
    else: return f"Ada {nm} {p}."

def rambu_msg(name):
    n=name.lower()
    m={'stop sign':'Rambu stop. Berhenti!','rambu stop':'Rambu stop. Berhenti!',
       'dilarang masuk':'Dilarang masuk!','hati-hati':'Hati-hati!',
       'lampu lalu lintas':'Ada lampu lalu lintas'}
    for k,v in m.items():
        if k in n: return v
    return f"Ada {indo(n)}"

# ============================================================
# OCR — AMBIL CONFIDENCE TERTINGGI SAJA
# ============================================================
def ocr_read(frame, engine, min_conf=0.15):
    """
    Baca teks — HANYA ambil hasil dengan confidence tertinggi.
    Tidak gabung semua hasil. Lebih akurat, kurang typo.
    """
    if engine is None: return ""
    try:
        if len(frame.shape)==3: gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        else: gray=frame
        h,w=gray.shape
        if w<500:
            s=640/w; gray=cv2.resize(gray,(640,int(h*s)),interpolation=cv2.INTER_CUBIC)
        
        clahe=cv2.createCLAHE(clipLimit=2.5,tileGridSize=(8,8))
        enh=clahe.apply(gray)
        kern=np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
        sharp=cv2.filter2D(enh,-1,kern)

        # Coba beberapa preprocessing — paragraph=False agar teks kecil kedetect
        best_texts = []  # (text, conf)
        for img in [sharp, enh, frame]:
            try:
                res=engine.readtext(img, detail=1, paragraph=False)
                if res:
                    for (bbox,txt,conf) in res:
                        txt=txt.strip()
                        if len(txt)>=1 and conf>min_conf:
                            best_texts.append((txt, conf))
            except: continue
            if best_texts: break

        if not best_texts:
            return ""
        
        # Sort by confidence descending
        best_texts.sort(key=lambda x:x[1], reverse=True)
        
        # Ambil hasil — prioritaskan confidence tinggi, skip duplikat
        final_parts = []
        for txt, conf in best_texts:
            cleaned = re.sub(r'[^\w\s\.,!?\-:/()%]', '', txt, flags=re.UNICODE).strip()
            if len(cleaned) < 1: continue
            # Skip single char kecuali angka
            if len(cleaned)==1 and not cleaned.isdigit(): continue
            is_dup = False
            for ex in final_parts:
                if _similar(cleaned, ex, 0.6):
                    is_dup=True; break
            if not is_dup:
                final_parts.append(cleaned)
        
        if final_parts:
            result = ' '.join(final_parts)
            return ' '.join(result.split())
        return ""
    except Exception as e:
        logger.error(f"OCR err: {e}")
        return ""

def _similar(a, b, thr=0.6):
    a,b=a.lower(),b.lower()
    if a==b: return True
    if len(a)<3 or len(b)<3: return a==b
    sa=set(a[i:i+2] for i in range(len(a)-1))
    sb=set(b[i:i+2] for i in range(len(b)-1))
    if not sa or not sb: return False
    return len(sa&sb)/len(sa|sb) >= thr

# ============================================================
# DETEKSI
# ============================================================
    # Class YOLO yang RELEVAN untuk tunanetra (dari COCO dataset)
    # Sisanya (trotoar, bench, laptop, dll) di-SKIP
M1_RELEVANT = {
    0:'person', 1:'bicycle', 2:'car', 3:'motorcycle', 5:'bus',
    7:'truck', 9:'traffic light', 11:'stop sign', 13:'bench',
    15:'cat', 16:'dog', 17:'horse',
}
# Class ID yang relevan
M1_RELEVANT_IDS = set(M1_RELEVANT.keys())

def detect_single(frame, model, conf=0.35, source_model='m1'):
    """
    - m1: HANYA orang/kendaraan/hewan → AMAN (info saja)
    - m2: semua class dari model → BISA BAHAYA
    - m3: semua class dari model → RAMBU (info)
    """
    if model is None: return frame, []
    try:
        # M1 pakai conf rendah supaya orang kedetect
        use_conf = max(conf - 0.05, 0.20) if source_model == 'm1' else conf
        # M2 obstacle lebih sensitif
        if source_model == 'm2': use_conf = max(conf - 0.10, 0.15)
        
        results = model.predict(frame, conf=use_conf, iou=0.45, verbose=False)
        dets = []
        if results and results[0].boxes is not None and len(results[0].boxes)>0:
            r=results[0]
            boxes=r.boxes.xyxy.cpu().numpy()
            confs=r.boxes.conf.cpu().numpy()
            clses=r.boxes.cls.cpu().numpy().astype(int)
            fh,fw=frame.shape[:2]
            
            for box,cs,ci in zip(boxes,confs,clses):
                x1,y1,x2,y2=map(int,box)
                cn=r.names[ci]
                ar=(x2-x1)*(y2-y1)/(fw*fh)
                px=((x1+x2)/2)/fw
                
                # ── M1: FILTER HANYA CLASS RELEVAN ──
                if source_model == 'm1':
                    if ci not in M1_RELEVANT_IDS:
                        continue  # Skip trotoar, bench, dll
                    if ar < 0.01: continue  # Terlalu kecil
                
                # M2/M3: tampilkan semua class dari model custom
                if source_model in ('m2','m3'):
                    if ar < 0.005: continue
                
                # ── RISK LEVEL ──
                if source_model == 'm2':
                    # M2 = obstacle → BISA BAHAYA
                    if cs > 0.35 and ar > 0.03:
                        rl = 'BAHAYA'
                    elif cs > 0.20 and ar > 0.01:
                        rl = 'WASPADA'
                    else:
                        rl = 'AMAN'
                elif source_model == 'm3':
                    rl = 'RAMBU'
                else:
                    # M1 = TIDAK PERNAH BAHAYA
                    rl = 'AMAN'
                
                dets.append({
                    'class':cn,'confidence':float(cs),'area_ratio':ar,
                    'position_x':px,'risk_level':rl,'bbox':(x1,y1,x2,y2),
                    'timestamp':datetime.now(),'source':source_model
                })

                # Draw
                if rl=='BAHAYA': clr=(0,0,255)
                elif rl=='WASPADA': clr=(0,165,255)
                elif rl=='RAMBU': clr=(255,165,0)
                else: clr=(0,255,0)
                cv2.rectangle(frame,(x1,y1),(x2,y2),clr,2)
                lbl=f"{indo(cn)} {cs:.0%}"
                f2=cv2.FONT_HERSHEY_SIMPLEX
                (tw,th),_=cv2.getTextSize(lbl,f2,0.5,2)
                cv2.rectangle(frame,(x1,y1-th-6),(x1+tw+4,y1),clr,-1)
                cv2.putText(frame,lbl,(x1+2,y1-3),f2,0.5,(255,255,255),2)
        return frame, dets
    except Exception as e:
        logger.error(f"Det err: {e}")
        return frame, []

def detect_all(frame, m1, m2, m3, conf=0.35):
    """Jalankan semua model berurutan"""
    out=frame.copy()
    all_d=[]
    if m1:
        out,d=detect_single(out,m1,conf,'m1'); all_d.extend(d)
    if m2:
        # M2 obstacle: conf lebih rendah agar lubang terdeteksi
        out,d=detect_single(out,m2,max(conf-0.10,0.15),'m2'); all_d.extend(d)
    if m3:
        out,d=detect_single(out,m3,conf,'m3'); all_d.extend(d)
    return out, all_d

# ─── ALERT ───
def do_alerts(dets, en_audio, cooldown, w_ph, s_ph, ad_ph, ar_ph):
    bahaya=[d for d in dets if d['risk_level']=='BAHAYA']
    waspada=[d for d in dets if d['risk_level']=='WASPADA']
    rambu=[d for d in dets if d['risk_level']=='RAMBU']
    
    cur={d['class'] for d in dets}
    st.session_state.alerted_objects -= (st.session_state.alerted_objects - cur)
    now=time.time()

    if bahaya and en_audio:
        d=bahaya[0]; k=d['class']
        if k not in st.session_state.alerted_objects:
            msg=alert_msg(d['class'],d['position_x'],d['area_ratio'])
            w_ph.markdown(f'<div class="alert-danger">🚨 {msg}</div>',unsafe_allow_html=True)
            a=tts(msg)
            if a: ad_ph.empty(); play_audio(ad_ph,a)
            st.session_state.alerted_objects.add(k)
            add_log(f"BAHAYA: {indo(k)}")
        else:
            w_ph.markdown(f'<div class="alert-danger">🚨 Masih ada {indo(d["class"])}!</div>',unsafe_allow_html=True)
        s_ph.markdown('<div class="pills"><span class="pill pill-run">● AKTIF</span><span class="pill pill-danger">● BAHAYA</span></div>',unsafe_allow_html=True)
    elif waspada and en_audio:
        d=waspada[0]; k=f"w_{d['class']}"
        if (now-st.session_state.last_alert_time[k])>=cooldown:
            msg=alert_msg(d['class'],d['position_x'],d['area_ratio'])
            w_ph.markdown(f'<div class="alert-warning">⚡ {msg}</div>',unsafe_allow_html=True)
            a=tts(msg)
            if a: ad_ph.empty(); play_audio(ad_ph,a)
            st.session_state.last_alert_time[k]=now
        s_ph.markdown('<div class="pills"><span class="pill pill-run">● AKTIF</span><span class="pill pill-danger" style="background:#fffbe6;color:#ad6800;border-color:#ffe58f">● WASPADA</span></div>',unsafe_allow_html=True)
    elif rambu and en_audio:
        d=rambu[0]; k=f"r_{d['class']}"
        if (now-st.session_state.last_alert_time[k])>=cooldown:
            msg=rambu_msg(d['class'])
            w_ph.markdown(f'<div class="alert-info">ℹ️ {msg}</div>',unsafe_allow_html=True)
            a=tts(msg)
            if a: ar_ph.empty(); play_audio(ar_ph,a)
            st.session_state.last_alert_time[k]=now
        s_ph.markdown('<div class="pills"><span class="pill pill-run">● AKTIF</span><span class="pill pill-ocr">● RAMBU</span></div>',unsafe_allow_html=True)
    else:
        w_ph.markdown('<div class="alert-success">✅ Jalur aman</div>',unsafe_allow_html=True)
        s_ph.markdown('<div class="pills"><span class="pill pill-run">● AKTIF</span><span class="pill pill-ok">● AMAN</span></div>',unsafe_allow_html=True)
    return bahaya, rambu

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
with c2: st.markdown('<div class="header-text"><h1>Asisten Navigasi Tunanetra</h1><p>Deteksi Objek + Baca Teks — v5.1</p></div>',unsafe_allow_html=True)
st.divider()

# ─── SIDEBAR ───
with st.sidebar:
    st.markdown("### ⚙️ Pengaturan")
    if st.button("📥 Load YOLO",use_container_width=True):
        with st.spinner("Loading..."):
            st.session_state.model1=load_yolo(); add_log("YOLO loaded"); st.success("✅ YOLO!")
    if st.button("📥 Load OCR",use_container_width=True):
        with st.spinner("Loading..."):
            st.session_state.ocr_engine=load_ocr(); add_log("OCR loaded"); st.success("✅ OCR!")
    st.markdown("---")
    with st.expander("🏔️ M2: Lubang/Tangga/Rintangan"):
        st.caption("Upload model yang sudah kamu training untuk deteksi lubang, tangga, tiang, dll.")
        up2=st.file_uploader("Upload best.pt",type=['pt'],key='m2up')
        if up2 and st.button("Load M2"):
            with st.spinner("Memuat M2..."):
                with tempfile.NamedTemporaryFile(delete=False,suffix='.pt') as tmp:
                    tmp.write(up2.read())
                from ultralytics import YOLO
                m2=YOLO(tmp.name)
                st.session_state.model2=m2
                # Tampilkan class names dari model
                names=m2.names if hasattr(m2,'names') else {}
                st.session_state.m2_classes=set(names.values()) if names else set()
                add_log(f"M2 loaded: {list(st.session_state.m2_classes)[:5]}")
                st.success(f"✅ M2! Classes: {list(st.session_state.m2_classes)}")
    with st.expander("🚦 M3: Rambu"):
        up3=st.file_uploader("Upload best_rambu.pt",type=['pt'],key='m3up')
        if up3 and st.button("Load M3"):
            with st.spinner("Memuat M3..."):
                with tempfile.NamedTemporaryFile(delete=False,suffix='.pt') as tmp:
                    tmp.write(up3.read())
                from ultralytics import YOLO
                st.session_state.model3=YOLO(tmp.name)
                add_log("M3 loaded"); st.success("✅ M3!")
    st.markdown("---")
    s1="✅" if st.session_state.model1 else "⚠️"
    s2="✅" if st.session_state.model2 else "⚠️"
    s3="✅" if st.session_state.model3 else "⚠️"
    so="✅" if st.session_state.ocr_engine else "⚠️"
    st.markdown(f'<div class="pills"><span class="pill pill-model">{s1} M1</span><span class="pill pill-model">{s2} M2</span><span class="pill pill-model">{s3} M3</span><span class="pill pill-model">{so} OCR</span></div>',unsafe_allow_html=True)
    if st.session_state.m2_classes:
        st.caption(f"M2 classes: {', '.join(st.session_state.m2_classes)}")
    
    conf_threshold=st.slider("Confidence",0.1,0.9,0.30,0.05)
    enable_audio=st.checkbox("🔊 Audio",value=True)
    alert_cooldown=st.slider("Cooldown (s)",2,10,5)
    ocr_min_conf=st.slider("OCR Conf",0.1,0.9,0.25,0.05)
    enable_tts=st.checkbox("🔊 TTS",value=True)
    show_logs=st.checkbox("📋 Logs",value=False)
    frame_skip=st.slider("Frame Skip",2,15,5,help="Makin tinggi = makin cepat tapi kurang detail")

# ─── TABS ───
tab1,tab2,tab3=st.tabs(["🎯 Detection","📖 Text Reading","📊 Statistics"])

with tab1:
    mode=st.radio("Mode:",["📹 Webcam","📤 Upload Video"],horizontal=True)
    st.divider()

    if mode=="📹 Webcam":
        st.markdown('<div class="alert-info">📸 Ambil foto → otomatis dianalisis objek & bahaya. Klik "Baca Teks" untuk scan tulisan.</div>',unsafe_allow_html=True)
        
        fp=st.empty(); sp=st.empty(); wp=st.empty(); op=st.empty()
        adp=st.empty(); arp=st.empty(); aocp=st.empty()
        c1,c2,c3=st.columns(3)
        with c1: md=st.empty()
        with c2: mb=st.empty()
        with c3: mm=st.empty()
        if show_logs: lp=st.expander("📋").empty()

        col1,col2=st.columns(2)
        with col2: btn_ocr_cam=st.button("📖 Baca Teks",key="ocr_cam",use_container_width=True)

        cam=st.camera_input("📸 Ambil foto")
        if cam:
            if not st.session_state.model1 and not st.session_state.model2:
                st.error("⚠️ Load YOLO dulu!")
            else:
                from PIL import Image
                img=np.array(Image.open(cam))
                bgr=cv2.cvtColor(img,cv2.COLOR_RGB2BGR) if len(img.shape)==3 else img
                bgr=cv2.resize(bgr,(640,480)); orig=bgr.copy()
                ann,dets=detect_all(bgr,st.session_state.model1,st.session_state.model2,st.session_state.model3,conf_threshold)
                fp.image(cv2.cvtColor(ann,cv2.COLOR_BGR2RGB),caption=f"{len(dets)} objek",use_container_width=True)
                for d in dets: st.session_state.detection_history.append(d)
                st.session_state.detection_history=st.session_state.detection_history[-500:]
                dl,rl=do_alerts(dets,enable_audio,alert_cooldown,wp,sp,adp,arp)
                md.metric("Objek",len(dets)); mb.metric("⚠️ Bahaya",len(dl)); mm.metric("Mode","📸")
                
                # OCR hanya jika tombol diklik
                if btn_ocr_cam and st.session_state.ocr_engine:
                    text=ocr_read(orig,st.session_state.ocr_engine,ocr_min_conf)
                    if text and len(text)>3:
                        op.markdown(f'<div class="ocr-result">📝 {text}</div>',unsafe_allow_html=True)
                        if enable_tts:
                            a=tts(f"Ada tulisan: {text}")
                            if a: aocp.empty(); play_audio(aocp,a)
                    else:
                        op.markdown('<div class="ocr-result" style="opacity:0.5">📝 Tidak ada teks terdeteksi</div>',unsafe_allow_html=True)
                
                if show_logs: lp.markdown('<br>'.join([f'[{t}] {m}' for t,m in st.session_state.log[:10]]),unsafe_allow_html=True)

    else:
        uploaded=st.file_uploader("Upload video",type=['mp4','avi','mov','mkv','webm'])
        fp=st.empty(); sp=st.empty(); wp=st.empty(); op=st.empty()
        adp=st.empty(); arp=st.empty(); aocp=st.empty()
        c1,c2,c3=st.columns(3)
        with c1: md=st.empty()
        with c2: mb=st.empty()
        with c3: mf=st.empty()
        if show_logs: lp=st.expander("📋").empty()

        if uploaded:
            cn=uploaded.name
            if st.session_state.last_uploaded_name!=cn:
                st.session_state.ocr_triggered=False; st.session_state.ocr_frame_count=0
                st.session_state.last_uploaded_name=cn; st.session_state.last_ocr_text=''
                st.session_state.alerted_objects.clear()

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
            suf=os.path.splitext(uploaded.name)[1] or '.mp4'
            with tempfile.NamedTemporaryFile(delete=False,suffix=suf) as tmp:
                tmp.write(uploaded.read()); vp=tmp.name
            try:
                cap=cv2.VideoCapture(vp)
                if not cap.isOpened():
                    st.error("❌ Video gagal dibuka! Coba MP4 H.264."); st.stop()
                tot=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fpsv=cap.get(cv2.CAP_PROP_FPS) or 30.0
                m1,m2,m3=st.session_state.model1,st.session_state.model2,st.session_state.model3
                ocr=st.session_state.ocr_engine
                cnt=0; t0=time.time()
                prog=st.progress(0)
                while True:
                    ok,fr=cap.read()
                    if not ok: break
                    cnt+=1
                    if cnt%frame_skip!=0: continue
                    
                    # Resize kecil = cepat
                    fr=cv2.resize(fr,(416,312))
                    orig=fr.copy()
                    sec=cnt/fpsv
                    prog.progress(min(cnt/max(tot,1),1.0),text=f"{cnt}/{tot} ({sec:.0f}s)")
                    
                    ann,dets=detect_all(fr,m1,m2,m3,conf_threshold)
                    fp.image(cv2.cvtColor(ann,cv2.COLOR_BGR2RGB),use_container_width=True)
                    st.session_state.last_frame=orig
                    for d in dets: st.session_state.detection_history.append(d)
                    st.session_state.detection_history=st.session_state.detection_history[-500:]
                    
                    dl,rl=do_alerts(dets,enable_audio,alert_cooldown,wp,sp,adp,arp)
                    
                    # OCR hanya jika diklik + interval
                    if st.session_state.ocr_triggered and ocr:
                        st.session_state.ocr_frame_count+=1
                        if st.session_state.ocr_frame_count%max(ocr_min_conf*10,3)==0:
                            text=ocr_read(orig,ocr,ocr_min_conf)
                            if text and len(text)>3:
                                if not _similar(text,st.session_state.last_ocr_text):
                                    st.session_state.last_ocr_text=text
                                    op.markdown(f'<div class="ocr-result">📝 {text}</div>',unsafe_allow_html=True)
                                    add_log(f"OCR: {text[:30]}")
                                    if enable_tts:
                                        a=tts(f"Ada tulisan: {text}")
                                        if a: aocp.empty(); play_audio(aocp,a)
                    
                    md.metric("Objek",len(dets)); mb.metric("⚠️",len(dl))
                    el=time.time()-t0
                    mf.metric("FPS",f"{cnt/el:.1f}" if el>0 else "0")
                    if show_logs: lp.markdown('<br>'.join([f'[{t}] {m}' for t,m in st.session_state.log[:10]]),unsafe_allow_html=True)
                
                cap.release(); prog.empty()
                st.success(f"✅ Selesai! {cnt} frame.")
                st.session_state.ocr_triggered=False
            except Exception as e:
                st.error(f"Error: {e}")
            finally:
                try: os.unlink(vp)
                except: pass

with tab2:
    st.markdown("### 📖 Text Reading")
    mode2=st.radio("Input:",["📷 Kamera","📤 Upload"],horizontal=True)
    ip,rp,ap=st.empty(),st.empty(),st.empty()
    if mode2=="📷 Kamera":
        cm=st.camera_input("📸 Foto")
        if cm:
            if not st.session_state.ocr_engine: st.error("⚠️ Load OCR!")
            else:
                from PIL import Image
                arr=np.array(Image.open(cm))
                ip.image(arr,use_container_width=True)
                text=ocr_read(arr,st.session_state.ocr_engine,ocr_min_conf)
                if text and len(text)>3:
                    rp.markdown(f'<div class="ocr-result">📝 {text}</div>',unsafe_allow_html=True)
                    if enable_tts:
                        a=tts(f"Ada tulisan: {text}")
                        if a: ap.empty(); play_audio(ap,a)
                else:
                    rp.markdown('<div class="ocr-result" style="opacity:0.5">📝 Tidak ada teks</div>',unsafe_allow_html=True)
    else:
        ui=st.file_uploader("Upload gambar",type=['jpg','jpeg','png','bmp'])
        if ui:
            from PIL import Image
            img=Image.open(ui); arr=np.array(img)
            ip.image(img,use_container_width=True)
            if not st.session_state.ocr_engine: st.error("⚠️ Load OCR!")
            else:
                text=ocr_read(arr,st.session_state.ocr_engine,ocr_min_conf)
                if text and len(text)>3:
                    rp.markdown(f'<div class="ocr-result">📝 {text}</div>',unsafe_allow_html=True)
                    if enable_tts:
                        a=tts(f"Ada tulisan: {text}")
                        if a: ap.empty(); play_audio(ap,a)
                else:
                    rp.markdown('<div class="ocr-result" style="opacity:0.5">📝 Tidak ada teks</div>',unsafe_allow_html=True)

with tab3:
    st.markdown("### 📊 Statistics")
    if st.session_state.detection_history:
        h=st.session_state.detection_history
        t=len(h); dg=len([d for d in h if d['risk_level']=='BAHAYA'])
        w=len([d for d in h if d['risk_level']=='WASPADA'])
        am=len([d for d in h if d['risk_level']=='AMAN'])
        c1,c2,c3,c4=st.columns(4)
        c1.markdown(f'<div class="stat-box"><div class="stat-value">{t}</div><div class="stat-label">Total</div></div>',unsafe_allow_html=True)
        c2.markdown(f'<div class="stat-box"><div class="stat-value" style="color:#ff4444">{dg}</div><div class="stat-label">Bahaya</div></div>',unsafe_allow_html=True)
        c3.markdown(f'<div class="stat-box"><div class="stat-value" style="color:#faad14">{w}</div><div class="stat-label">Waspada</div></div>',unsafe_allow_html=True)
        c4.markdown(f'<div class="stat-box"><div class="stat-value" style="color:#00c073">{am}</div><div class="stat-label">Aman</div></div>',unsafe_allow_html=True)
        df=pd.DataFrame([{
            'Waktu':d['timestamp'].strftime("%H:%M:%S"),
            'Objek':indo(d['class']),
            'Conf':f"{d['confidence']:.0%}",
            'Risk':d['risk_level'],
            'Model':d.get('source','?'),
            'Pos':'Kiri' if d['position_x']<0.3 else('Kanan' if d['position_x']>0.7 else'Depan'),
        } for d in h[-100:]])
        st.dataframe(df,use_container_width=True)
        if st.button("🗑️ Hapus"):
            st.session_state.detection_history=[]; st.session_state.alerted_objects.clear(); st.rerun()
    else:
        st.info("📊 Belum ada data.")

st.divider()
st.markdown('<div style="text-align:center;color:#999;font-size:.8rem;padding:1rem 0"><strong>Asisten Navigasi Tunanetra v5.1</strong> • YOLOv11 • EasyOCR • gTTS</div>',unsafe_allow_html=True)
