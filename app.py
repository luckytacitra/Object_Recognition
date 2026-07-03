# =====================================================================
# ASISTEN NAVIGASI TUNANETRA v6.0 — STREAMLIT
# Disesuaikan dari Colab notebook Fiks_Indo_Video_2.ipynb
# =====================================================================

import streamlit as st
import cv2, os, re, time, base64, tempfile
import numpy as np
import pandas as pd
from gtts import gTTS
from datetime import datetime
from io import BytesIO
from collections import defaultdict, Counter
from difflib import get_close_matches
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

# =====================================================================
# ID_MAP — DARI COLAB (LENGKAP)
# =====================================================================
ID_MAP = {
    # COCO (model1)
    "person":"orang","car":"mobil","bus":"bus","truck":"truk",
    "motorcycle":"motor","bicycle":"sepeda","train":"kereta",
    "dog":"anjing","cat":"kucing","fire hydrant":"hidran",
    "stop sign":"rambu stop","traffic light":"lampu lalu lintas",
    # Model2 (SafeWalkBD) — huruf kapital
    "Animal":"hewan","Person":"orang","Vehicle":"kendaraan","Train":"kereta",
    "Traffic-light":"lampu lalu lintas",
    "Stairs":"tangga","Pothole":"lubang di jalan","Over-bridge":"jembatan penyeberangan",
    "Railway":"rel kereta","Road-barrier":"pembatas jalan","Sidewalk":"trotoar",
    "Crosswalk":"jalur penyeberangan","Obstacle":"rintangan","Pole":"tiang",
    "Tree":"pohon","Traffic-sign":"rambu",
    # Model3 (rambu Indonesia)
    "Balai Pertolongan Pertama":"balai pertolongan pertama",
    "Banyak Anak-Anak":"area banyak anak-anak",
    "Dilarang Belok Kanan":"dilarang belok kanan",
    "Dilarang Berhenti":"dilarang berhenti",
    "Dilarang Berjalan Terus":"dilarang berjalan terus",
    "Dilarang Masuk":"dilarang masuk",
    "Dilarang Mendahului":"dilarang mendahului",
    "Dilarang Parkir":"dilarang parkir",
    "Dilarang Putar Balik":"dilarang putar balik",
    "Gereja":"gereja","Hati-Hati":"rambu hati-hati",
    "Jalur Penyebrangan":"jalur penyeberangan",
    "Lampu Lalu Lintas":"lampu lalu lintas",
    "Larangan Kecepatan - 30km-jam":"batas kecepatan 30 km per jam",
    "Larangan Kecepatan - 40km-jam":"batas kecepatan 40 km per jam",
    "Larangan Kendaraan MST - 10 Ton":"batas muatan 10 ton",
    "Masjid":"masjid","Pemberhentian Bus":"pemberhentian bus",
    "Perintah Ikuti Bundaran":"ikuti arah bundaran",
    "Perintah Jalur Sepeda":"jalur sepeda",
    "Perintah Lajur Kiri":"gunakan lajur kiri",
    "Perintah Pilih Satu Jalur":"pilih satu jalur",
    "Persimpangan 3 Prioritas":"persimpangan tiga",
    "Persimpangan 3 Sisi Kanan Prioritas":"persimpangan tiga kanan prioritas",
    "Persimpangan 3 Sisi Kiri Prioritas":"persimpangan tiga kiri prioritas",
    "Persimpangan Empat":"persimpangan empat",
    "Putar Balik":"area putar balik","Rumah Sakit":"rumah sakit",
    "SPBU":"pom bensin","Tempat Parkir":"tempat parkir",
}

def id_nama(n):
    if not isinstance(n, str): return str(n)
    nc = n.strip()
    for k, v in ID_MAP.items():
        if k.lower() == nc.lower(): return v
    return nc.lower().replace("-", " ")

# Class M2 yang DIPAKAI — dari Colab
KELAS_BEST_DIPAKAI = {
    "Stairs","Pothole","Over-bridge","Railway",
    "Road-barrier","Sidewalk","Crosswalk","Obstacle","Pole",
    "Vehicle","Animal",
}

# Class COCO relevan — dari Colab
KELAS_YOLO_RELEVAN = {
    "person","bicycle","car","motorcycle","bus","truck",
    "traffic light","stop sign","fire hydrant","dog","cat",
}

# Objek FATAL — dari Colab SEL 11
FATAL_OBJECTS = {"pothole","obstacle","stairs","road-barrier","pole"}

# Conf minimum per kelas fatal — dari Colab SEL 11
CONF_MIN_FATAL = {
    "pothole":0.40,"obstacle":0.45,"stairs":0.35,
    "road-barrier":0.35,"pole":0.35,
}

# ─── SESSION ───
_defaults = {
    'model1':None,'model2':None,'model3':None,'ocr_engine':None,
    'last_frame':None,'log':[],'detection_history':[],
    'last_alert_time':defaultdict(lambda:-99.0),
    'alerted_objects':set(),
    'ocr_triggered':False,'ocr_frame_count':0,
    'last_uploaded_name':None,'last_ocr_text':'',
    'm2_classes':set(),
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

# =====================================================================
# OCR — DARI COLAB SEL 12 (multi-variant + filter noise + autocorrect)
# =====================================================================
KATA_DASAR = {
    'kiri','kanan','depan','belakang','atas','bawah',
    'jalan','trotoar','lubang','tangga','rambu','lampu',
    'truk','mobil','motor','sepeda','bus','kereta',
    'rumah','toko','masjid','gereja','sekolah','kantor',
    'parkir','berhenti','hati','awas','bahaya','waktu',
    'indonesia','jakarta','bandung','surabaya','medan',
    'rintangan','pembatas','penyeberangan','jembatan',
    'zona','inspirasi','photobox','toilet','mushola',
    'keluar','masuk','lantai','lift','eskalator',
    'informasi','loket','tiket','antrian','kasir',
    'dilarang','merokok','push','pull','tarik','dorong',
    'emergency','exit','pintu','darurat',
}

def buat_variant_frame(frame):
    h,w = frame.shape[:2]
    variants = {}
    variants['original'] = frame.copy()
    variants['upscale_2x'] = cv2.resize(frame,(w*2,h*2),interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY) if len(frame.shape)==3 else frame
    clahe = cv2.createCLAHE(clipLimit=2.5,tileGridSize=(8,8))
    enh = clahe.apply(gray)
    variants['clahe'] = cv2.cvtColor(enh,cv2.COLOR_GRAY2BGR)
    kern = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
    variants['sharp'] = cv2.filter2D(frame,-1,kern)
    return variants

def filter_noise_ocr(teks):
    if not teks or len(teks)<3: return None
    teks = re.sub(r'[^a-zA-Z0-9\s\.\-\']',' ',teks)
    kata_kata = teks.split()
    kata_valid = []
    for kata in kata_kata:
        if len(kata)<2: continue
        huruf = sum(1 for c in kata if c.isalpha())
        if huruf==0: continue
        if huruf/len(kata)<0.4: continue
        if len(set(kata.lower()))==1 and len(kata)>3: continue
        kata_valid.append(kata)
    if len(kata_valid)<1: return None
    hasil = ' '.join(kata_valid)
    # Autocorrect
    words = hasil.split()
    corrected = []
    for w in words:
        matches = get_close_matches(w.lower(), KATA_DASAR, n=1, cutoff=0.7)
        if matches and matches[0]!=w.lower():
            corrected.append(matches[0].capitalize() if w[0].isupper() else matches[0])
        else:
            corrected.append(w)
    return ' '.join(corrected)

def ocr_read(frame, engine, min_conf=0.25):
    """OCR dari Colab SEL 12 — multi-variant, filter noise, ambil terbaik"""
    if engine is None: return ""
    try:
        variants = buat_variant_frame(frame)
        semua = []
        for nama, img in variants.items():
            try:
                res = engine.readtext(img, detail=1, paragraph=False)
                if res:
                    for (bbox,txt,conf) in res:
                        txt = txt.strip()
                        if len(txt)>=2 and conf>min_conf:
                            semua.append((txt,conf,nama))
            except: continue
            # Stop jika sudah dapat hasil bagus
            if any(c>0.7 for _,c,_ in semua): break

        if not semua: return ""

        # Deduplikasi
        seen = set()
        unik = []
        for txt,conf,var in sorted(semua,key=lambda x:x[1],reverse=True):
            key = txt.lower().strip()
            if key not in seen:
                seen.add(key)
                unik.append((txt,conf))

        # Ambil top 5 by confidence
        top = unik[:5]
        gabung = ' '.join([t for t,c in top])

        # Filter noise + autocorrect
        hasil = filter_noise_ocr(gabung)
        if hasil and len(hasil)>2:
            return hasil.title()
        return ""
    except Exception as e:
        logger.error(f"OCR err: {e}")
        return ""

def _similar(a,b,thr=0.6):
    a,b=a.lower(),b.lower()
    if a==b: return True
    if len(a)<3 or len(b)<3: return a==b
    sa=set(a[i:i+2] for i in range(len(a)-1))
    sb=set(b[i:i+2] for i in range(len(b)-1))
    if not sa or not sb: return False
    return len(sa&sb)/len(sa|sb)>=thr

# =====================================================================
# DETEKSI — DARI COLAB SEL 7 + SEL 11
# =====================================================================
def detect_m1(frame, model, conf=0.30):
    """Model 1: YOLO umum — HANYA class relevan, conf dari Colab"""
    if model is None: return frame, []
    try:
        results = model.predict(frame,conf=conf,iou=0.5,verbose=False)
        dets = []
        if results and results[0].boxes is not None:
            r=results[0]
            for box,cs,ci in zip(r.boxes.xyxy.cpu().numpy(),
                                  r.boxes.conf.cpu().numpy(),
                                  r.boxes.cls.cpu().numpy().astype(int)):
                cn=r.names[ci]
                if cn not in KELAS_YOLO_RELEVAN: continue
                x1,y1,x2,y2=map(int,box)
                fh,fw=frame.shape[:2]
                ar=(x2-x1)*(y2-y1)/(fw*fh)
                px=((x1+x2)/2)/fw
                if ar<0.01: continue
                # M1: TIDAK PERNAH BAHAYA — dari Colab logika
                rl='AMAN'
                dets.append({'class':cn,'confidence':float(cs),'area_ratio':ar,
                    'position_x':px,'risk_level':rl,'bbox':(x1,y1,x2,y2),
                    'timestamp':datetime.now(),'source':'yolo11'})
                cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
                lbl=f"{id_nama(cn)} {cs:.0%}"
                f2=cv2.FONT_HERSHEY_SIMPLEX
                (tw,th),_=cv2.getTextSize(lbl,f2,0.5,2)
                cv2.rectangle(frame,(x1,y1-th-6),(x1+tw+4,y1),(0,255,0),-1)
                cv2.putText(frame,lbl,(x1+2,y1-3),f2,0.5,(255,255,255),2)
        return frame, dets
    except: return frame, []

def detect_m2(frame, model, conf=0.25):
    """Model 2: best.pt — obstacle. conf=0.25 dari Colab"""
    if model is None: return frame, []
    try:
        results = model.predict(frame,conf=conf,iou=0.45,verbose=False)
        dets = []
        if results and results[0].boxes is not None:
            r=results[0]
            for box,cs,ci in zip(r.boxes.xyxy.cpu().numpy(),
                                  r.boxes.conf.cpu().numpy(),
                                  r.boxes.cls.cpu().numpy().astype(int)):
                cn=r.names[ci]
                if cn not in KELAS_BEST_DIPAKAI: continue
                x1,y1,x2,y2=map(int,box)
                fh,fw=frame.shape[:2]
                ar=(x2-x1)*(y2-y1)/(fw*fh)
                px=((x1+x2)/2)/fw
                cy=(y1+y2)/2
                # Skip objek di bawah 90% frame (noise) — dari Colab BATAS_BAWAH_REL
                if (cy/fh)>0.9: continue
                if ar<0.005: continue
                # BAHAYA logic dari Colab SEL 11
                cn_lower = cn.lower()
                is_fatal = cn_lower in FATAL_OBJECTS or cn in FATAL_OBJECTS
                min_c = CONF_MIN_FATAL.get(cn_lower, 0.30)
                if is_fatal and cs>=min_c and ar>=0.03:
                    rl='BAHAYA'
                elif is_fatal and cs>=min_c*0.7:
                    rl='WASPADA'
                else:
                    rl='AMAN'
                dets.append({'class':cn,'confidence':float(cs),'area_ratio':ar,
                    'position_x':px,'risk_level':rl,'bbox':(x1,y1,x2,y2),
                    'timestamp':datetime.now(),'source':'best'})
                clr=(0,0,255) if rl=='BAHAYA' else (0,165,255) if rl=='WASPADA' else (0,200,0)
                cv2.rectangle(frame,(x1,y1),(x2,y2),clr,2)
                lbl=f"{id_nama(cn)} {cs:.0%}"
                f2=cv2.FONT_HERSHEY_SIMPLEX
                (tw,th),_=cv2.getTextSize(lbl,f2,0.5,2)
                cv2.rectangle(frame,(x1,y1-th-6),(x1+tw+4,y1),clr,-1)
                cv2.putText(frame,lbl,(x1+2,y1-3),f2,0.5,(255,255,255),2)
        return frame, dets
    except: return frame, []

def detect_m3(frame, model, conf=0.70):
    """Model 3: rambu — conf=0.70 dari Colab"""
    if model is None: return frame, []
    try:
        results = model.predict(frame,conf=conf,iou=0.45,verbose=False)
        dets = []
        if results and results[0].boxes is not None:
            r=results[0]
            for box,cs,ci in zip(r.boxes.xyxy.cpu().numpy(),
                                  r.boxes.conf.cpu().numpy(),
                                  r.boxes.cls.cpu().numpy().astype(int)):
                cn=r.names[ci]
                x1,y1,x2,y2=map(int,box)
                fh,fw=frame.shape[:2]
                ar=(x2-x1)*(y2-y1)/(fw*fh)
                px=((x1+x2)/2)/fw
                dets.append({'class':cn,'confidence':float(cs),'area_ratio':ar,
                    'position_x':px,'risk_level':'RAMBU','bbox':(x1,y1,x2,y2),
                    'timestamp':datetime.now(),'source':'rambu'})
                cv2.rectangle(frame,(x1,y1),(x2,y2),(255,165,0),2)
                lbl=f"{id_nama(cn)} {cs:.0%}"
                f2=cv2.FONT_HERSHEY_SIMPLEX
                (tw,th),_=cv2.getTextSize(lbl,f2,0.5,2)
                cv2.rectangle(frame,(x1,y1-th-6),(x1+tw+4,y1),(255,165,0),-1)
                cv2.putText(frame,lbl,(x1+2,y1-3),f2,0.5,(255,255,255),2)
        return frame, dets
    except: return frame, []

def detect_all(frame, m1, m2, m3):
    out=frame.copy()
    all_d=[]
    out,d=detect_m1(out,m1,0.30); all_d.extend(d)
    out,d=detect_m2(out,m2,0.25); all_d.extend(d)
    out,d=detect_m3(out,m3,0.70); all_d.extend(d)
    return out, all_d

# ─── ALERT — dari Colab SEL 11 logic ───
def zona(px):
    if px<0.2: return "kiri"
    if px>0.8: return "kanan"
    return "tengah"

def instruksi_dari_dets(dets):
    """Logika instruksi dari Colab SEL 9"""
    if not dets: return "JALAN TERUS"
    fatal = [d for d in dets if d['class'].lower() in FATAL_OBJECTS
             and d['risk_level'] in ('BAHAYA','WASPADA')]
    if not fatal: return "JALAN TERUS"
    tengah = [d for d in fatal if zona(d['position_x'])=='tengah']
    kiri = [d for d in fatal if zona(d['position_x'])=='kiri']
    kanan = [d for d in fatal if zona(d['position_x'])=='kanan']
    if tengah:
        if kiri and kanan: return "BERHENTI"
        if kiri: return "GESER KE KANAN"
        if kanan: return "GESER KE KIRI"
        return "GESER KE KIRI"
    return "JALAN TERUS"

def do_alerts(dets, en_audio, cooldown, w_ph, s_ph, ad_ph, ar_ph):
    bahaya=[d for d in dets if d['risk_level']=='BAHAYA']
    waspada=[d for d in dets if d['risk_level']=='WASPADA']
    rambu=[d for d in dets if d['risk_level']=='RAMBU']
    cur={d['class'] for d in dets}
    st.session_state.alerted_objects -= (st.session_state.alerted_objects - cur)
    now=time.time()

    if bahaya and en_audio:
        d=bahaya[0]; k=d['class']
        aksi = instruksi_dari_dets(dets)
        nama = id_nama(d['class'])
        if k not in st.session_state.alerted_objects:
            msg=f"Awas! Ada {nama}. {aksi}."
            w_ph.markdown(f'<div class="alert-danger">🚨 {msg}</div>',unsafe_allow_html=True)
            a=tts(msg)
            if a: ad_ph.empty(); play_audio(ad_ph,a)
            st.session_state.alerted_objects.add(k)
            add_log(f"BAHAYA: {nama}")
        else:
            w_ph.markdown(f'<div class="alert-danger">🚨 Masih ada {nama}!</div>',unsafe_allow_html=True)
        s_ph.markdown('<div class="pills"><span class="pill pill-run">● AKTIF</span><span class="pill pill-danger">● BAHAYA</span></div>',unsafe_allow_html=True)
    elif waspada and en_audio:
        d=waspada[0]; k=f"w_{d['class']}"
        if (now-st.session_state.last_alert_time[k])>=cooldown:
            nama=id_nama(d['class'])
            aksi=instruksi_dari_dets(dets)
            msg=f"Hati-hati, ada {nama}. {aksi}."
            w_ph.markdown(f'<div class="alert-warning">⚡ {msg}</div>',unsafe_allow_html=True)
            a=tts(msg)
            if a: ad_ph.empty(); play_audio(ad_ph,a)
            st.session_state.last_alert_time[k]=now
        s_ph.markdown('<div class="pills"><span class="pill pill-run">● AKTIF</span><span class="pill pill-danger" style="background:#fffbe6;color:#ad6800;border-color:#ffe58f">● WASPADA</span></div>',unsafe_allow_html=True)
    elif rambu and en_audio:
        d=rambu[0]; k=f"r_{d['class']}"
        if (now-st.session_state.last_alert_time[k])>=cooldown:
            nama=id_nama(d['class'])
            msg=f"Info: Ada {nama}."
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
with c2: st.markdown('<div class="header-text"><h1>Asisten Navigasi Tunanetra</h1><p>Deteksi Objek + Baca Teks — v6.0 (Colab-synced)</p></div>',unsafe_allow_html=True)
st.divider()

# ─── SIDEBAR ───
with st.sidebar:
    st.markdown("### ⚙️ Pengaturan")
    if st.button("📥 Load YOLO",use_container_width=True):
        with st.spinner("Loading..."): st.session_state.model1=load_yolo(); add_log("YOLO loaded"); st.success("✅ YOLO!")
    if st.button("📥 Load OCR",use_container_width=True):
        with st.spinner("Loading..."): st.session_state.ocr_engine=load_ocr(); add_log("OCR loaded"); st.success("✅ OCR!")
    st.markdown("---")
    with st.expander("🏔️ M2: best.pt (Lubang/Tangga)"):
        st.caption("Model SafeWalkBD: Stairs, Pothole, Obstacle, dll")
        up2=st.file_uploader("Upload best.pt",type=['pt'],key='m2up')
        if up2 and st.button("Load M2"):
            with st.spinner("Memuat M2..."):
                with tempfile.NamedTemporaryFile(delete=False,suffix='.pt') as tmp: tmp.write(up2.read())
                from ultralytics import YOLO
                m2=YOLO(tmp.name); st.session_state.model2=m2
                names=m2.names if hasattr(m2,'names') else {}
                st.session_state.m2_classes=set(names.values())
                add_log(f"M2: {list(st.session_state.m2_classes)[:8]}")
                st.success(f"✅ M2! {len(names)} classes")
    with st.expander("🚦 M3: best_rambu.pt"):
        up3=st.file_uploader("Upload best_rambu.pt",type=['pt'],key='m3up')
        if up3 and st.button("Load M3"):
            with st.spinner("Memuat M3..."):
                with tempfile.NamedTemporaryFile(delete=False,suffix='.pt') as tmp: tmp.write(up3.read())
                from ultralytics import YOLO; st.session_state.model3=YOLO(tmp.name)
                add_log("M3 loaded"); st.success("✅ M3!")
    st.markdown("---")
    s1="✅" if st.session_state.model1 else "⚠️"
    s2="✅" if st.session_state.model2 else "⚠️"
    s3="✅" if st.session_state.model3 else "⚠️"
    so="✅" if st.session_state.ocr_engine else "⚠️"
    st.markdown(f'<div class="pills"><span class="pill pill-model">{s1} M1 YOLO</span><span class="pill pill-model">{s2} M2 Best</span><span class="pill pill-model">{s3} M3 Rambu</span><span class="pill pill-model">{so} OCR</span></div>',unsafe_allow_html=True)
    if st.session_state.m2_classes:
        st.caption(f"M2: {', '.join(sorted(st.session_state.m2_classes))}")
    enable_audio=st.checkbox("🔊 Audio",value=True)
    alert_cooldown=st.slider("Cooldown (s)",2,10,4)
    enable_tts=st.checkbox("🔊 TTS Baca Teks",value=True)
    show_logs=st.checkbox("📋 Logs",value=False)
    frame_skip=st.slider("Frame Skip",2,15,5)

# ─── TABS ───
tab1,tab2,tab3=st.tabs(["🎯 Detection","📖 Text Reading","📊 Statistics"])

with tab1:
    mode=st.radio("Mode:",["📹 Webcam","📤 Upload Video"],horizontal=True)
    st.divider()

    if mode=="📹 Webcam":
        st.markdown('<div class="alert-info">📸 Ambil foto → otomatis deteksi. Klik "Baca Teks" untuk OCR.</div>',unsafe_allow_html=True)
        fp=st.empty();sp=st.empty();wp=st.empty();op=st.empty()
        adp=st.empty();arp=st.empty();aocp=st.empty()
        c1,c2,c3=st.columns(3)
        with c1: md=st.empty()
        with c2: mb=st.empty()
        with c3: mm=st.empty()
        if show_logs: lp=st.expander("📋").empty()
        _,col_btn=st.columns(2)
        with col_btn: btn_ocr_cam=st.button("📖 Baca Teks",key="ocr_cam",use_container_width=True)
        cam=st.camera_input("📸 Ambil foto")
        if cam:
            if not st.session_state.model1 and not st.session_state.model2:
                st.error("⚠️ Load YOLO dulu!")
            else:
                from PIL import Image
                img=np.array(Image.open(cam))
                bgr=cv2.cvtColor(img,cv2.COLOR_RGB2BGR) if len(img.shape)==3 else img
                bgr=cv2.resize(bgr,(640,480)); orig=bgr.copy()
                ann,dets=detect_all(bgr,st.session_state.model1,st.session_state.model2,st.session_state.model3)
                fp.image(cv2.cvtColor(ann,cv2.COLOR_BGR2RGB),caption=f"{len(dets)} objek",use_container_width=True)
                for d in dets: st.session_state.detection_history.append(d)
                st.session_state.detection_history=st.session_state.detection_history[-500:]
                dl,rl=do_alerts(dets,enable_audio,alert_cooldown,wp,sp,adp,arp)
                md.metric("Objek",len(dets));mb.metric("⚠️",len(dl));mm.metric("Mode","📸")
                if btn_ocr_cam and st.session_state.ocr_engine:
                    text=ocr_read(orig,st.session_state.ocr_engine)
                    if text:
                        op.markdown(f'<div class="ocr-result">📝 {text}</div>',unsafe_allow_html=True)
                        if enable_tts:
                            a=tts(f"Ada tulisan: {text}")
                            if a: aocp.empty();play_audio(aocp,a)
                    else:
                        op.markdown('<div class="ocr-result" style="opacity:0.5">📝 Tidak ada teks</div>',unsafe_allow_html=True)
                if show_logs: lp.markdown('<br>'.join([f'[{t}] {m}' for t,m in st.session_state.log[:10]]),unsafe_allow_html=True)

    else:
        uploaded=st.file_uploader("Upload video",type=['mp4','avi','mov','mkv','webm'])
        fp=st.empty();sp=st.empty();wp=st.empty();op=st.empty()
        adp=st.empty();arp=st.empty();aocp=st.empty()
        c1,c2,c3=st.columns(3)
        with c1: md=st.empty()
        with c2: mb=st.empty()
        with c3: mf=st.empty()
        if show_logs: lp=st.expander("📋").empty()
        if uploaded:
            cn=uploaded.name
            if st.session_state.last_uploaded_name!=cn:
                st.session_state.ocr_triggered=False;st.session_state.ocr_frame_count=0
                st.session_state.last_uploaded_name=cn;st.session_state.last_ocr_text=''
                st.session_state.alerted_objects.clear()
        col1,col2=st.columns(2)
        with col1: btn_start=st.button("▶️ Start",use_container_width=True)
        with col2: btn_ocr=st.button("📖 Baca Teks",use_container_width=True)
        if btn_ocr:
            st.session_state.ocr_triggered=True;st.session_state.ocr_frame_count=0
            st.session_state.last_ocr_text='';st.info("🔍 OCR aktif.")
        if uploaded and btn_start:
            if not st.session_state.model1 and not st.session_state.model2:
                st.error("⚠️ Load YOLO!");st.stop()
            st.session_state.alerted_objects.clear()
            suf=os.path.splitext(uploaded.name)[1] or '.mp4'
            with tempfile.NamedTemporaryFile(delete=False,suffix=suf) as tmp:
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
                OCR_INTERVAL=int(fpsv*3) # scan OCR tiap 3 detik — dari Colab INTERVAL_OCR_DETIK=3
                while True:
                    ok,fr=cap.read()
                    if not ok: break
                    cnt+=1
                    if cnt%frame_skip!=0: continue
                    fr=cv2.resize(fr,(480,360))
                    orig=fr.copy()
                    sec=cnt/fpsv
                    prog.progress(min(cnt/max(tot,1),1.0),text=f"{cnt}/{tot} ({sec:.0f}s)")
                    ann,dets=detect_all(fr,m1,m2,m3)
                    fp.image(cv2.cvtColor(ann,cv2.COLOR_BGR2RGB),use_container_width=True)
                    st.session_state.last_frame=orig
                    for d in dets: st.session_state.detection_history.append(d)
                    st.session_state.detection_history=st.session_state.detection_history[-500:]
                    dl,rl=do_alerts(dets,enable_audio,alert_cooldown,wp,sp,adp,arp)
                    # OCR tiap 3 detik — dari Colab
                    if st.session_state.ocr_triggered and ocr_eng:
                        st.session_state.ocr_frame_count+=1
                        if st.session_state.ocr_frame_count%max(OCR_INTERVAL//frame_skip,1)==0:
                            text=ocr_read(orig,ocr_eng)
                            if text and len(text)>2:
                                if not _similar(text,st.session_state.last_ocr_text):
                                    st.session_state.last_ocr_text=text
                                    op.markdown(f'<div class="ocr-result">📝 {text}</div>',unsafe_allow_html=True)
                                    add_log(f"OCR: {text[:30]}")
                                    if enable_tts:
                                        a=tts(f"Ada tulisan: {text}")
                                        if a: aocp.empty();play_audio(aocp,a)
                    md.metric("Objek",len(dets));mb.metric("⚠️",len(dl))
                    el=time.time()-t0
                    mf.metric("FPS",f"{cnt/el:.1f}" if el>0 else "0")
                    if show_logs: lp.markdown('<br>'.join([f'[{t}] {m}' for t,m in st.session_state.log[:10]]),unsafe_allow_html=True)
                cap.release();prog.empty()
                st.success(f"✅ Selesai! {cnt} frame.")
                st.session_state.ocr_triggered=False
            except Exception as e: st.error(f"Error: {e}")
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
                text=ocr_read(arr,st.session_state.ocr_engine)
                if text:
                    rp.markdown(f'<div class="ocr-result">📝 {text}</div>',unsafe_allow_html=True)
                    if enable_tts:
                        a=tts(f"Ada tulisan: {text}")
                        if a: ap.empty();play_audio(ap,a)
                else:
                    rp.markdown('<div class="ocr-result" style="opacity:0.5">📝 Tidak ada teks</div>',unsafe_allow_html=True)
    else:
        ui=st.file_uploader("Upload gambar",type=['jpg','jpeg','png','bmp'])
        if ui:
            from PIL import Image
            img=Image.open(ui);arr=np.array(img)
            ip.image(img,use_container_width=True)
            if not st.session_state.ocr_engine: st.error("⚠️ Load OCR!")
            else:
                text=ocr_read(arr,st.session_state.ocr_engine)
                if text:
                    rp.markdown(f'<div class="ocr-result">📝 {text}</div>',unsafe_allow_html=True)
                    if enable_tts:
                        a=tts(f"Ada tulisan: {text}")
                        if a: ap.empty();play_audio(ap,a)
                else:
                    rp.markdown('<div class="ocr-result" style="opacity:0.5">📝 Tidak ada teks</div>',unsafe_allow_html=True)

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
        c3.markdown(f'<div class="stat-box"><div class="stat-value" style="color:#faad14">{w}</div><div class="stat-label">Waspada</div></div>',unsafe_allow_html=True)
        c4.markdown(f'<div class="stat-box"><div class="stat-value" style="color:#00c073">{am+rm}</div><div class="stat-label">Aman/Rambu</div></div>',unsafe_allow_html=True)
        df=pd.DataFrame([{
            'Waktu':d['timestamp'].strftime("%H:%M:%S"),
            'Objek':id_nama(d['class']),
            'Conf':f"{d['confidence']:.0%}",
            'Risk':d['risk_level'],
            'Model':d.get('source','?'),
        } for d in h[-100:]])
        st.dataframe(df,use_container_width=True)
        if st.button("🗑️ Hapus"):
            st.session_state.detection_history=[];st.session_state.alerted_objects.clear();st.rerun()
    else:
        st.info("📊 Belum ada data.")

st.divider()
st.markdown('<div style="text-align:center;color:#999;font-size:.8rem;padding:1rem 0"><strong>Asisten Navigasi Tunanetra v6.0</strong> • YOLOv11 • EasyOCR • gTTS</div>',unsafe_allow_html=True)
