# app.py
# -*- coding: utf-8 -*-
# Snap2Search — 이미지→텍스트(PaddleOCR)→임베딩→코사인 검색 (세션보존 업로드)

import os, json, uuid, tempfile, http.client, requests
import numpy as np
import streamlit as st
from io import BytesIO
from PIL import Image
import cv2

# ===================== Secrets =====================
CLOVA_API_KEY = st.secrets.get("CLOVA_API_KEY", "")  # 예: "Bearer nv-********"
CLOVA_HOST = st.secrets.get("CLOVA_HOST", "clovastudio.stream.ntruss.com")
if not CLOVA_API_KEY:
    st.error("❌ CLOVA_API_KEY를 Secrets에 설정하세요. 예) 'Bearer nv-***'")
    st.stop()

# ===================== 인덱스 경로 =================
INDEX_DIR = "index"
VEC_PATH = os.path.join(INDEX_DIR, "index.npy")    # (N, D)
META_PATH = os.path.join(INDEX_DIR, "meta.json")   # [{"source":..., "text":...}, ...]
os.makedirs(INDEX_DIR, exist_ok=True)

# ===================== 세션 상태 ====================
if "uploads" not in st.session_state:
    st.session_state["uploads"] = []  # [{"name":..., "type":..., "data": b"..."}]

def _store_upload():
    files = st.session_state.get("uploader_main")
    saved = []
    if files:
        for f in files:
            saved.append({"name": f.name, "type": f.type, "data": f.getvalue()})
    st.session_state["uploads"] = saved

# ===================== CLOVA v2 임베딩 ====================
def clova_embed(text: str) -> np.ndarray:
    headers = {
        "Content-Type": "application/json; charset=utf-8",
        "Authorization": CLOVA_API_KEY,
        "X-NCP-CLOVASTUDIO-REQUEST-ID": uuid.uuid4().hex,
    }
    payload = {"text": text}
    conn = http.client.HTTPSConnection(CLOVA_HOST)
    conn.request("POST", "/v1/api-tools/embedding/v2", json.dumps(payload), headers)
    res = conn.getresponse()
    data = json.loads(res.read().decode("utf-8"))
    conn.close()
    if data.get("status", {}).get("code") != "20000":
        raise RuntimeError(f"CLOVA embedding error: {data}")
    return np.asarray(data["result"]["embedding"], dtype=np.float32)

# ===================== PaddleOCR (한글) ====================
@st.cache_resource
def load_ocr():
    # 설치/로딩이 무거우므로 1회 캐시
    from paddleocr import PaddleOCR
    return PaddleOCR(use_angle_cls=True, lang="korean", use_gpu=False)

def preprocess_for_ocr(img: Image.Image) -> np.ndarray:
    """명암대비 강화 + 이진화 등 간단 전처리"""
    img = img.convert("RGB")
    arr = np.array(img)
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    # 노이즈 약간 제거 후 OTSU
    gray = cv2.medianBlur(gray, 3)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return th  # cv2 이미지

def paddle_ocr_extract(tmp_path: str) -> str:
    ocr = load_ocr()
    # PIL로 열어 전처리 → 임시 png 저장 → OCR
    pil = Image.open(tmp_path).convert("RGB")
    pre = preprocess_for_ocr(pil)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as t2:
        cv2.imwrite(t2.name, pre)
        in_path = t2.name
    result = ocr.ocr(in_path, cls=True)
    lines = []
    for blk in (result or []):
        for ln in (blk or []):
            txt = ln[1][0]
            if txt:
                lines.append(txt)
    return "\n".join(lines).strip()

# ===================== 인덱스 IO/검색 =====================
def load_index():
    if not (os.path.exists(VEC_PATH) and os.path.exists(META_PATH)):
        return None, []
    return np.load(VEC_PATH), json.load(open(META_PATH, "r", encoding="utf-8"))

def save_index(vecs: np.ndarray, meta: list):
    np.save(VEC_PATH, vecs)
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

def add_documents(docs: list):
    # docs: [{"source": <tmp_path>, "text": <ocr_text>}]
    new_vecs = []
    for d in docs:
        v = clova_embed(d["text"])
        v = v / (np.linalg.norm(v) + 1e-12)
        new_vecs.append(v)
    new_vecs = np.stack(new_vecs, axis=0)

    old_vecs, old_meta = load_index()
    if old_vecs is None:
        save_index(new_vecs, docs)
    else:
        save_index(np.vstack([old_vecs, new_vecs]), old_meta + docs)

def search(query: str, k: int = 5):
    vecs, meta = load_index()
    if vecs is None or len(meta) == 0:
        return []
    q = clova_embed(query)
    q = q / (np.linalg.norm(q) + 1e-12)
    sims = vecs @ q
    idx = np.argsort(-sims)[:k]
    return [(float(sims[i]), meta[i]) for i in idx]

# ===================== UI ================================
st.set_page_config(page_title="Snap2Search (PaddleOCR)", page_icon="📷", layout="wide")
st.title("📷 Snap2Search — 이미지→텍스트(PaddleOCR)→임베딩→코사인 검색")

tab1, tab2 = st.tabs(["📥 인덱스 만들기", "🔍 검색하기"])

with tab1:
    st.subheader("이미지 업로드 & 인덱싱")
    st.caption("JPG/PNG/WEBP 권장 (고해상도도 OK)")

    files_widget = st.file_uploader(
        "이미지 업로드 (여러 장)",
        type=["jpg","jpeg","png","webp"],
        accept_multiple_files=True,
        key="uploader_main",
        on_change=_store_upload
    )

    st.caption(f"업로드 된 파일 수: {len(st.session_state['uploads'])}")
    if st.session_state["uploads"]:
        st.info([x["name"] for x in st.session_state["uploads"]][:10])

    if st.button("인덱싱 실행", use_container_width=True):
        if not st.session_state["uploads"]:
            st.warning("먼저 이미지를 업로드하세요.")
        else:
            docs = []
            with st.spinner("🔎 OCR 및 임베딩 중... (첫 실행은 모델 로딩으로 수초 소요)"):
                for up in st.session_state["uploads"]:
                    suffix = os.path.splitext(up["name"])[1] or ".jpg"
                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                        tmp.write(up["data"])
                        tmp_path = tmp.name
                    text = paddle_ocr_extract(tmp_path)
                    if not text:
                        text = f"filename: {os.path.basename(tmp_path)}"
                    docs.append({"source": tmp_path, "text": text})
                add_documents(docs)

            st.success(f"✅ {len(docs)}개 이미지 인덱싱 완료")
            with st.expander("추출된 텍스트 미리보기"):
                for d in docs:
                    st.write(f"**{os.path.basename(d['source'])}**")
                    st.code(d["text"][:800] + ("..." if len(d["text"]) > 800 else ""))

with tab2:
    st.subheader("자연어로 검색")
    q = st.text_input("예: '매일 두유 99.9', '설탕 무첨가 9.0g'")
    k = st.slider("결과 수 (k)", 1, 10, 5)
    cols = st.columns(2)
    with cols[0]:
        if st.button("인덱스 초기화", use_container_width=True):
            try:
                if os.path.exists(VEC_PATH): os.remove(VEC_PATH)
                if os.path.exists(META_PATH): os.remove(META_PATH)
                st.success("🧹 인덱스 초기화 완료")
            except Exception as e:
                st.error("초기화 중 오류")
                st.exception(e)

    if st.button("검색 실행", use_container_width=True):
        if not q.strip():
            st.warning("검색어를 입력하세요.")
        else:
            try:
                with st.spinner("🔍 검색 중..."):
                    results = search(q, k=k)
                if not results:
                    st.info("인덱스가 비었거나 결과가 없습니다.")
                else:
                    for score, m in results:
                        st.markdown("---")
                        if os.path.exists(m["source"]):
                            st.image(m["source"], caption=os.path.basename(m["source"]), use_column_width=True)
                        st.caption(f"유사도: {score:.4f}")
                        st.code(m["text"][:800] + ("..." if len(m["text"]) > 800 else ""))
            except Exception as e:
                st.error("검색 중 오류가 발생했습니다.")
                st.exception(e)
