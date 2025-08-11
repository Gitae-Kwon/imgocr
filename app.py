# app.py
# -*- coding: utf-8 -*-
import os, json, uuid, tempfile, http.client, requests
import numpy as np
import streamlit as st
from PIL import Image

# -----------------------------
# Secrets
# -----------------------------
CLOVA_API_KEY = st.secrets.get("CLOVA_API_KEY", "")
CLOVA_HOST = st.secrets.get("CLOVA_HOST", "clovastudio.stream.ntruss.com")
OCR_SPACE_API_KEY = st.secrets.get("OCR_SPACE_API_KEY", "")

if not CLOVA_API_KEY:
    st.error("❌ CLOVA_API_KEY를 Secrets에 설정하세요.")
    st.stop()

INDEX_DIR = "index"
VEC_PATH = os.path.join(INDEX_DIR, "index.npy")    # (N, D)
META_PATH = os.path.join(INDEX_DIR, "meta.json")   # [{"source":..., "text":...}, ...]

os.makedirs(INDEX_DIR, exist_ok=True)

# -----------------------------
# CLOVA v2 임베딩
# -----------------------------
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
    vec = np.array(data["result"]["embedding"], dtype=np.float32)
    return vec

# -----------------------------
# OCR.space
# -----------------------------
def extract_text_from_image(path: str) -> str:
    key = OCR_SPACE_API_KEY or "helloworld"
    try:
        with open(path, "rb") as f:
            r = requests.post(
                "https://api.ocr.space/parse/image",
                files={"filename": f},
                data={"apikey": key, "language": "kor"},
                timeout=90,
            )
        r.raise_for_status()
        data = r.json()
        if data.get("IsErroredOnProcessing"):
            msg = data.get("ErrorMessage") or data.get("ErrorDetails")
            return f"(OCR 실패) {msg}"
        prs = data.get("ParsedResults") or []
        return (prs[0].get("ParsedText") if prs else "") or ""
    except Exception as e:
        return f"(OCR 예외) {e}"

# -----------------------------
# 인덱스 IO
# -----------------------------
def load_index():
    if not (os.path.exists(VEC_PATH) and os.path.exists(META_PATH)):
        return None, []
    vecs = np.load(VEC_PATH)
    with open(META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return vecs, meta

def save_index(vecs: np.ndarray, meta: list):
    np.save(VEC_PATH, vecs)
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

def add_documents(docs: list):
    """
    docs: [{"source": <path>, "text": <str>}]
    """
    new_vecs = []
    for d in docs:
        v = clova_embed(d["text"])
        # 정규화(코사인 유사도용)
        n = np.linalg.norm(v) + 1e-12
        new_vecs.append(v / n)
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

    # 코사인 유사도 = q dot vecs^T (정규화되어 있으므로 내적)
    sims = vecs @ q
    idx = np.argsort(-sims)[:k]
    return [(float(sims[i]), meta[i]) for i in idx]

# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="Snap2Search (초경량)", page_icon="📷", layout="wide")
st.title("📷 Snap2Search — 이미지→텍스트(OCR)→임베딩→코사인 검색 (초경량)")

tab1, tab2 = st.tabs(["📥 인덱스 만들기", "🔍 검색하기"])

with tab1:
    st.subheader("이미지 업로드 & 인덱싱")
    files = st.file_uploader("이미지 업로드", type=["png","jpg","jpeg","webp"], accept_multiple_files=True)
    if st.button("인덱싱 실행", use_container_width=True):
        if not files:
            st.warning("파일을 업로드하세요.")
        else:
            docs, tmp_paths = [], []
            for f in files:
                suffix = os.path.splitext(f.name)[1]
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(f.read())
                    tmp_paths.append(tmp.name)

            with st.spinner("🔎 OCR 및 임베딩 중..."):
                for p in tmp_paths:
                    txt = extract_text_from_image(p).strip()
                    if not txt:
                        txt = f"filename: {os.path.basename(p)}"
                    docs.append({"source": p, "text": txt})
                add_documents(docs)
            st.success(f"✅ {len(docs)}개 이미지 인덱싱 완료")
            with st.expander("추출 텍스트 미리보기"):
                for d in docs:
                    st.write(f"**{os.path.basename(d['source'])}**")
                    st.code(d["text"][:800] + ("..." if len(d["text"]) > 800 else ""))

with tab2:
    st.subheader("자연어로 검색")
    q = st.text_input("예: '영수증 총 결제 금액' / '빨간 나이키 신발'")
    k = st.slider("결과 수 (k)", 1, 10, 5)
    if st.button("검색 실행", use_container_width=True):
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
