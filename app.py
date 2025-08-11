# app.py
# -*- coding: utf-8 -*-
# Snap2Search — Google Vision (OCR + Label + Web Best-Guess) → CLOVA Embedding v2 → 코사인 검색

import os, json, uuid, tempfile, http.client
import numpy as np
import streamlit as st
from PIL import Image

# ===================== Secrets =====================
CLOVA_API_KEY = st.secrets.get("CLOVA_API_KEY", "")
CLOVA_HOST = st.secrets.get("CLOVA_HOST", "clovastudio.stream.ntruss.com")
GCP_SA_INFO = st.secrets.get("gcp_service_account", None)

if not CLOVA_API_KEY:
    st.error("❌ Secrets에 CLOVA_API_KEY가 필요합니다. 예) 'Bearer nv-***'")
    st.stop()
if not GCP_SA_INFO:
    st.error("❌ Secrets에 [gcp_service_account] 블록이 필요합니다.")
    st.stop()

# ===================== 인덱스 경로 =================
INDEX_DIR = "index"
VEC_PATH = os.path.join(INDEX_DIR, "index.npy")
META_PATH = os.path.join(INDEX_DIR, "meta.json")
os.makedirs(INDEX_DIR, exist_ok=True)

# ===================== 세션 상태 ====================
if "uploads" not in st.session_state:
    st.session_state["uploads"] = []  # [{"name":..., "type":..., "data": b"..."}]

def _store_upload_from_form(files):
    st.session_state["uploads"] = []
    if files:
        for f in files:
            st.session_state["uploads"].append(
                {"name": f.name, "type": f.type, "data": f.getvalue()}
            )

# ===================== Google Vision =====================
@st.cache_resource
def get_vision_client():
    from google.cloud import vision
    return vision.ImageAnnotatorClient.from_service_account_info(dict(st.secrets["gcp_service_account"]))

def _vision_image_from_path(path):
    from google.cloud import vision
    with open(path, "rb") as f:
        return vision.Image(content=f.read())

def vision_ocr_extract(path, use_document=True):
    """문서/영수증은 document_text_detection이 더 정확."""
    from google.cloud import vision
    client = get_vision_client()
    image = _vision_image_from_path(path)
    ctx = vision.ImageContext(language_hints=["ko", "en"])
    resp = client.document_text_detection(image=image, image_context=ctx) if use_document \
        else client.text_detection(image=image, image_context=ctx)
    if resp.error.message:
        return "", f"(OCR 오류) {resp.error.message}"
    ann = getattr(resp, "full_text_annotation", None)
    return (ann.text or "").strip() if ann else "", None

def vision_labels_and_caption(path, max_labels=5):
    """라벨, 웹 베스트 게스(제목 느낌), 간단 캡션 생성."""
    from google.cloud import vision
    client = get_vision_client()
    image = _vision_image_from_path(path)

    # 1) 라벨 감지
    lr = client.label_detection(image=image)
    labels = [l.description for l in (lr.label_annotations or [])][:max_labels]

    # 2) Web detection: best guess labels (제목 비슷한 텍스트)
    wr = client.web_detection(image=image)
    best_guess = ""
    if wr and wr.web_detection and wr.web_detection.best_guess_labels:
        best_guess = wr.web_detection.best_guess_labels[0].label or ""

    # 3) 간단 캡션 생성 규칙
    caption = ""
    if best_guess:
        caption = best_guess
    elif labels:
        # 가장 상위 라벨 1~3개로 한글 캡션 흉내
        head = " · ".join(labels[:3])
        caption = f"이 사진에는 {head} 등이 보입니다."
    return labels, best_guess, caption

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

# ===================== 인덱스 IO / 검색 ===================
def load_index():
    if not (os.path.exists(VEC_PATH) and os.path.exists(META_PATH)):
        return None, []
    return np.load(VEC_PATH), json.load(open(META_PATH, "r", encoding="utf-8"))

def save_index(vecs: np.ndarray, meta: list):
    np.save(VEC_PATH, vecs)
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

def add_documents(docs: list):
    # docs: [{"source":..., "text":..., "labels": [...], "caption": "..."}]
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
st.set_page_config(page_title="Snap2Search (Vision OCR + Label + Caption)", page_icon="📷", layout="wide")
st.title("📷 Snap2Search — Vision OCR + 라벨/캡션 → CLOVA 임베딩 → 코사인 검색")

tab1, tab2 = st.tabs(["📥 인덱스 만들기", "🔍 검색하기"])

with tab1:
    st.subheader("이미지 업로드 & 인덱싱")

    with st.form("index_form", clear_on_submit=False):
        files = st.file_uploader(
            "이미지 업로드 (여러 장)",
            type=["jpg","jpeg","png","webp","heic","HEIC","pdf"],
            accept_multiple_files=True,
            key="uploader_main_form",
        )
        colA, colB = st.columns(2)
        with colA:
            use_document = st.checkbox("문서 최적화 OCR(document_text_detection)", True)
        with colB:
            include_labels = st.checkbox("OCR 없으면 라벨/캡션으로 대체", True)
        submitted = st.form_submit_button("업로드 확정")

    if submitted:
        _store_upload_from_form(files)

    st.caption(f"업로드 된 파일 수: {len(st.session_state.get('uploads', []))}")
    if st.session_state.get("uploads"):
        st.info([x["name"] for x in st.session_state["uploads"]][:10])

    if st.button("인덱싱 실행", use_container_width=True, key="do_index"):
        if not st.session_state.get("uploads"):
            st.warning("먼저 이미지를 업로드하고 '업로드 확정'을 눌러주세요.")
        else:
            docs = []
            with st.spinner("🔎 Vision 분석 및 임베딩 중..."):
                for up in st.session_state["uploads"]:
                    suffix = os.path.splitext(up["name"])[1] or ".jpg"
                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                        tmp.write(up["data"])
                        tmp_path = tmp.name

                    # 1) OCR
                    ocr_text, ocr_err = vision_ocr_extract(tmp_path, use_document=use_document)
                    # 2) 라벨/캡션
                    labels, best_guess, caption = vision_labels_and_caption(tmp_path, max_labels=5)

                    # 3) 최종 텍스트 만들기
                    if ocr_text.strip():
                        final_text = ocr_text
                        enrich = []
                        if labels:
                            enrich.append("Labels: " + ", ".join(labels))
                        if best_guess:
                            enrich.append("BestGuess: " + best_guess)
                        if enrich:
                            final_text += "\n\n" + "\n".join(enrich)
                    else:
                        if include_labels:
                            # OCR이 비었으면 캡션/라벨로 대체
                            fallback = caption or best_guess or ", ".join(labels) or f"filename: {os.path.basename(tmp_path)}"
                            final_text = fallback
                        else:
                            final_text = f"filename: {os.path.basename(tmp_path)}"

                    docs.append({
                        "source": tmp_path,
                        "text": final_text,
                        "labels": labels,
                        "caption": caption,
                        "best_guess": best_guess
                    })

                add_documents(docs)

            st.success(f"✅ {len(docs)}개 이미지 인덱싱 완료")
            with st.expander("추출 텍스트 미리보기"):
                for d in docs:
                    st.write(f"**{os.path.basename(d['source'])}**")
                    st.code(d["text"][:1000] + ("..." if len(d["text"]) > 1000 else ""))

with tab2:
    st.subheader("자연어로 검색")
    q = st.text_input("예: '빨간 사과', '영수증 총 금액', '나이키 로고 있는 사진'")
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
