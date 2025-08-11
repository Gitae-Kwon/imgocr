# app.py
# -*- coding: utf-8 -*-
# Snap2Search — Vision OCR + 룰기반 캡션 → CLOVA Embedding v2 → 코사인 검색

import os, json, uuid, tempfile, http.client
import numpy as np
import streamlit as st

# ===== Secrets =====
CLOVA_API_KEY = st.secrets.get("CLOVA_API_KEY", "")
CLOVA_HOST = st.secrets.get("CLOVA_HOST", "clovastudio.stream.ntruss.com")
GCP_SA = st.secrets.get("gcp_service_account", None)
if not CLOVA_API_KEY:
    st.error("❌ Secrets에 CLOVA_API_KEY 필요 (예: 'Bearer nv-***').")
    st.stop()
if not GCP_SA:
    st.error("❌ Secrets에 [gcp_service_account] 블록 필요.")
    st.stop()

# ===== Paths =====
INDEX_DIR = "index"
VEC_PATH = os.path.join(INDEX_DIR, "index.npy")
META_PATH = os.path.join(INDEX_DIR, "meta.json")
os.makedirs(INDEX_DIR, exist_ok=True)

# ===== Session =====
if "uploads" not in st.session_state:
    st.session_state["uploads"] = []  # [{"name":..., "type":..., "data": b"..."}]

def _store_upload_from_form(files):
    st.session_state["uploads"] = []
    if files:
        for f in files:
            st.session_state["uploads"].append(
                {"name": f.name, "type": f.type, "data": f.getvalue()}
            )

# ===== Google Vision =====
@st.cache_resource
def vision_client():
    from google.cloud import vision
    return vision.ImageAnnotatorClient.from_service_account_info(dict(GCP_SA))

def _vimg_from_path(path):
    from google.cloud import vision
    with open(path, "rb") as f:
        return vision.Image(content=f.read())

def vision_ocr_text(path, use_document=True):
    from google.cloud import vision
    c = vision_client()
    img = _vimg_from_path(path)
    ctx = vision.ImageContext(language_hints=["ko", "en"])
    resp = c.document_text_detection(image=img, image_context=ctx) if use_document \
        else c.text_detection(image=img, image_context=ctx)
    if resp.error.message:
        return ""
    ann = getattr(resp, "full_text_annotation", None)
    return (ann.text or "").strip() if ann else ""

def vision_rule_caption(path, max_labels=6):
    """라벨/객체/색상/웹베스트게스를 조합해 자연스러운 1~3문장 생성 (무료)."""
    from google.cloud import vision
    c = vision_client()
    img = _vimg_from_path(path)

    # 라벨
    lr = c.label_detection(image=img)
    labels = [l.description for l in (lr.label_annotations or [])][:max_labels]

    # 객체
    or_ = c.object_localization(image=img)
    objs = [(o.name, o.score) for o in (or_.localized_object_annotations or [])]
    objs = [n for n, s in sorted(objs, key=lambda x: -x[1])][:3]

    # 웹 베스트게스
    wr = c.web_detection(image=img)
    best = ""
    if wr and wr.web_detection and wr.web_detection.best_guess_labels:
        best = wr.web_detection.best_guess_labels[0].label or ""

    # 대표 색상
    pr = c.image_properties(image=img)
    colors = []
    if pr and pr.image_properties_annotation:
        for ci in pr.image_properties_annotation.dominant_colors.colors[:2]:
            r, g, b = int(ci.color.red), int(ci.color.green), int(ci.color.blue)
            colors.append(f"#{r:02x}{g:02x}{b:02x}")

    # 문장 조합
    subject = best or (objs[0] if objs else (labels[0] if labels else "장면"))
    extras = [x for x in labels if subject.lower() not in x.lower()][:3]
    s1 = f"이 이미지는 '{subject}'를 중심으로 한 장면입니다."
    s2 = f" 주변 요소로 {', '.join(extras)}가 보입니다." if extras else ""
    s3 = f" 주요 색상은 {', '.join(colors)}입니다." if colors else ""
    return (s1 + s2 + s3).strip()

# ===== CLOVA Embedding v2 =====
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

# ===== Index I/O & Search =====
def load_index():
    if not (os.path.exists(VEC_PATH) and os.path.exists(META_PATH)):
        return None, []
    return np.load(VEC_PATH), json.load(open(META_PATH, "r", encoding="utf-8"))

def save_index(vecs: np.ndarray, meta: list):
    np.save(VEC_PATH, vecs)
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

def add_documents(docs: list):
    new_vecs = []
    for d in docs:
        v = clova_embed(d["text"])
        v = v / (np.linalg.norm(v) + 1e-12)  # 코사인용 정규화
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

# ===== UI =====
st.set_page_config(page_title="Snap2Search (Vision 무료 설명)", page_icon="📷", layout="wide")
st.title("📷 Snap2Search — Vision OCR + 무료 캡션 → CLOVA 임베딩 → 코사인 검색")

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
            use_doc = st.checkbox("문서 최적화 OCR(document_text_detection)", True)
        with colB:
            use_caption = st.checkbox("OCR이 없으면 무료 캡션 사용", True)
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

                    ocr_text = vision_ocr_text(tmp_path, use_document=use_doc).strip()
                    if ocr_text:
                        final_text = ocr_text
                    else:
                        final_text = vision_rule_caption(tmp_path) if use_caption \
                            else f"filename: {os.path.basename(tmp_path)}"

                    docs.append({
                        "source": tmp_path,
                        "text": final_text
                    })

                add_documents(docs)

            st.success(f"✅ {len(docs)}개 이미지 인덱싱 완료")
            with st.expander("추출 텍스트 미리보기"):
                for d in docs:
                    st.write(f"**{os.path.basename(d['source'])}**")
                    st.code(d["text"][:1000] + ("..." if len(d["text"]) > 1000 else ""))

with tab2:
    st.subheader("자연어로 검색")
    q = st.text_input("예: '빨간 사과', '영수증 총 금액', '밤하늘의 보름달'")
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
