# app.py
# -*- coding: utf-8 -*-
import os
import json
import uuid
import tempfile
import http.client
import requests

import streamlit as st
from PIL import Image

from langchain.embeddings.base import Embeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document

# -----------------------------
# 환경변수: Streamlit Secrets에서 읽기
# -----------------------------
CLOVA_API_KEY = st.secrets.get("CLOVA_API_KEY", "")
CLOVA_HOST = st.secrets.get("CLOVA_HOST", "clovastudio.stream.ntruss.com")
OCR_SPACE_API_KEY = st.secrets.get("OCR_SPACE_API_KEY", "")

if not CLOVA_API_KEY:
    st.error("❌ CLOVA_API_KEY가 secrets.toml에 설정되어야 합니다.")
    st.stop()

# -----------------------------
# CLOVA v2 임베딩 실행기
# -----------------------------
class CompletionExecutor:
    def __init__(self, host: str, api_key: str, request_id: str):
        self._host = host
        self._api_key = api_key
        self._request_id = request_id

    def _send_request(self, completion_request: dict):
        headers = {
            "Content-Type": "application/json; charset=utf-8",
            "Authorization": self._api_key,
            "X-NCP-CLOVASTUDIO-REQUEST-ID": self._request_id,
        }
        conn = http.client.HTTPSConnection(self._host)
        conn.request("POST", "/v1/api-tools/embedding/v2", json.dumps(completion_request), headers)
        response = conn.getresponse()
        result = json.loads(response.read().decode("utf-8"))
        conn.close()
        return result

    def execute(self, completion_request: dict):
        res = self._send_request(completion_request)
        if isinstance(res, dict) and res.get("status", {}).get("code") == "20000":
            return res.get("result")
        return {"error": res}

class ClovaV2Embeddings(Embeddings):
    """LangChain Embeddings 어댑터 (v2, 단건 호출)"""
    def __init__(self, host: str, api_key: str):
        self.host = host
        self.api_key = api_key

    def _embed_one(self, text: str):
        executor = CompletionExecutor(
            host=self.host,
            api_key=self.api_key,
            request_id=uuid.uuid4().hex
        )
        payload = {"text": text}
        res = executor.execute(payload)
        if "error" in res:
            raise RuntimeError(f"CLOVA v2 embedding error: {res['error']}")
        if "embedding" in res:
            return res["embedding"]
        raise RuntimeError(f"Unexpected embedding response: {res}")

    def embed_query(self, text: str):
        return self._embed_one(text)

    def embed_documents(self, texts):
        return [self._embed_one(t) for t in texts]

# -----------------------------
# OCR.space 기반 OCR
# -----------------------------
def extract_text_from_image(tmp_path: str) -> str:
    key = OCR_SPACE_API_KEY or "helloworld"  # demo 키
    try:
        with open(tmp_path, "rb") as f:
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
        results = data.get("ParsedResults")
        if not results:
            return "(OCR 결과 없음)"
        return (results[0].get("ParsedText") or "").strip()
    except Exception as e:
        return f"(OCR 예외) {e}"

def build_document_from_image(tmp_path: str) -> Document:
    text = extract_text_from_image(tmp_path)
    if not text:
        text = f"filename: {os.path.basename(tmp_path)}"
    return Document(page_content=text, metadata={"source": tmp_path})

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Snap2Search (OCR.space + CLOVA v2)", page_icon="📷", layout="wide")
st.title("📷 Snap2Search — 이미지→텍스트(OCR)→임베딩(v2)→FAISS 검색")
st.caption("OCR.space + CLOVA Studio Embedding v2 + LangChain + FAISS")

# 임베딩 준비
try:
    embeddings = ClovaV2Embeddings(host=CLOVA_HOST, api_key=CLOVA_API_KEY)
except Exception as e:
    st.error("❌ CLOVA 임베딩 초기화 실패. secrets.toml 내용을 확인하세요.")
    st.exception(e)
    st.stop()

INDEX_DIR = "faiss_index"

tab1, tab2 = st.tabs(["📥 인덱스 만들기", "🔍 검색하기"])

with tab1:
    st.subheader("이미지 업로드 & 인덱싱")
    uploaded_files = st.file_uploader(
        "이미지를 업로드하세요 (여러 장 가능)", type=["png", "jpg", "jpeg", "webp"], accept_multiple_files=True
    )
    if st.button("인덱싱 실행", use_container_width=True):
        if not uploaded_files:
            st.warning("먼저 이미지를 업로드하세요.")
        else:
            docs = []
            tmp_paths = []
            for f in uploaded_files:
                suffix = os.path.splitext(f.name)[1]
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(f.read())
                    tmp_paths.append(tmp.name)

            with st.spinner("🔎 OCR 처리 및 임베딩 중..."):
                for p in tmp_paths:
                    doc = build_document_from_image(p)
                    docs.append(doc)

                if os.path.exists(INDEX_DIR):
                    db = FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)
                    db.add_documents(docs)
                else:
                    db = FAISS.from_documents(docs, embeddings)
                db.save_local(INDEX_DIR)

            st.success(f"✅ {len(docs)}개 이미지 인덱싱 완료")
            with st.expander("추출된 텍스트 미리보기"):
                for d in docs:
                    st.write(f"**{os.path.basename(d.metadata['source'])}**")
                    st.code(d.page_content[:800] + ("..." if len(d.page_content) > 800 else ""))

with tab2:
    st.subheader("자연어로 검색")
    query = st.text_input("예: '영수증에서 총 결제 금액' / '빨간 나이키 신발' 등")
    topk = st.slider("가져올 결과 수 (k)", 1, 10, 5)
    if st.button("검색 실행", use_container_width=True):
        if not os.path.exists(INDEX_DIR):
            st.error("❌ 먼저 인덱스를 생성하세요.")
        elif not query.strip():
            st.warning("검색어를 입력하세요.")
        else:
            with st.spinner("🔍 유사도 검색 중..."):
                db = FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)
                results = db.similarity_search(query, k=topk)

            if not results:
                st.info("검색 결과가 없습니다.")
            else:
                for r in results:
                    st.markdown("---")
                    src = r.metadata.get("source")
                    if src and os.path.exists(src):
                        st.image(src, caption=os.path.basename(src), use_column_width=True)
                    st.caption("추출 텍스트 (일부)")
                    st.code(r.page_content[:800] + ("..." if len(r.page_content) > 800 else ""))
