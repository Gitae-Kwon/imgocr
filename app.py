# app.py
# -*- coding: utf-8 -*-
import os
import json
import uuid
import tempfile
import http.client

import streamlit as st
from PIL import Image
from dotenv import load_dotenv

from paddleocr import PaddleOCR
from langchain.embeddings.base import Embeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document

# -----------------------------
# 환경 변수 로드
# -----------------------------
load_dotenv()
CLOVA_HOST = os.getenv("CLOVA_HOST", "clovastudio.stream.ntruss.com")
CLOVA_API_KEY = os.getenv("nv-b886542b94514d4ba1a90e9149bec488yApG")  # 예: Bearer <api-key>

if not CLOVA_API_KEY:
    st.error("❌ 환경변수 CLOVA_API_KEY가 설정되어 있지 않습니다. (.env 또는 Streamlit Cloud Secrets에 설정하세요)")
    st.stop()

INDEX_DIR = "faiss_index"

# -----------------------------
# CLOVA v2 임베딩 실행기 (네가 준 코드 기반)
# -----------------------------
class CompletionExecutor:
    def __init__(self, host, api_key, request_id):
        self._host = host
        self._api_key = api_key
        self._request_id = request_id

    def _send_request(self, completion_request):
        headers = {
            'Content-Type': 'application/json; charset=utf-8',
            'Authorization': self._api_key,
            'X-NCP-CLOVASTUDIO-REQUEST-ID': self._request_id
        }
        conn = http.client.HTTPSConnection(self._host)
        conn.request('POST', '/v1/api-tools/embedding/v2', json.dumps(completion_request), headers)
        response = conn.getresponse()
        result = json.loads(response.read().decode(encoding='utf-8'))
        conn.close()
        return result

    def execute(self, completion_request):
        res = self._send_request(completion_request)
        if isinstance(res, dict) and 'status' in res and res['status'].get('code') == '20000':
            return res.get('result')  # 보통 {"embedding":[...]} 형태
        else:
            # 에러 메시지 표출을 위해 원문 반환
            return {'error': res}

# -----------------------------
# LangChain Embeddings 어댑터(v2)
# -----------------------------
class ClovaV2Embeddings(Embeddings):
    """
    CLOVA Studio 임베딩 v2 엔드포인트를 LangChain Embeddings로 감싸는 어댑터.
    NOTE: v2는 요청 포맷이 {"text": "..."} (단건) 기준이므로 여기선 단건 호출 루프.
    """
    def __init__(self, host: str, api_key: str):
        self.host = host
        self.api_key = api_key

    def _embed_one(self, text: str):
        # 요청마다 request_id 새로 생성 (권장)
        executor = CompletionExecutor(
            host=self.host,
            api_key=self.api_key,
            request_id=str(uuid.uuid4()).replace("-", "")
        )
        payload = {"text": text}
        res = executor.execute(payload)
        if isinstance(res, dict) and "embedding" in res:
            return res["embedding"]
        # 혹시 스키마가 다르면 에러 핸들
        if isinstance(res, dict) and "error" in res:
            raise RuntimeError(f"CLOVA v2 embedding error: {res['error']}")
        raise RuntimeError(f"Unexpected embedding response: {res}")

    def embed_query(self, text: str):
        return self._embed_one(text)

    def embed_documents(self, texts):
        # 간단히 순차 호출 (대량이면 멀티프로세싱/배치 권장)
        return [self._embed_one(t) for t in texts]

# -----------------------------
# OCR 함수 (PaddleOCR)
# -----------------------------
@st.cache_resource
def get_ocr():
    # 한국어 문서 위주이면 lang='korean'
    return PaddleOCR(use_angle_cls=True, lang='korean')

def extract_text_from_image(tmp_path: str) -> str:
    ocr = get_ocr()
    result = ocr.ocr(tmp_path, cls=True)
    lines = []
    for res in result:
        if res is None:
            continue
        for line in res:
            txt = line[1][0]
            if txt:
                lines.append(txt)
    return "\n".join(lines).strip()

def build_document_from_image(tmp_path: str) -> Document:
    text = extract_text_from_image(tmp_path)
    if not text:
        text = f"filename: {os.path.basename(tmp_path)}"
    return Document(page_content=text, metadata={"source": tmp_path})

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Snap2Search (CLOVA v2)", page_icon="📷", layout="wide")
st.title("📷 Snap2Search — 이미지→텍스트(OCR)→임베딩(v2)→FAISS 검색")
st.caption("CLOVA Studio Embedding v2 + PaddleOCR + LangChain + FAISS")

embeddings = ClovaV2Embeddings(host=CLOVA_HOST, api_key=CLOVA_API_KEY)

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
                    st.code(d.page_content[:500] + ("..." if len(d.page_content) > 500 else ""))

with tab2:
    st.subheader("자연어로 검색")
    query = st.text_input("예: '영수증에서 결제 금액 32,000원' / '빨간 나이키 신발'")
    topk = st.slider("가져올 결과 수 (k)", min_value=1, max_value=10, value=5)
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
