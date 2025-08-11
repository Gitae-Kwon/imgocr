import os
import tempfile
import streamlit as st
from langchain_community.vectorstores import FAISS
from utils import ClovaEmbeddings, build_document_from_image
from dotenv import load_dotenv

load_dotenv()

CLOVA_EMBED_API_URL = os.getenv("CLOVA_EMBED_API_URL")
CLOVA_API_KEY = os.getenv("CLOVA_API_KEY")

if not CLOVA_EMBED_API_URL or not CLOVA_API_KEY:
    st.error("❌ .env 파일에서 CLOVA API 정보를 설정해주세요.")
    st.stop()

embeddings = ClovaEmbeddings(api_url=CLOVA_EMBED_API_URL, api_key=CLOVA_API_KEY)
INDEX_DIR = "faiss_index"

st.title("📷 이미지 기반 RAG 검색 데모")
st.caption("이미지 → 텍스트(OCR) 변환 → CLOVA 임베딩 → FAISS 검색")

tab1, tab2 = st.tabs(["📥 인덱스 만들기", "🔍 검색하기"])

with tab1:
    st.header("이미지 업로드 & 인덱싱")
    uploaded_files = st.file_uploader("이미지를 업로드하세요", type=["png", "jpg", "jpeg"], accept_multiple_files=True)
    if st.button("인덱싱 실행") and uploaded_files:
        docs = []
        for file in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.name)[1]) as tmp:
                tmp.write(file.read())
                tmp_path = tmp.name
            doc = build_document_from_image(tmp_path)
            docs.append(doc)

        if os.path.exists(INDEX_DIR):
            db = FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)
            db.add_documents(docs)
        else:
            db = FAISS.from_documents(docs, embeddings)
        db.save_local(INDEX_DIR)
        st.success(f"{len(docs)}개의 이미지가 인덱싱되었습니다.")

with tab2:
    st.header("이미지 검색")
    query = st.text_input("검색어를 입력하세요")
    if st.button("검색 실행") and query:
        if not os.path.exists(INDEX_DIR):
            st.error("❌ 먼저 이미지를 인덱싱하세요.")
        else:
            db = FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)
            results = db.similarity_search(query, k=5)
            for r in results:
                st.image(r.metadata["source"], caption=r.page_content[:100] + "...")
