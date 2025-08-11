import os
import json
import requests
from dotenv import load_dotenv
from paddleocr import PaddleOCR
from langchain.embeddings.base import Embeddings
from langchain.docstore.document import Document

load_dotenv()

CLOVA_EMBED_API_URL = os.getenv("CLOVA_EMBED_API_URL")
CLOVA_API_KEY = os.getenv("CLOVA_API_KEY")

class ClovaEmbeddings(Embeddings):
    def __init__(self, api_url: str, api_key: str, timeout: int = 30):
        self.api_url = api_url
        self.api_key = api_key
        self.timeout = timeout

    def _post_embed(self, texts):
        payload = {"inputs": texts}
        headers = {
            "Content-Type": "application/json",
            "X-NCP-CLOVASTUDIO-API-KEY": self.api_key
        }
        resp = requests.post(self.api_url, headers=headers, data=json.dumps(payload), timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()
        if "data" in data and "embeddings" in data["data"]:
            return data["data"]["embeddings"]
        if "embeddings" in data:
            return data["embeddings"]
        raise ValueError(f"임베딩 응답 포맷 오류: {data}")

    def embed_documents(self, texts):
        return self._post_embed(texts)

    def embed_query(self, text):
        return self._post_embed([text])[0]

def extract_text_from_image(img_path: str) -> str:
    ocr = PaddleOCR(use_angle_cls=True, lang='korean')
    result = ocr.ocr(img_path, cls=True)
    lines = []
    for res in result:
        if res is None:
            continue
        for line in res:
            txt = line[1][0]
            if txt:
                lines.append(txt)
    return "\n".join(lines).strip()

def build_document_from_image(img_path: str) -> Document:
    ocr_text = extract_text_from_image(img_path)
    if not ocr_text:
        ocr_text = f"filename: {os.path.basename(img_path)}"
    return Document(page_content=ocr_text, metadata={"source": img_path})
