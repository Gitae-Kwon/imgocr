# upload_debug.py
import streamlit as st

st.title("Upload debug")

# ① 탭/폼 없이 순수 업로더 (고유 key 지정)
files = st.file_uploader(
    "이미지 선택", type=["png","jpg","jpeg","webp"],
    accept_multiple_files=True, key="u_dbg1"
)
st.write("files is None? ->", files is None)
if files:
    st.write("len(files) ->", len(files))
    st.write([f.name for f in files])
    st.image(files[0])

# ② 사이드바 업로더 (브라우저 드래그 제약 우회)
sfiles = st.sidebar.file_uploader(
    "사이드바 업로더", type=["png","jpg","jpeg","webp"],
    accept_multiple_files=True, key="u_dbg2"
)
st.sidebar.write("sidebar files:", [f.name for f in sfiles] if sfiles else None)

# ③ 카메라 입력(파일 입력이 막혀있는지 확인)
cam = st.camera_input("카메라로 찍어서 올리기(테스트)", key="u_dbg3")
if cam:
    st.success(f"camera_input OK: {len(cam.getvalue())} bytes")
