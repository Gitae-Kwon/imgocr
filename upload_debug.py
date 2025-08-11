import streamlit as st

st.title("Upload debug")
files = st.file_uploader("이미지", type=["png","jpg","jpeg","webp"],
                         accept_multiple_files=True, key="u_dbg")

st.write("files is None? ->", files is None)
if files:
    st.write("len(files) ->", len(files))
    st.write([f.name for f in files][:5])
    st.image(files[0])
