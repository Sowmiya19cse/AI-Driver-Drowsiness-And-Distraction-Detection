import streamlit as st

st.set_page_config(page_title="Camera Test")

st.title("Camera Test")

st.write("Click the camera button below.")

picture = st.camera_input("Take a picture")

if picture is not None:
    st.success("Camera is working!")
    st.image(picture)
else:
    st.warning("Camera is not available.")